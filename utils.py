"""
公共工具：随机种子、DataLoader 构造、日志、断点续训、OOM 自适应步进。

断点续训的三层保障（对应需求 3）：
  1. 每 epoch 存 last.pt；每 N step 存 step_latest.pt（SAVE_EVERY_STEPS>0 时）
  2. SIGTERM/SIGINT 信号 -> 当前 step 结束后落盘再退出（手动 kill 不丢进度）
  3. CUDA OOM -> 自动把 batch 切细重试，实在不行跳过该 step 并记录（显存中断不崩）
    进程被硬杀（kill -9 / 断电）时，下一次运行从最近一次 last.pt / step_latest.pt 恢复。
"""

import os
import sys
import json
import math
import time
import random
import signal
import glob
import shutil
import contextlib

import numpy as np

# ---------------------------------------------------------------------------
# 随机种子
# ---------------------------------------------------------------------------


def seed_everything(seed=42, deterministic=None):
    """完整可复现随机种子：Python hash / numpy / torch / cuDNN / SDPA 全覆盖。"""
    global _BASE_SEED
    _BASE_SEED = seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    # PyTorch（CPU+GPU，torch.manual_seed 会一并设置 CUDA）
    import torch
    torch.manual_seed(seed)

    if deterministic is None:
        try:
            import config
            deterministic = config.DETERMINISTIC
        except Exception:
            deterministic = True

    # 强制 cuDNN 确定性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 强制 PyTorch 所有运算确定性（warn_only 避免个别算子直接报错）
    torch.use_deterministic_algorithms(True, warn_only=True)

    if deterministic:
        # 注意：SDPA 的 flash / mem-efficient 两个后端在反向传播时是非确定的
        # （PyTorch 只会给个 UserWarning，warn_only 模式下不会报错，容易被忽略）。
        # 这里显式关掉它们，强制走 math 后端，代价是注意力稍慢、显存略高。
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        # **必须也关掉 cuDNN 的 SDPA 后端**（v4 已修，zql 版漏了这一行）。
        # 较新的 torch 有三个 SDPA 后端；只关 flash + mem-efficient 的话，
        # 注意力会落到 cuDNN 上，在 RTX 3090 + 该 cuDNN 版本下直接报
        # "cuDNN Frontend error: No execution plans support the graph"。
        # 症状是模型**一前向就崩**，而错误信息指向 cuDNN、看不出是缺了这行配置。
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # 固定线程数
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # 已经启动过并行工作后不允许再设，忽略
        pass


# DataLoader worker 的基准种子。由 seed_everything 写入，seed_worker 在子进程里读取。
# fork 出的 worker 会继承这个模块级变量，所以两步之间保持一致。
_BASE_SEED = 42


def seed_worker(worker_id):
    """DataLoader 每个 worker 的种子固定，保证多进程也完全可复现。"""
    worker_seed = _BASE_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_rng_states():
    """抓取所有随机源状态，用于断点续训时精确还原。"""
    import torch
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_states(state):
    import torch
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def build_loader(dataset, batch_size, shuffle, seed, num_workers=None, drop_last=False):
    """
    构造完全确定性的 DataLoader：
      - generator 固定洗牌顺序
      - worker_init_fn 固定每个 worker 的种子
    注意 num_workers=0 最严格但极慢；>0 时上述两项已保证可复现。
    """
    import torch
    from torch.utils.data import DataLoader
    import config

    if num_workers is None:
        num_workers = config.NUM_WORKERS

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------


class Logger:
    """同时写 stdout 与 jsonl 的训练日志。jsonl 便于事后画曲线。"""

    def __init__(self, log_path, also_stdout=True):
        self.log_path = log_path
        self.also_stdout = also_stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fh = open(log_path, 'a', buffering=1, encoding='utf-8')

    def log(self, msg, **fields):
        rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'msg': msg}
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        self._fh.write(line + '\n')
        if self.also_stdout:
            extra = ' '.join(f'{k}={v}' for k, v in fields.items())
            print(f'[{rec["time"]}] {msg} {extra}', flush=True)

    def close(self):
        with contextlib.suppress(Exception):
            self._fh.close()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 断点续训
# ---------------------------------------------------------------------------


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch=0, global_step=0, best=None, extra=None):
    """原子写：先写 .tmp 再 rename，避免写一半被杀导致断点损坏。"""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model': _unwrap(model),
        'epoch': epoch,
        'global_step': global_step,
        'best': best,
        'rng': get_rng_states(),
        'extra': extra or {},
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()

    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _unwrap(model):
    """取出裸模型（兼容 DDP / torch.compile 包装）。"""
    return getattr(model, '_orig_mod', model).state_dict()


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    restore_rng=True, strict=True):
    """加载断点，返回 (epoch, global_step, best, extra)。"""
    import torch
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    _get(model).load_state_dict(ckpt['model'], strict=strict)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    if restore_rng and ckpt.get('rng'):
        set_rng_states(ckpt['rng'])
    return (ckpt.get('epoch', 0), ckpt.get('global_step', 0),
            ckpt.get('best'), ckpt.get('extra', {}))


def _get(model):
    return getattr(model, '_orig_mod', model)


def resolve_resume(resume, ckpt_dir):
    """
    解析 config.RESUME：
      'none' -> 不续训，返回 None
      'auto' -> ckpt_dir/last.pt 存在则用它，否则从零开始
      '<路径>' -> 直接用该文件（不存在则报错）
    """
    if resume is None or str(resume).lower() == 'none':
        return None
    if str(resume).lower() == 'auto':
        last = os.path.join(ckpt_dir, 'last.pt')
        return last if os.path.exists(last) else None
    if not os.path.exists(resume):
        raise FileNotFoundError(f'指定的断点不存在: {resume}')
    return resume


def cleanup_snapshots(ckpt_dir, keep=3):
    """只保留最近 keep 个 step_*.pt 快照。"""
    files = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.pt')),
                   key=os.path.getmtime)
    for f in files[:-keep] if keep > 0 else files:
        with contextlib.suppress(OSError):
            os.remove(f)


def find_latest_snapshot(ckpt_dir):
    """最后一个 step_*.pt（last.pt 损坏或被硬杀时的兜底）。"""
    files = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.pt')),
                   key=os.path.getmtime)
    return files[-1] if files else None


# ---------------------------------------------------------------------------
# 中断信号处理
# ---------------------------------------------------------------------------


class InterruptHandler:
    """
    SIGTERM / SIGINT 时置标志位，训练循环在当前 step 结束后保存断点再退出。
    用法:
        ih = InterruptHandler()
        ...
        if ih.should_stop: save(); break
    """

    def __init__(self):
        self.should_stop = False
        self._orig = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._orig[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def _handle(self, signum, frame):
        if self.should_stop:
            # 第二次收到信号 -> 立即退出
            print(f'\n[中断] 再次收到信号 {signum}，立即退出', flush=True)
            self.restore()
            sys.exit(130)
        self.should_stop = True
        print(f'\n[中断] 收到信号 {signum}，将在当前 step 结束后保存断点并退出...',
              flush=True)

    def restore(self):
        for sig, h in self._orig.items():
            with contextlib.suppress(Exception):
                signal.signal(sig, h)


# ---------------------------------------------------------------------------
# 早停
# ---------------------------------------------------------------------------


class EarlyStopping:
    """
    连续 patience 个 epoch 的验证指标都没实质改善，就置位告诉调用方停训。

    「实质改善」= 比历史最优再低/再高 min_delta 以上（取决于 mode）。
    min_delta 在验证曲线末段是噪声级下降时是必需的：实测 MAE 的 val_loss
    每轮只降 0.00004~0.0007，只看 patience 的话"没改善"永远不成立。

    与各训练脚本自己的 best/is_best 逻辑**并存但不重复**：本类只负责计数，
    不改 best.pt 的保存判据 —— 那个仍用 min_delta=0 的严格比较，保证
    best.pt 始终是真正最优的那一份（早停在 patience 轮之后才触发，
    此时内存里的权重已经不是最优的了，调用方需要自己回读 best.pt）。
    """

    def __init__(self, patience, min_delta=0.0, mode='min'):
        if mode not in ('min', 'max'):
            raise ValueError(f'mode 只能是 min / max，收到 {mode!r}')
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.enabled = self.patience > 0
        self.best = None
        self.bad_epochs = 0
        self.triggered = False

    def seed_best(self, value):
        """断点续训时用断点里记的 best 当基准，避免 patience 从零重算。
        值无效（None / nan / inf）时忽略，保持 None。"""
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(v):
            self.best = v

    def step(self, score):
        """喂一个 epoch 的验证分数，返回 True 表示该早停了。"""
        if not self.enabled:
            return False
        # 指标无效（如 OOM 跳步污染出的 nan）：既不更新最优也不计入耐心。
        # 不能让它落进 else 分支，否则一次 nan 就会白白吃掉一轮耐心。
        try:
            s = float(score)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(s):
            return False

        if self.best is None:
            self.best = s
            return False

        improved = (s < self.best - self.min_delta) if self.mode == 'min' \
            else (s > self.best + self.min_delta)
        if improved:
            self.best = s
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        self.triggered = self.bad_epochs >= self.patience
        return self.triggered


# ---------------------------------------------------------------------------
# OOM 自适应步进
# ---------------------------------------------------------------------------


class OOMAdaptiveStep:
    """
    统一的「前向 + 反向 + 优化器步进」，自带显存溢出自动降级。

    chunks = 把逻辑 batch 切成几片做梯度累积（等价于梯度累积步数）。
    遇 CUDA OOM 时：清空缓存 -> chunks 翻倍（每片更小）-> 重跑该 step。
    chunks 涨到 == batch_size（逐样本）仍 OOM 时，跳过该 step 并计数。

    这样单卡 24GB 无需人工调参即可占满且不崩。
    """

    def __init__(self, optimizer, scaler=None, chunks=1, grad_clip=0.0,
                 max_skips=20, device='cuda:0'):
        import torch
        self.torch = torch
        self.optimizer = optimizer
        self.scaler = scaler
        self.chunks = max(1, int(chunks))
        self.grad_clip = grad_clip
        self.max_skips = max_skips
        self.device = device
        self.skip_count = 0
        # 供日志观察当前自适应到的切分粒度
        self.current_chunks = self.chunks

    def __call__(self, forward_loss_fn, batch):
        """
        forward_loss_fn(micro_batch) -> loss（标量，未缩放）
        batch: 支持 len() 与切片（Tensor / list / tuple / dict of Tensor）
        返回 (loss_value, ok)；ok=False 表示该 step 被跳过
        """
        torch = self.torch
        n = _batch_len(batch)
        chunks = min(self.chunks, n)

        while True:
            try:
                loss_val = self._run(forward_loss_fn, batch, chunks)
                # 记住这次成功的粒度：降级之后就稳定在这个粒度，
                # 避免每个 step 都重演一次 OOM -> 加倍 -> 重试的浪费
                self.chunks = chunks
                self.current_chunks = chunks
                return loss_val, True

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                if chunks >= n:
                    self.skip_count += 1
                    print(f'[OOM] 逐样本仍溢出，跳过该 step '
                          f'({self.skip_count}/{self.max_skips})', flush=True)
                    torch.cuda.empty_cache()
                    if self.skip_count >= self.max_skips:
                        raise RuntimeError(
                            '连续 OOM 次数过多，请调小 config 里的 BATCH_SIZE') from None
                    return float('nan'), False
                chunks = min(n, chunks * 2)
                print(f'[OOM] 清空缓存并把 batch 切细到 {chunks} 片后重试',
                      flush=True)

    def _run(self, forward_loss_fn, batch, chunks):
        torch = self.torch
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        micros = _split_batch(batch, chunks)

        # 按各片的样本数加权，而不是简单除以片数：
        # 否则 batch=8 切成 [3,3,2] 时，2 样本那片会和 3 样本那片等权，
        # 梯度就不再等价于整 batch 的均值了。
        n_total = max(1, _batch_len(batch))
        for mb in micros:
            n_mb = _batch_len(mb)
            w = n_mb / n_total
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=self.scaler is not None):
                loss = forward_loss_fn(mb)
            scaled = loss * w
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total += float(loss.detach()) * w

        if self.scaler is not None:
            try:
                self.scaler.unscale_(self.optimizer)
            except RuntimeError:
                # 上一次尝试在 unscale_ 之后、step 之前就 OOM 了，
                # 梯度已经是反缩放过的，这里跳过即可
                pass
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g['params']],
                self.grad_clip)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return total


def _batch_len(batch):
    """取 batch 的样本数。支持 dict / (dict, dict) 元组 / Tensor / list。"""
    if isinstance(batch, dict):
        return len(next(iter(batch.values())))
    if isinstance(batch, (list, tuple)):
        return _batch_len(batch[0])
    return len(batch)


def _split_batch(batch, chunks):
    """把 batch 均分成 chunks 份。最后一份吃掉余数。"""
    if chunks <= 1:
        return [batch]
    n = _batch_len(batch)
    size = (n + chunks - 1) // chunks
    out = []
    for i in range(0, n, size):
        out.append(_slice_batch(batch, i, min(i + size, n)))
    return out


def _slice_batch(batch, start, end):
    if isinstance(batch, dict):
        return {k: (v[start:end] if hasattr(v, '__getitem__') else v)
                for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_slice_batch(b, start, end) for b in batch)
    return batch[start:end]


def move_to_device(batch, device):
    """递归把 batch 搬到 device。dict/list/tuple/Tensor 都支持。"""
    import torch
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


# ---------------------------------------------------------------------------
# 杂项
# ---------------------------------------------------------------------------


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(model):
    n = count_parameters(model)
    return f'{n/1e6:.2f}M ({n:,})'


def gpu_mem_str(device='cuda:0'):
    import torch
    if not torch.cuda.is_available():
        return 'no-cuda'
    return (f'peak={torch.cuda.max_memory_allocated(device)/2**30:.2f}GB '
            f'cur={torch.cuda.memory_allocated(device)/2**30:.2f}GB')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def eval_batch_size():
    """
    验证/测试用的 batch size。config.EVAL_BATCH_SIZE > 0 时用它，
    否则沿用 SEG_BATCH_SIZE。推理无梯度，占用远小于训练。
    """
    import config
    if getattr(config, 'EVAL_BATCH_SIZE', 0) > 0:
        return config.EVAL_BATCH_SIZE
    return config.SEG_BATCH_SIZE


# ---------------------------------------------------------------------------
# 快速文件列举（NFS 上 Path.rglob 慢到不可用）
# ---------------------------------------------------------------------------


def scan_files(root, exts=None):
    """
    用 os.scandir 递归列出 root 下的所有文件（已排序）。

    为什么不用 pathlib.Path.rglob：scandir 的 DirEntry 自带 d_type，
    判定文件/目录不需要对每个条目额外 stat；在 NFS 上实测 3072 个文件
    rglob 超过 280 秒跑不完，scandir 只要 23 秒。
    """
    out = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        if exts is None or \
                                os.path.splitext(e.name)[1].lower() in exts:
                            out.append(e.path)
        except OSError:
            continue
    out.sort()
    return out


def list_files(root, exts=None, cache_dir=None, refresh=False):
    """
    带磁盘缓存的文件列举。数据集目录是只读的，列一次存下来，
    后续 train/eval 各脚本直接读缓存，省掉每次几十秒。

    缓存失效判据：root 自身的 mtime_ns 变化（删/加了一层目录）。
    数据集被替换时用 refresh=True 强制重建。
    """
    import hashlib

    if cache_dir is None:
        try:
            import config
            cache_dir = os.path.join(config.OUTPUT_ROOT, '_cache', 'filelists')
        except Exception:
            cache_dir = None

    key = None
    if cache_dir:
        sig = f'{os.path.abspath(root)}|{sorted(exts) if exts else "all"}'
        key = hashlib.md5(sig.encode()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f'files_{key}.json')
        if not refresh and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                if data.get('root') == os.path.abspath(root) and \
                        data.get('root_mtime') == _mtime(root):
                    return data['files']
            except Exception:
                pass
    else:
        cache_path = None

    files = scan_files(root, exts)

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'root': os.path.abspath(root), 'root_mtime': _mtime(root),
                       'files': files}, f)
        os.replace(tmp, cache_path)
    return files


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return -1


def copy_config_snapshot(dst_dir, src_file=None):
    """把当前 config.py 拷进输出目录，保证每次实验可追溯。"""
    src_file = src_file or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'config.py')
    dst = os.path.join(dst_dir, 'config_snapshot.py')
    ensure_dir(dst_dir)
    with contextlib.suppress(Exception):
        shutil.copyfile(src_file, dst)
    return dst


def cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps):
    """线性 warmup + 余弦退火，返回当前 step 的 lr 系数（乘 base_lr）。"""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    import math
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


# ---------------------------------------------------------------------------
# 命令行覆盖（不改 config.py 就能做冒烟测试/小实验）
# ---------------------------------------------------------------------------


def apply_cli_overrides(argv=None):
    """
    支持 `--KEY value` 与 `--KEY=value`，KEY 是 config.py 里任意大写变量名。
    例: --MAE_EPOCHS 1 --MAE_BATCH_SIZE 8 --RXRX1_SUBSET_RATIO 0.01
    """
    import config
    argv = list(sys.argv[1:] if argv is None else argv)
    i = 0
    applied = {}
    while i < len(argv):
        a = argv[i]
        if not a.startswith('--'):
            i += 1
            continue
        if '=' in a:
            key, val = a[2:].split('=', 1)
            i += 1
        elif i + 1 < len(argv):
            key, val = a[2:], argv[i + 1]
            i += 2
        else:
            # 末尾的裸 --FLAG（没有值）：当作开关/无关参数跳过。
            # 之前这里直接索引 argv[i+1] 会 IndexError，把调用方的参数顺序
            # 差异放大成崩溃。
            i += 1
            continue
        key = key.upper()
        if not hasattr(config, key):
            # 脚本自己的参数（如 eval_seg.py 的 --ckpt）不是 config 变量，跳过即可
            continue
        new = _coerce(val, getattr(config, key))
        setattr(config, key, new)
        applied[key] = new

    # 目录名由全部影响指标的参数拼成，所以任何覆盖都可能改路径：一律重算，
    # 只在路径真的变了时提示（否则 --DEVICE_ID 4 这类运行期旋钮也会刷屏）
    before = config.OUTPUT_ROOT
    config.refresh_output_dirs()
    if config.OUTPUT_ROOT != before:
        print(f"[CLI 覆盖] 输出目录 -> {config.OUTPUT_ROOT}", flush=True)

    # DEVICE_ID 要特殊处理：CUDA_VISIBLE_DEVICES 虽然在 config 导入时就设好了，
    # 但只要还没真正初始化 CUDA 上下文，重设它依然生效（torch 是惰性读取的）。
    if 'DEVICE_ID' in applied:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(applied['DEVICE_ID'])
        config.DEVICE = 'cuda:0'
        print(f"[CLI 覆盖] 已切换 CUDA_VISIBLE_DEVICES="
              f"{applied['DEVICE_ID']}（须在任何 CUDA 调用之前生效）", flush=True)

    if applied:
        print(f'[CLI 覆盖] {applied}', flush=True)
    return applied


def _coerce(val, cur):
    if isinstance(cur, bool):
        return str(val).lower() in ('1', 'true', 'yes', 'y')
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(float(val))
    if isinstance(cur, float):
        return float(val)
    if isinstance(cur, (tuple, list)):
        parts = [p.strip() for p in str(val).split(',') if p.strip()]
        cast = type(cur[0]) if cur else str
        return type(cur)(cast(p) for p in parts)
    return val
