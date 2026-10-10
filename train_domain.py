"""
Stage 2 —— 域适应。用「重建损失 + 域判别损失(GRL)」把编码器适配到目标域。

源域固定为 RxRx1（荧光），目标域固定为 OOC（明场），见 config.py C 节。

用法:
    /data/user/miniconda3/envs/mumo/bin/python train_domain.py

断点续训: 直接重跑同一命令，从 {DIR_STAGE2}/ckpt/last.pt 恢复。

产物:
    {DIR_STAGE2}/ckpt/last.pt | best.pt
    {DIR_STAGE2}/encoder.pt        域适配后的编码器，供 Stage 3 加载
    {DIR_STAGE2}/train_log.jsonl
    {DIR_STAGE2}/metrics.json
"""
import setproctitle
setproctitle.setproctitle("python3")
import config  # noqa: F401  —— 必须最先 import

import functools
import os
import sys
import time

import torch
import torch.nn as nn

import utils
from data.ooc import build_ooc
from data.rxrx1 import build_rxrx1
from models import MAE, ClsHead, DomainDiscriminator

SOURCE_DOMAIN_ID = 0
TARGET_DOMAIN_ID = 1


# ---------------------------------------------------------------------------
# 数据集组装：统一输出 {'image', 'domain', 'aux_label', 'index'}
# ---------------------------------------------------------------------------


class DomainWrapper(torch.utils.data.Dataset):
    """给基础数据集补上域标签。aux_from='ooc' 时透传 good/bad 标签。"""

    def __init__(self, base, domain_id, aux_from=None):
        self.base = base
        self.domain_id = domain_id
        self.aux_from = aux_from

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        aux = -1
        if self.aux_from == 'ooc' and s.get('aux_label', -1) >= 0:
            aux = int(s['aux_label'])
        return {'image': s['image'], 'domain': self.domain_id,
                'aux_label': aux, 'index': idx}


def build_domain_dataset(name, img_size, domain_id):
    """构造单个域的数据集，返回已带 domain 字段的 Dataset。

    只认识 'rxrx1'（源域）与 'ooc'（目标域）两个名字 —— 多域拼接（'+'）与 'hts'
    目标域的实现已剔除，其它取值一律报错，避免静默退化。
    """
    if name == 'rxrx1':
        base = build_rxrx1(phase='train', img_size=img_size,
                           train_split_only=config.DOM_RXRX1_TRAIN_SPLIT_ONLY)
        return DomainWrapper(base, domain_id, aux_from=None)
    if name == 'ooc':
        base = build_ooc(splits=config.OOC_SPLITS_FOR_DOMAIN, img_size=img_size,
                         train=True)
        return DomainWrapper(base, domain_id, aux_from='ooc')
    raise ValueError(
        f'未知的域 {name!r}：本工程只实现 DOM_SOURCE="rxrx1" / DOM_TARGET="ooc"，'
        f'多域拼接与 hts 域的代码已删除')


def cycle(loader):
    """无限循环一个 DataLoader，用于源域/目标域步数对齐。"""
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# 留出集验证
# ---------------------------------------------------------------------------


class _EvenlySpaced(torch.utils.data.Dataset):
    """在 [0, len(base)) 上均匀取 n 个下标，把留出集固定成一个不随 epoch
    变化的小集合。

    不能取「前 n 个」：OOC 的文件列表是按标签分组的（good 全部在前、bad 在后，
    实测 val 的 286 张里前 163 张全是 good），取前 n 个会只拿到单一类别，
    留出集上的 aux_acc 直接退化成 100%。
    """

    def __init__(self, base, n):
        self.base = base
        n = min(int(n), len(base))
        if n <= 1:
            self.idx = list(range(n))
        else:
            step = (len(base) - 1) / (n - 1)
            self.idx = sorted({int(round(i * step)) for i in range(n)})

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def build_domain_eval_dataset(name, img_size, n_samples):
    """
    构造留出集用的数据集：走 eval 变换（不随机增强），截取 n_samples 张，
    保证每个 epoch 评估的都是同一批图。

    **必须用训练时没见过的数据**，否则指标虚高到没有参考价值
    （用训练图片评估时 aux_acc 直接是 1.0）：
        rxrx1 -> dataset=='test' 的那批 site
        ooc   -> 'val' split
    """
    if name == 'rxrx1':
        base = build_rxrx1(phase='eval', img_size=img_size,
                           dataset_split='test')
        has_aux = False
    elif name == 'ooc':
        base = build_ooc(splits=['val'], img_size=img_size, train=False)
        has_aux = True
    else:
        raise ValueError(
            f'未知的域 {name!r}：本工程只实现 DOM_SOURCE="rxrx1" / '
            f'DOM_TARGET="ooc"，多域拼接与 hts 域的代码已删除')
    if len(base) == 0:
        raise RuntimeError(
            f'域 {name!r} 的留出集为空，无法评估。请检查数据目录，'
            f'或把 config.DOM_VAL_SAMPLES 设为 0 关闭留出集验证。')
    # aux 标签：只有 OOC 有 good/bad，RxRx1 那侧填 -1
    return _WithAux(_EvenlySpaced(base, n_samples), has_aux=has_aux)


class _WithAux(torch.utils.data.Dataset):
    """给留出集补上 aux_label 字段（OOC 有 good/bad，其它域为 -1）。"""

    def __init__(self, base, has_aux):
        self.base = base
        self.has_aux = has_aux

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        aux = int(s['aux_label']) if (self.has_aux and 'aux_label' in s) else -1
        return {'image': s['image'], 'aux_label': aux, 'index': idx}


@torch.no_grad()
def evaluate_domain(mae, disc, aux_head, src_loader, tgt_loader, device, ce):
    """
    留出集上评估域适应效果。返回的指标里最该盯的两个：
        dom_acc   域判别准确率，越接近 0.5 说明两个域在特征空间越不可分（适配成功）
        recon_tgt 目标域重建损失，应该随训练下降
        aux_acc   OOC good/bad 分类准确率，衡量编码器有没有学到目标域的语义
    """
    mae.eval()
    disc.eval()
    if aux_head is not None:
        aux_head.eval()

    # 注意 aux_acc 初值必须是 0.0 而不是 nan：这里用的是 += 累加，
    # nan 加上任何数仍然是 nan，会把整个指标吃掉（n_aux==0 时才返回 nan）
    # recon_src / recon_tgt 各自需要独立的样本计数：源域和目标域的 loader 长度、
    # 最后一批的批大小都可能不同，共用 n 会把两者都乘上"自己的占比"（实测各减半）。
    stats = {'recon_src': 0.0, 'recon_tgt': 0.0, 'domain': 0.0,
             'dom_acc': 0.0, 'aux_acc': 0.0,
             'n': 0, 'n_src': 0, 'n_tgt': 0, 'n_aux': 0}

    for sb, tb in zip(src_loader, tgt_loader):
        sb = utils.move_to_device(sb, device)
        tb = utils.move_to_device(tb, device)
        tok_s, cls_s, mask_s, ids_s = mae.encoder(sb['image'],
                                                  mask_ratio=mae.mask_ratio)
        recon_s = mae.compute_loss(mae.forward_decoder(tok_s, ids_s),
                                   sb['image'], mask_s)
        tok_t, cls_t, mask_t, ids_t = mae.encoder(tb['image'],
                                                  mask_ratio=mae.mask_ratio)
        recon_t = mae.compute_loss(mae.forward_decoder(tok_t, ids_t),
                                   tb['image'], mask_t)

        dom_logits = disc(torch.cat([cls_s, cls_t], dim=0))
        dom_target = torch.cat([
            torch.zeros(cls_s.shape[0], dtype=torch.long, device=device),
            torch.ones(cls_t.shape[0], dtype=torch.long, device=device)])

        bs = sb['image'].shape[0] + tb['image'].shape[0]
        stats['recon_src'] += float(recon_s) * sb['image'].shape[0]
        stats['recon_tgt'] += float(recon_t) * tb['image'].shape[0]
        stats['domain'] += float(ce(dom_logits, dom_target)) * bs
        stats['dom_acc'] += float((dom_logits.argmax(1) == dom_target).sum())
        stats['n'] += bs
        stats['n_src'] += sb['image'].shape[0]
        stats['n_tgt'] += tb['image'].shape[0]

        if aux_head is not None:
            aux_y = torch.cat([sb['aux_label'], tb['aux_label']], dim=0)
            ok = aux_y >= 0
            if bool(ok.any()):
                pred = aux_head(torch.cat([cls_s, cls_t], dim=0)).argmax(1)
                stats['aux_acc'] += float((pred[ok] == aux_y[ok]).sum())
                stats['n_aux'] += int(ok.sum())

    mae.train()
    disc.train()
    if aux_head is not None:
        aux_head.train()

    n = max(1, stats['n'])
    out = {'recon_src': stats['recon_src'] / max(1, stats['n_src']),
           'recon_tgt': stats['recon_tgt'] / max(1, stats['n_tgt']),
           'domain': stats['domain'] / n,
           'dom_acc': stats['dom_acc'] / n}
    out['aux_acc'] = (stats['aux_acc'] / stats['n_aux']
                      if stats['n_aux'] else float('nan'))
    return out


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def main():
    utils.apply_cli_overrides()
    utils.seed_everything(config.SEED)

    out_dir = utils.ensure_dir(config.DIR_STAGE2)
    ckpt_dir = utils.ensure_dir(os.path.join(out_dir, 'ckpt'))
    utils.copy_config_snapshot(out_dir)
    log = utils.Logger(os.path.join(out_dir, 'train_log.jsonl'))

    device = config.DEVICE
    img_size = config.MAE_INPUT_SIZE
    log.log('Stage2 启动', source=config.DOM_SOURCE, target=config.DOM_TARGET,
            device=device, aux_cls=config.DOM_AUX_CLS)

    # ---------------- 数据 ----------------
    src_ds = build_domain_dataset(config.DOM_SOURCE, img_size, SOURCE_DOMAIN_ID)
    tgt_ds = build_domain_dataset(config.DOM_TARGET, img_size, TARGET_DOMAIN_ID)

    src_loader = utils.build_loader(src_ds, config.DOM_BATCH_SIZE, shuffle=True,
                                    seed=config.SEED, drop_last=True)
    tgt_loader = utils.build_loader(tgt_ds, config.DOM_BATCH_SIZE, shuffle=True,
                                    seed=config.SEED + 1, drop_last=True)
    # 固定留出集：源域/目标域各取 DOM_VAL_SAMPLES 张，走 eval 变换、不增强，
    # 每个 epoch 评估同一批图，指标才可跨 epoch 比较
    val_src_loader = val_tgt_loader = None
    if config.DOM_VAL_SAMPLES > 0:
        val_src_loader = utils.build_loader(
            build_domain_eval_dataset(config.DOM_SOURCE, img_size,
                                      config.DOM_VAL_SAMPLES),
            config.DOM_BATCH_SIZE, shuffle=False, seed=config.SEED)
        val_tgt_loader = utils.build_loader(
            build_domain_eval_dataset(config.DOM_TARGET, img_size,
                                      config.DOM_VAL_SAMPLES),
            config.DOM_BATCH_SIZE, shuffle=False, seed=config.SEED)

    steps_per_epoch = len(src_loader)

    if steps_per_epoch == 0:
        raise SystemExit(
            f"steps_per_epoch=0：DOM_BATCH_SIZE={config.DOM_BATCH_SIZE} 比源域"
            f"({len(src_ds)}) 还大，drop_last=True 会把所有 batch 丢掉，请调小。")

    total_steps = steps_per_epoch * config.DOM_EPOCHS
    warmup_steps = steps_per_epoch * config.DOM_WARMUP_EPOCHS
    log.log('数据就绪', n_source=len(src_ds), n_target=len(tgt_ds),
            steps_per_epoch=steps_per_epoch, total_steps=total_steps)

    # ---------------- 模型 ----------------
    # 编码器由源域(RxRx1，取 len(RXRX1_CHANNELS) 个通道)与目标域(OOC，RGB 3 通道)
    # 共用，两边通道数必须相同；也必须与 Stage 1 导出的 encoder.pt 一致。之前这里
    # 硬编码 3，改了 RXRX1_CHANNELS 就会在载入 Stage 1 权重时炸出 size mismatch。
    in_chans = len(config.RXRX1_CHANNELS)
    if in_chans != 3:
        raise SystemExit(
            f'RXRX1_CHANNELS={config.RXRX1_CHANNELS} 有 {in_chans} 个通道，但目标域 '
            f'OOC 是 RGB 3 通道，二者必须相同，请把 RXRX1_CHANNELS 设为 3 个通道')
    mae = MAE(img_size=img_size, patch_size=config.MAE_PATCH_SIZE, in_chans=in_chans,
              enc_dim=config.MAE_ENC_DIM, enc_depth=config.MAE_ENC_DEPTH,
              enc_heads=config.MAE_ENC_HEADS,
              dec_dim=config.MAE_DEC_DIM, dec_depth=config.MAE_DEC_DEPTH,
              dec_heads=config.MAE_DEC_HEADS,
              mask_ratio=config.MAE_MASK_RATIO,
              norm_pix_loss=config.MAE_NORM_PIX_LOSS).to(device)
    disc = DomainDiscriminator(in_dim=config.MAE_ENC_DIM, hidden=256,
                               num_domains=2,
                               lambd=config.DOM_GRL_LAMBDA).to(device)
    aux_head = ClsHead(in_dim=config.MAE_ENC_DIM, num_classes=2).to(device) \
        if config.DOM_AUX_CLS else None

    joint = JointModel(mae, disc, aux_head).to(device)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=config.DOM_LR,
                                  weight_decay=config.DOM_WEIGHT_DECAY,
                                  betas=tuple(config.MAE_BETAS))
    scaler = torch.amp.GradScaler('cuda') if config.DOM_AMP else None
    log.log('模型就绪', params=utils.format_params(joint))

    # ---------------- 载入 Stage1 预训练编码器 ----------------
    pretrained = find_stage1_encoder(log)
    if pretrained:
        state = torch.load(pretrained, map_location='cpu', weights_only=False)
        missing, unexpected = mae.encoder.load_state_dict(state, strict=False)
        log.log('已载入 Stage1 编码器', path=pretrained,
                missing=len(missing), unexpected=len(unexpected))
    else:
        log.log('[警告] 未找到 Stage1 编码器，Stage2 将从随机初始化开始')

    # ---------------- 断点续训 ----------------
    start_epoch, global_step, best, _ = 0, 0, float('inf'), {}
    resume_path = utils.resolve_resume(config.RESUME, ckpt_dir)
    if resume_path:
        start_epoch, global_step, best, _ = utils.load_checkpoint(
            resume_path, joint, optimizer, None, scaler)
        log.log('已恢复断点', path=resume_path, epoch=start_epoch,
                step=global_step)
    else:
        log.log('未发现断点，从零开始')

    stepper = utils.OOMAdaptiveStep(optimizer, scaler=scaler, chunks=1,
                                    grad_clip=config.DOM_GRAD_CLIP, device=device)
    interrupt = utils.InterruptHandler()
    early_stop = utils.EarlyStopping(config.DOM_EARLY_STOP_PATIENCE,
                                     config.DOM_EARLY_STOP_MIN_DELTA,
                                     mode='min')
    early_stop.seed_best(best)
    if early_stop.enabled:
        if val_src_loader is None:
            # DOM_VAL_SAMPLES=0 时没有留出集，judge 不了收敛，早停静默失效最容易
            # 被误认为"早停没生效"，所以这里显式说出来
            log.log('[警告] DOM_VAL_SAMPLES=0，没有留出集指标，早停不会生效')
        else:
            log.log('早停已启用', patience=early_stop.patience,
                    min_delta=early_stop.min_delta, mode=early_stop.mode,
                    judge='留出集 recon_tgt')
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    torch.cuda.reset_peak_memory_stats(device)

    tgt_iter = cycle(tgt_loader)
    run, run_n = {}, 0
    t_last = time.time()
    early_stopped = False
    epoch = start_epoch - 1
    info = {}          # loss_fn 每个 step 往里写各分量，日志里分开打印
    val_history = []   # 每 epoch 的留出集指标，最后写进 metrics.json

    for epoch in range(start_epoch, config.DOM_EPOCHS):
        if interrupt.should_stop:
            break
        for src_batch in src_loader:
            tgt_batch = next(tgt_iter)

            lr = utils.cosine_lr(global_step, total_steps, config.DOM_LR,
                                 config.DOM_LR * 0.05, warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = lr

            info.clear()
            step_loss = functools.partial(
                loss_fn, _extra=(mae, disc, aux_head, device, info))
            loss_val, ok = stepper(step_loss, (src_batch, tgt_batch))
            if ok:
                run['loss'] = run.get('loss', 0.0) + loss_val
                for k, v in info.items():
                    run[k] = run.get(k, 0.0) + v
                run_n += 1
            global_step += 1

            if global_step % config.LOG_EVERY_STEPS == 0:
                now = time.time()
                avg = {k: round(v / max(1, run_n), 4) for k, v in run.items()}
                log.log('train', epoch=epoch, step=global_step, **avg,
                        lr=f'{lr:.2e}',
                        sec_per_step=round((now - t_last) / config.LOG_EVERY_STEPS, 3),
                        mem=utils.gpu_mem_str(device))
                # 训练时最该盯的两个数：dom_acc 越接近 0.5 说明域判别越平衡
                run, run_n, t_last = {}, 0, now

            if config.SAVE_EVERY_STEPS > 0 and                     global_step % config.SAVE_EVERY_STEPS == 0:
                # 本阶段每个 epoch 约 65 分钟,原来只在 epoch 末尾存 last.pt —— 一旦被
                # 外部杀掉(本机今天 7 次),整轮一小时算力白扔。这里把周期检查点直接
                # 写成 last.pt:续训逻辑读的就是它,因此**不需要改任何续训代码**;
                # save_checkpoint 是原子写(tmp+rename),写一半被杀也不会损坏断点。
                try:
                    utils.save_checkpoint(os.path.join(ckpt_dir, 'last.pt'),
                                          joint, optimizer, None, scaler, epoch,
                                          global_step, best)
                    snap = os.path.join(ckpt_dir, f'step_{global_step:08d}.pt')
                    try:
                        os.link(os.path.join(ckpt_dir, 'last.pt'), snap)
                    except OSError:
                        import shutil as _shutil
                        _shutil.copy2(os.path.join(ckpt_dir, 'last.pt'), snap)
                    utils.cleanup_snapshots(ckpt_dir, config.KEEP_LAST_N)
                    log.log('周期检查点已刷新 last.pt', step=global_step)
                except Exception as e:      # 存检查点失败绝不能弄死训练
                    log.log('[警告] 周期检查点写入失败', err=str(e))

            if interrupt.should_stop:
                break

        # ---- 每个 epoch 在固定留出集上评估，指标进 train_log.jsonl ----
        val = {}
        is_best = False
        stop_now = False
        es_score = None
        if val_src_loader is not None:
            val = evaluate_domain(mae, disc, aux_head, val_src_loader,
                                  val_tgt_loader, device, ce)
            val_history.append({'epoch': epoch, **val})
            # 选择判据：目标域重建损失，越小越好（与 Stage1 的 val_loss 同口径）
            score = val['recon_tgt']
            is_best = score < best
            if is_best:
                best = score
            stop_now = early_stop.step(score)
            es_score = score
        log.log('epoch 结束', epoch=epoch,
                **{k: round(v, 5) for k, v in val.items()},
                best=None if best == float('inf') else round(best, 5),
                is_best=is_best, bad_epochs=early_stop.bad_epochs,
                es_score=es_score, es_best=early_stop.best,
                mem=utils.gpu_mem_str(device))
        t_last = time.time()

        # ---- epoch 结束：先验证再落盘，best.pt 只在改善时写 ----
        if config.SAVE_EVERY_EPOCH:
            utils.save_checkpoint(os.path.join(ckpt_dir, 'last.pt'), joint,
                                  optimizer, None, scaler, epoch + 1,
                                  global_step, best)
            if is_best:
                utils.save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), joint,
                                      optimizer, None, scaler, epoch + 1,
                                      global_step, best)

        if stop_now:
            early_stopped = True
            log.log('早停触发', epoch=epoch, bad_epochs=early_stop.bad_epochs,
                    patience=early_stop.patience,
                    min_delta=early_stop.min_delta,
                    best_recon_tgt=round(early_stop.best, 5))
            break

    # ---------------- 导出适配后的编码器 ----------------
    # 必须从 best.pt 回读再导出，不能用内存里当前的权重：早停是在 best 之后又
    # 忍了 patience 轮才触发的，此时内存里的权重已经比 best 差了一截。
    # （不加早停时「最后一轮 ≈ 最优」通常成立，该问题不会暴露。）
    best_ckpt = os.path.join(ckpt_dir, 'best.pt')
    if os.path.exists(best_ckpt):
        utils.load_checkpoint(best_ckpt, joint, restore_rng=False)
        log.log('已回读 best.pt 再导出编码器', path=best_ckpt)
    enc_path = os.path.join(out_dir, 'encoder.pt')
    torch.save(mae.encoder_state_dict(), enc_path)

    summary = {
        'stage': 'stage2_domain',
        'source': config.DOM_SOURCE, 'target': config.DOM_TARGET,
        'epochs_done': min(epoch + 1, config.DOM_EPOCHS),
        'global_step': global_step,
        'es_best': early_stop.best,
        'n_source': len(src_ds), 'n_target': len(tgt_ds),
        'peak_gpu_gb': round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        'oom_skipped_steps': stepper.skip_count,
        'encoder_path': enc_path,
        'interrupted': interrupt.should_stop,
        'early_stopped': early_stopped,
        'val_history': val_history,
    }
    utils.write_json(os.path.join(out_dir, 'metrics.json'), summary)
    log.log('Stage2 完成', **summary)
    interrupt.restore()
    log.close()


def loss_fn(batch, _extra):
    """
    一个训练 step 的损失。编码器每次只前向一遍：
    用同一次（带掩码的）前向同时拿到重建损失和用于域判别的全局特征。

    各个损失分量写进 _extra 里传入的 info 字典，便于在日志里分开观察：
    DANN 训练时"域判别损失上升"和"重建损失上升"是完全不同的信号，
    只看总损失没法判断该调哪个权重。
    """
    mae, disc, aux_head, device, info = _extra
    sb, tb = batch
    sb = utils.move_to_device(sb, device)
    tb = utils.move_to_device(tb, device)

    # --- 一次编码器前向，同时得到重建与判别所需的一切 ---
    tok_s, cls_s, mask_s, ids_s = mae.encoder(sb['image'],
                                              mask_ratio=mae.mask_ratio)
    recon_s = mae.compute_loss(mae.forward_decoder(tok_s, ids_s),
                               sb['image'], mask_s)
    tok_t, cls_t, mask_t, ids_t = mae.encoder(tb['image'],
                                              mask_ratio=mae.mask_ratio)
    recon_t = mae.compute_loss(mae.forward_decoder(tok_t, ids_t),
                               tb['image'], mask_t)
    loss_recon = 0.5 * (recon_s + recon_t)

    # --- 域判别：梯度反转层让编码器学到域不可分的表征 ---
    dom_logits = disc(torch.cat([cls_s, cls_t], dim=0))
    dom_target = torch.cat([
        torch.full((cls_s.shape[0],), SOURCE_DOMAIN_ID, dtype=torch.long,
                   device=device),
        torch.full((cls_t.shape[0],), TARGET_DOMAIN_ID, dtype=torch.long,
                   device=device),
    ])
    loss_dom = nn.functional.cross_entropy(dom_logits, dom_target)

    loss = (config.DOM_RECON_WEIGHT * loss_recon
            + config.DOM_DOMAIN_WEIGHT * loss_dom)
    info['recon'] = float(loss_recon.detach())
    info['recon_src'] = float(recon_s.detach())
    info['recon_tgt'] = float(recon_t.detach())
    info['domain'] = float(loss_dom.detach())
    # 判别器准确率：0.5 表示编码器已经骗过它，是最理想的状态
    with torch.no_grad():
        acc = (dom_logits.argmax(dim=1) == dom_target).float().mean()
        info['dom_acc'] = float(acc)

    # --- OOC good/bad 辅助分类（半监督，只在有标签样本上算）---
    # 必须传 ignore_index=-1：只有 OOC 的样本有 good/bad 标签，
    # 另一侧的样本 aux_label=-1，不屏蔽掉会直接触发 nll_loss 的下标越界断言
    if aux_head is not None:
        aux_y = torch.cat([sb['aux_label'], tb['aux_label']], dim=0)
        if bool((aux_y >= 0).any()):
            loss_aux = nn.functional.cross_entropy(
                aux_head(torch.cat([cls_s, cls_t], dim=0)), aux_y,
                ignore_index=-1)
            loss = loss + config.DOM_AUX_WEIGHT * loss_aux
            info['aux'] = float(loss_aux.detach())

    return loss


class JointModel(nn.Module):
    """把编码器/解码器/判别头/辅助头打包成一个模块，便于统一存断点。"""

    def __init__(self, mae, disc, aux=None):
        super().__init__()
        self.mae = mae
        self.disc = disc
        self.aux = aux


def find_stage1_encoder(log):
    """
    优先用 Stage1 导出的 encoder.pt；没有就回落到 Stage1 的 best.pt/last.pt，
    从中抽取 encoder.* 前缀并存成 encoder.pt 缓存。
    """
    direct = os.path.join(config.DIR_STAGE1, 'encoder.pt')
    if os.path.exists(direct):
        return direct

    for name in ('best.pt', 'last.pt'):
        ck = os.path.join(config.DIR_STAGE1, 'ckpt', name)
        if not os.path.exists(ck):
            continue
        state = torch.load(ck, map_location='cpu', weights_only=False)
        model_state = state.get('model', state)
        enc = {k.replace('encoder.', ''): v for k, v in model_state.items()
               if k.startswith('encoder.')}
        if not enc:
            continue
        cache = os.path.join(config.DIR_STAGE1, 'encoder.pt')
        torch.save(enc, cache)
        log.log('从完整断点抽取编码器', src=ck, dst=cache)
        return cache
    return None


if __name__ == '__main__':
    sys.exit(main())
