"""
Stage 3 评估 —— 在 HTS-Seg 测试集上算分割指标，并与 baseline 论文 Table 2 对齐。

指标口径与 baseline 的 metri() 完全一致（见 metrics.py），测试集划分也完全一致
（见 data/hts.py 的 baseline_split），因此结果可直接和论文比较。

用法:
    /data/user/miniconda3/envs/mumo/bin/python eval_seg.py
    /data/user/miniconda3/envs/mumo/bin/python eval_seg.py --ckpt <路径>

产物:
    {DIR_STAGE3}/eval/test_metrics.json    全部指标（逐图 + 汇总口径）
    {DIR_STAGE3}/eval/table2.md            与论文 Table 2 同格式的对照表
    {DIR_STAGE3}/eval/test_predictions.npz 测试集预测掩码（供 classify.py 复用）
    {DIR_STAGE3}/eval/figures/             若干定性对比图
"""

import config  # noqa: F401  —— 必须最先 import

import gc
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import metrics as M
import boundary
import utils
from data.hts import build_hts_datasets
from models import SegHead, ViTEncoder


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class SegModel(nn.Module):
    """编码器 + 分割头。"""

    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size=config.HTS_IMG_SIZE, patch_size=config.MAE_PATCH_SIZE,
            in_chans=3, embed_dim=config.MAE_ENC_DIM,
            depth=config.MAE_ENC_DEPTH, num_heads=config.MAE_ENC_HEADS)
        grid = self.encoder.grid_size
        self.seg_head = SegHead(embed_dim=config.MAE_ENC_DIM,
                                num_classes=num_classes, grid_size=grid,
                                patch_size=config.MAE_PATCH_SIZE,
                                mode=config.SEG_DECODER)
    def forward(self, x):
        tokens, _cls = self.encoder(x, mask_ratio=0.0)
        return self.seg_head(tokens)


def build_seg_model(device):
    model = SegModel().to(device)
    if config.SEG_FREEZE_LAYERS > 0:
        for blk in model.encoder.blocks[:config.SEG_FREEZE_LAYERS]:
            for p in blk.parameters():
                p.requires_grad = False
    return model


def load_encoder_weights(encoder, device, log=None, verbose=True):
    """
    按 Stage2 -> Stage1 的顺序找预训练编码器；都找不到就随机初始化。
    返回实际使用的路径（None 表示没找到）。
    """
    cands = [
        os.path.join(config.DIR_STAGE2, 'encoder.pt'),
        os.path.join(config.DIR_STAGE2, 'ckpt', 'best.pt'),
        os.path.join(config.DIR_STAGE1, 'encoder.pt'),
        os.path.join(config.DIR_STAGE1, 'ckpt', 'best.pt'),
    ]
    for c in cands:
        if not os.path.exists(c):
            continue
        state = torch.load(c, map_location='cpu', weights_only=False)
        model_state = state.get('model', state)
        if any(k.startswith('mae.encoder.') for k in model_state):
            enc = {k.replace('mae.encoder.', ''): v for k, v in model_state.items()
                   if k.startswith('mae.encoder.')}
        elif any(k.startswith('encoder.') for k in model_state):
            enc = {k.replace('encoder.', ''): v for k, v in model_state.items()
                   if k.startswith('encoder.')}
        else:
            enc = model_state
        missing, unexpected = encoder.load_state_dict(enc, strict=False)
        if verbose:
            msg = f'载入编码器权重: {c} (missing={len(missing)}, ' \
                  f'unexpected={len(unexpected)})'
            print(f'[eval] {msg}', flush=True)
            if log:
                log.log(msg)
        return c
    if verbose:
        # 输出目录名由参数拼成，三阶段被传了不同的身份参数就会落进不同根目录，
        # 于是这里全部落空、静默退化成随机初始化。横幅列出实际找过的路径便于自查。
        print('[eval] ' + '!' * 66, flush=True)
        print('[eval] [严重] 未找到任何预训练编码器，本次使用随机初始化！', flush=True)
        print('[eval] 已查找以下路径（均不存在）:', flush=True)
        for c in cands:
            print(f'[eval]   {c}', flush=True)
        print('[eval] 若是流水线运行，说明三阶段被传了不同的身份参数、落进了不同'
              '根目录。请用 bash run_all.sh，或保证三个阶段传同一组参数。', flush=True)
        print('[eval] ' + '!' * 66, flush=True)
    return None


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


def _forward_once(model, img):
    """单次带 autocast 的前向，可能抛 CUDA OOM。"""
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.SEG_AMP):
        return model(img)


def _forward_oom_safe(model, img, device):
    """
    推理前向：OOM 时把每片 batch 减半，逐片处理，直到全部算完。

    训练侧的 OOM 由 OOMAdaptiveStep 兜底；推理没有梯度、走的是另一条代码路径，
    必须单独保护 —— 否则一个偏大的 batch size 会让**验证阶段**直接崩掉训练。

    为什么用 while 循环 + 逐片处理，而不是在 except 里递归：
    Python 的 traceback 会持有异常发生点各层栈帧的局部变量，其中就包括
    大块中间张量。在 except 块里继续分配显存，这些"本该已释放"的张量
    会一直被引用着无法回收，实测可用显存从 5GB 一路掉到 14MB，
    越重试越失败。所以这里必须让 except 块尽快结束，清理放在它外面。
    """
    n = img.shape[0]
    chunk = n
    start = 0
    outputs = []
    while start < n:
        chunk = max(1, min(chunk, n - start))
        try:
            out = _forward_once(model, img[start:start + chunk])
        except torch.OutOfMemoryError:
            pass                                  # 立刻离开 except，释放 traceback
        else:
            outputs.append(out)
            start += chunk
            continue

        # ---- 下面是 OOM 后的处理，已经跳出 except 作用域 ----
        gc.collect()
        torch.cuda.empty_cache()
        torch.clear_autocast_cache()
        if chunk == 1:
            raise RuntimeError(
                f'推理时逐样本仍显存溢出（模型={utils.format_params(model)}）。'
                f'请调小 config.EVAL_BATCH_SIZE。')
        chunk //= 2
        print(f'[OOM] 推理 batch 溢出，切细到每片 {chunk} 张重试', flush=True)

    return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]


@torch.no_grad()
def predict_loader(model, loader, device, threshold=None, post_min_px=None):
    """
    返回 dict:
        gt / pred : (N, H, W) uint8 二值掩码，正类=球体区域
        labels    : (N,) 0=control 1=treatment
        probs     : (N, H, W) float16 概率图（供阈值分析）

    post_min_px>0 时对预测做连通域后处理(去掉小于该面积的碎点)。GT 不动 ——
    它是评测基准,不该被改;碎点对 GT 指标的影响由"过滤口径"的边界指标单独报告。
    """
    threshold = config.SEG_THRESHOLD if threshold is None else threshold
    if post_min_px is None:
        post_min_px = getattr(config, 'SEG_POST_MIN_COMPONENT_PX', 0)
    model.eval()
    gts, preds, labels, probs = [], [], [], []

    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        logits = _forward_oom_safe(model, img, device)
        prob = torch.sigmoid(logits.float())
        pred = (prob > threshold).to(torch.uint8).cpu().numpy()[:, 0]
        if post_min_px and post_min_px > 0:
            pred = np.stack([boundary.filter_components(p, post_min_px).astype(np.uint8)
                             for p in pred], axis=0)
        preds.append(pred)
        probs.append(prob.half().cpu().numpy()[:, 0])
        gts.append((batch['mask'] > 0.5).to(torch.uint8).numpy()[:, 0])
        labels.append(batch['label'].numpy())

    return {
        'gt': np.concatenate(gts, axis=0),
        'pred': np.concatenate(preds, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'probs': np.concatenate(probs, axis=0),
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def evaluate_model(model, loader, device, out_dir=None, split_name='test',
                   save_predictions=True, log=None):
    """跑一遍推理 + 指标 + 对照报告。"""
    res = predict_loader(model, loader, device)
    gt, pred = res['gt'], res['pred']

    report = M.evaluate_segmentation([g for g in gt], [p for p in pred])
    cmp = M.compare_with_baseline(report)
    table = M.format_table2(report, model_name='OSMOSIS (Ours)')

    # 边界敏感指标:区域指标(Dice/IoU)对轮廓质量不敏感,补 ASSD/HD95/bIoU,
    # 并按球体大小分桶,定位边界退化发生在哪类球体上。
    # 两个口径都算:
    #   - 原始:含 GT 里的碎点(实测 91.7% 连通域 <20px,会污染 max-HD)
    #   - 过滤:两侧都去掉 <MIN_PX 的连通域,衡量"真球体轮廓质量"
    min_px = getattr(config, 'SEG_EVAL_MIN_COMPONENT_PX', 0)
    bnd = boundary.evaluate_boundary([g for g in gt], [p for p in pred], band_d=2)
    bnd_bucket = {
        m: boundary.per_bucket(m, bnd[m]['values'], [g for g in gt])
        for m in bnd}
    bnd_f = (boundary.evaluate_boundary([g for g in gt], [p for p in pred],
                                       band_d=2, min_px=min_px)
             if min_px > 0 else None)

    out = {
        'split': split_name,
        'metrics': report,
        'boundary': bnd,
        'boundary_by_size': bnd_bucket,
        'vs_baseline': cmp,
        'baseline_table2': M.BASELINE_TABLE2,
    }
    if bnd_f is not None:
        out['boundary_filtered'] = bnd_f
        out['boundary_filtered_min_px'] = min_px

    if out_dir:
        utils.ensure_dir(out_dir)
        utils.write_json(os.path.join(out_dir, f'{split_name}_metrics.json'), out)
        with open(os.path.join(out_dir, 'table2.md'), 'w', encoding='utf-8') as f:
            f.write('# HTS-Seg 分割指标对照（与 baseline 论文 Table 2 对齐）\n\n')
            f.write(table + '\n\n')
            f.write('## baseline 论文原值\n\n')
            for name, vals in M.BASELINE_TABLE2.items():
                f.write(f'- {name}: ' + ', '.join(
                    f'{M.METRIC_NAMES[k]}={vals[k]:.4f}' for k in M.TABLE2_ORDER)
                    + '\n')
            f.write('\n## 逐指标差值（Ours - DeepLabv3）\n\n')
            f.write('| 指标 | Ours | DeepLabv3 | Δ | 结论 |\n|---|---|---|---|---|\n')
            for row in cmp['rows']:
                verdict = '优于' if row['better'] else '低于'
                f.write(f"| {row['metric']} | {row['ours']:.4f} | "
                        f"{row['baseline_deeplabv3']:.4f} | "
                        f"{row['delta']:+.4f} | {verdict} |\n")

            f.write('\n## 边界精度指标（像素）\n\n')
            f.write('| 指标 | mean | std | 小球体 | 中球体 | 大球体 |\n'
                    '|---|---|---|---|---|---|\n')
            for m, s in bnd.items():
                buckets = bnd_bucket.get(m, {})
                cell = lambda b: (f"{buckets[b][0]:.3f}±{buckets[b][1]:.3f}"
                                  if b in buckets else '—')
                f.write(f"| {m} | {s['mean']:.3f} | {s['std']:.3f} | "
                        f"{cell('small')} | {cell('medium')} | {cell('large')} |\n")

        if save_predictions:
            np.savez_compressed(
                os.path.join(out_dir, f'{split_name}_predictions.npz'),
                gt=gt, pred=pred, labels=res['labels'], probs=res['probs'])
        _save_qualitative(gt, pred, res['labels'], model, loader, out_dir)

    if log:
        log.log(f'{split_name} 评估', n=report['n_images'],
                **{k: round(report['per_image'][k]['mean'], 4)
                   for k in M.TABLE2_ORDER})
    return out


def _save_qualitative(gt, pred, labels, model, loader, out_dir, n=4):
    """
    存几张 GT / 预测 / 原图的三联图，便于人工核对。

    【这步是"装饰性产物"，绝不能让它弄死实验】实测 10-03:patch8 的实验
    已经跑完 52 个 epoch、测试指标也写盘了，却在画这 6 张图时用
    `model(img)` 直接跑满 batch=64(4096 token 的注意力单次要 4.5GB)，
    CUDA OOM 抛异常 → 整个作业 exit=1 被判失败。所以:
      * 前向改走 _forward_oom_safe(OOM 自动切细);
      * 图数从 6 降到 4;
      * 整段包 try/except —— 画不出来就跳过，作业照常成功。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        fig_dir = utils.ensure_dir(os.path.join(out_dir, 'figures'))
        device = config.DEVICE
        model.eval()
        batch = next(iter(loader))
        img = batch['image'][:n].to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                             enabled=config.SEG_AMP):
            prob = torch.sigmoid(
                _forward_oom_safe(model, img, device).float())[:, 0].cpu().numpy()

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None, :]
        for i in range(min(n, img.shape[0])):
            axes[i, 0].imshow(img[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title(f'input ({"treatment" if labels[i] else "control"})')
            axes[i, 1].imshow(gt[i], cmap='gray')
            axes[i, 1].set_title('ground truth')
            axes[i, 2].imshow(prob[i], cmap='viridis', vmin=0, vmax=1)
            axes[i, 2].set_title('prediction')
            for a in axes[i]:
                a.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'qualitative.png'), dpi=110)
        plt.close(fig)
    except Exception as e:      # 画图失败不影响指标与产物
        print(f'[warn] 定性对比图生成失败(已跳过): {type(e).__name__}: {e}',
              flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    utils.apply_cli_overrides()
    utils.seed_everything(config.SEED)

    eval_dir = utils.ensure_dir(os.path.join(config.DIR_STAGE3, 'eval'))
    log = utils.Logger(os.path.join(eval_dir, 'eval_log.jsonl'))
    device = config.DEVICE

    ckpt = _resolve_ckpt()
    if ckpt is None:
        raise SystemExit(
            f'找不到 Stage3 权重。请先运行 train_seg.py，或指定 --ckpt 路径。\n'
            f'已查找: {config.DIR_STAGE3}/ckpt/{{best,last}}.pt')

    _train_ds, _val_ds, test_ds, meta = build_hts_datasets()
    utils.write_json(os.path.join(eval_dir, 'test_split_meta.json'),
                     {k: v for k, v in meta.items() if k != 'test_stems'})

    model = build_seg_model(device)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state), strict=False)
    log.log('载入 Stage3 权重', path=ckpt)

    loader = utils.build_loader(test_ds, utils.eval_batch_size(), shuffle=False,
                                seed=config.SEED)
    out = evaluate_model(model, loader, device, out_dir=eval_dir,
                         split_name='test', log=log)

    print('\n' + M.format_table2(out['metrics'], 'OSMOSIS (Ours)'))
    bnd = out['boundary']
    print('\n边界精度(像素): ' + '  '.join(
        f"{m}={bnd[m]['mean']:.3f}±{bnd[m]['std']:.3f}" for m in bnd))
    print(f"\n测试集规模: {out['metrics']['n_images']} 张 "
          f"(划分与 baseline 一致, seed={config.SPLIT_SEED})")
    log.close()


def _resolve_ckpt():
    """优先用命令行 --ckpt；否则 Stage3 的 best.pt -> last.pt。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--ckpt' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--ckpt='):
            return a.split('=', 1)[1]
    for name in ('best.pt', 'last.pt'):
        p = os.path.join(config.DIR_STAGE3, 'ckpt', name)
        if os.path.exists(p):
            return p
    return None


if __name__ == '__main__':
    sys.exit(main())
