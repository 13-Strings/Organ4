"""
TTA + 阈值调优评估 —— 不改模型、不重训,只改推理环节的边界精度。

两个手段(都只作用于推理,因此可以和任何训练配置叠加):
  1. **TTA(测试时增强)**:把 8 个二面体变换(4 个 90° 旋转 × 是否翻转)各推理
     一遍,把概率图逆变换回原朝向后取平均。边界处概率最不稳定,平均化能
     明显压低轮廓抖动。
  2. **阈值调优**:固定 0.5 的阈值会让轮廓系统性内缩或外扩。在**验证集**上扫
     阈值、按边界指标选最优,再套用到测试集(不碰测试集,无泄漏)。

用法:
    $PY eval_tta.py                      # 用 Stage3 的 best.pt
    $PY eval_tta.py --ckpt <路径> --metric biou2
    $PY eval_tta.py --no-tta             # 只做阈值调优,便于分离两者贡献

产物:
    {DIR_STAGE3}/eval_tta/{val,test}_metrics.json + summary.json
"""

import config  # noqa: F401  —— 必须最先 import

import os
import sys

import numpy as np
import torch

import boundary
import metrics as M
import utils
from data.hts import build_hts_datasets
from eval_seg import build_seg_model, _forward_oom_safe

# 8 个二面体变换:(rot90 次数, 是否水平翻转)
_DIHEDRAL = [(k, f) for k in range(4) for f in (False, True)]


def _apply(t, k, flip):
    if k:
        t = torch.rot90(t, k, dims=(-2, -1))
    if flip:
        t = torch.flip(t, dims=(-1,))
    return t


def _invert(t, k, flip):
    # 逆变换顺序与正变换相反
    if flip:
        t = torch.flip(t, dims=(-1,))
    if k:
        t = torch.rot90(t, -k, dims=(-2, -1))
    return t


@torch.no_grad()
def predict_probs(model, loader, device, tta=True):
    """
    返回 (probs, gt, labels):
        probs: (N,H,W) float32 概率图(已做 TTA 平均)
        gt   : (N,H,W) uint8 真值
    """
    model.eval()
    all_p, all_g, all_l = [], [], []
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        if tta:
            acc = None
            for k, flip in _DIHEDRAL:
                x = _apply(img, k, flip)
                logits = _forward_oom_safe(model, x, device)
                p = torch.sigmoid(logits.float())
                p = _invert(p, k, flip)
                acc = p if acc is None else acc + p
            prob = acc / len(_DIHEDRAL)
        else:
            prob = torch.sigmoid(_forward_oom_safe(model, img, device).float())
        all_p.append(prob[:, 0].cpu().numpy())
        all_g.append((batch['mask'] > 0.5).to(torch.uint8).numpy()[:, 0])
        all_l.append(batch['label'].numpy())
    return (np.concatenate(all_p), np.concatenate(all_g),
            np.concatenate(all_l))


def _score_at(probs, gt, thr, metric):
    pred = (probs > thr).astype(np.uint8)
    if metric == 'fg_dice':
        v = [M.metri(g, p)['fg_dice'] for g, p in zip(gt, pred)]
    elif metric == 'biou2':
        v = [boundary.boundary_iou(g, p, 2) for g, p in zip(gt, pred)]
    elif metric == 'assd':
        v = [-boundary.assd_hd(g, p)[0] for g, p in zip(gt, pred)]  # 越大越好
    else:
        raise ValueError(metric)
    arr = np.asarray(v, dtype=np.float64)
    return float(np.nanmean(arr))


def tune_threshold(probs, gt, metric='biou2'):
    """在验证集上扫阈值,返回 (最优阈值, 扫描表)。"""
    table = []
    for thr in np.arange(0.30, 0.71, 0.02):
        table.append({'thr': float(round(thr, 2)),
                      'score': _score_at(probs, gt, thr, metric)})
    best = max(table, key=lambda r: r['score'])
    return best['thr'], table


def full_eval_and_vectors(probs, gt, thr, tag, band_d=2):
    """
    给定阈值,一次算完:区域 + 边界全套指标(均值) 与 逐图指标向量。

    为什么合并成一个函数:调用方常常要同时拿"汇总均值"和"逐图向量"
    (配对 t 检验、换阈值复算都靠后者),而两边用的是**同一个阈值**。
    分开写就会把同一套 436 张图的 metrics + boundary 完整跑两遍 ——
    其中 boundary 每张图要做 4 次距离变换(ASSD/HD95),重复一遍会多消耗数分钟。
    这里在同一个 pred 上算一次,两条口径都从这里取,结果与分别调用逐位相同。
    """
    pred = (probs > thr).astype(np.uint8)
    rep = M.evaluate_segmentation(list(gt), list(pred))
    bnd = boundary.evaluate_boundary(list(gt), list(pred), band_d=band_d)
    out = {
        'tag': tag, 'threshold': float(thr),
        'region': {k: rep['per_image'][k]['mean'] for k in M.TABLE2_ORDER},
        'region_fg_dice': rep['per_image']['fg_dice']['mean'],
        'boundary': {k: v['mean'] for k, v in bnd.items()},
        'boundary_by_size': {m: boundary.per_bucket(m, bnd[m]['values'], list(gt))
                             for m in bnd},
    }
    vec = {}
    for k in ('dice', 'jaccard', 'fg_dice'):
        if k in rep['per_image']:
            vec[k] = np.asarray(rep['per_image'][k]['values'], dtype=np.float64)
    for k in ('assd', 'hd95'):
        if k in bnd:
            vec[k] = np.asarray(bnd[k]['values'], dtype=np.float64)
    return out, vec


def full_eval(probs, gt, thr, tag, band_d=2):
    """给定阈值,算区域 + 边界全套指标(均值)。"""
    return full_eval_and_vectors(probs, gt, thr, tag, band_d=band_d)[0]


def _resolve_ckpt():
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


def main():
    utils.apply_cli_overrides()
    utils.seed_everything(config.SEED)
    argv = sys.argv[1:]
    no_tta = '--no-tta' in argv
    metric = 'biou2'
    for i, a in enumerate(argv):
        if a.startswith('--metric'):
            metric = a.split('=', 1)[1] if '=' in a else argv[i + 1]

    device = config.DEVICE
    ckpt = _resolve_ckpt()
    if ckpt is None:
        raise SystemExit('找不到 Stage3 权重,先用 train_seg.py 训练或指定 --ckpt')

    out_dir = utils.ensure_dir(os.path.join(config.DIR_STAGE3, 'eval_tta'))
    log = utils.Logger(os.path.join(out_dir, 'eval_tta_log.jsonl'))
    log.log('TTA 评估启动', ckpt=ckpt, tta=not no_tta, tune_metric=metric)

    _tr, val_ds, test_ds, meta = build_hts_datasets()
    model = build_seg_model(device)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state), strict=False)

    out = {'ckpt': ckpt, 'tta': not no_tta, 'tune_metric': metric,
           'n_val': len(val_ds), 'n_test': len(test_ds)}

    # ---- 验证集:扫阈值 ----
    val_loader = utils.build_loader(val_ds, utils.eval_batch_size(),
                                    shuffle=False, seed=config.SEED)
    vp, vg, _ = predict_probs(model, val_loader, device, tta=not no_tta)
    best_thr, scan = tune_threshold(vp, vg, metric=metric)
    out['val_scan'] = scan
    out['best_thr'] = best_thr
    out['val_at_best'] = full_eval(vp, vg, best_thr, 'val@best')
    out['val_at_050'] = full_eval(vp, vg, 0.50, 'val@0.5')
    log.log('阈值扫描完成', best_thr=best_thr,
            val_biou2_at_best=round(out['val_at_best']['boundary']['biou2'], 4),
            val_biou2_at_050=round(out['val_at_050']['boundary']['biou2'], 4))

    # ---- 测试集:三种配置对照 ----
    test_loader = utils.build_loader(test_ds, utils.eval_batch_size(),
                                     shuffle=False, seed=config.SEED)
    tp, tg, tl = predict_probs(model, test_loader, device, tta=not no_tta)
    out['test_at_050'] = full_eval(tp, tg, 0.50, 'test@0.5(旧口径)')
    out['test_at_best'] = full_eval(tp, tg, best_thr, 'test@tuned')
    np.savez_compressed(os.path.join(out_dir, 'test_probs.npz'),
                        probs=tp.astype(np.float16), gt=tg, labels=tl)

    for k in ('test_at_050', 'test_at_best'):
        r = out[k]
        log.log(k, **{kk: round(vv, 4) for kk, vv in r['region'].items()},
                **{kk: round(vv, 4) for kk, vv in r['boundary'].items()})

    utils.write_json(os.path.join(out_dir, 'summary.json'),
                     {k: v for k, v in out.items() if k != 'val_scan'})
    utils.write_json(os.path.join(out_dir, 'threshold_scan.json'), scan)

    print('\n=== TTA=%s 阈值调优结果 (选阈值的指标=%s) ===' % (not no_tta, metric))
    print(f"验证集最优阈值: {best_thr:.2f}")
    print('| 配置 | Dice | Jaccard | ASSD(px) | HD95(px) | bIoU@2px |')
    print('|---|---|---|---|---|---|')
    for k in ('test_at_050', 'test_at_best'):
        r = out[k]
        print(f"| {r['tag']} | {r['region']['dice']:.4f} | "
              f"{r['region']['jaccard']:.4f} | {r['boundary']['assd']:.3f} | "
              f"{r['boundary']['hd95']:.3f} | {r['boundary']['biou2']:.4f} |")
    print(f'\n结果已写入 {out_dir}/summary.json')
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
