"""
边界精度工具:边界敏感指标 + 边界感知损失 + GT 边界统计分析。

背景:现有评估(metrics.py)只有 Dice / IoU / Acc 等区域指标,对轮廓质量
不敏感 —— 同样的 Dice 下,轮廓可以差几个像素。本模块补齐:

1. 指标(纯 numpy + scipy):
   - ASSD    平均对称表面距离
   - HD95    Hausdorff 距离 95 分位(对离群轮廓点鲁棒)
   - bIoU(d) 边界带 IoU:GT 与预测各自取 d 像素宽的轮廓带,在带上算 IoU
   - 支持按 GT 面积分桶(小球体/中/大),定位"哪类球体边界退化"

2. 损失(torch,与 train_seg.py 集成):
   - 'band'     边界带加权 BCE:权重 w = 1 + alpha * exp(-D^2/(2*sigma^2)),
                D = 像素到 GT 边界的距离。梯度集中在轮廓附近,稳定易调。
   - 'kervadec' Kervadec et al. 2019 边界损失(符号距离区域形式):
                L = mean(prob * phi), phi = GT 符号距离(内负外正)。
                梯度方向把预测边界向 GT 边界推。

3. GT 分析(analyze_gt):球体大小分布、贴边球体比例等。
"""

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# 距离图
# ---------------------------------------------------------------------------


def gt_dist_map(mask):
    """
    到 GT **轮廓**的距离:边界上为 0,向内向外都递增。单位像素。

    问题(曾导致边界损失完全失效):不能用 `distance_transform_edt(mask)` ——
    它算的是"到最近背景像素的距离",于是**所有背景像素都是 0**,权重图变成
    "整个背景被加权 α+1 倍",而不是轮廓附近的窄带。实测后果:B 组(边界带加权)
    相对基线没有稳定优势。
    正确做法是两个方向的 EDT 取逐点最大:
        distance_transform_edt(m)   前景内:到背景的距离(背景处为 0)
        distance_transform_edt(~m)  背景处:到前景的距离(前景处为 0)
        max(...)  -> 边界上恒为 0,两侧都单调递增
    这样 `signed_dist`(内负外正)也随之正确 —— 原实现在 GT 外部恒为 0,
    导致 Kervadec 损失只会推高内部概率、从不从外部压制。
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return np.zeros(m.shape, dtype=np.float64)
    if m.all():
        return np.zeros(m.shape, dtype=np.float64)
    d_in = ndimage.distance_transform_edt(m)      # 前景 -> 背景
    d_out = ndimage.distance_transform_edt(~m)    # 背景 -> 前景
    return np.maximum(d_in, d_out)


def signed_dist(mask, dist=None):
    """GT 符号距离:内部为负、外部为正(Kervadec 口径)。"""
    d = gt_dist_map(mask) if dist is None else dist
    m = np.asarray(mask) > 0
    return np.where(m, -d, d)


def boundary_band_weight(mask, sigma=3.0, alpha=5.0, dist=None):
    """边界带权重 w = 1 + alpha * exp(-D^2 / (2 sigma^2))。"""
    d = gt_dist_map(mask) if dist is None else dist
    return 1.0 + alpha * np.exp(-(d ** 2) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# 连通域过滤
# ---------------------------------------------------------------------------


def filter_components(mask, min_px=0):
    """
    去掉面积小于 min_px 的连通域。

    为什么需要:实测 HTS-Seg 的 GT 掩码里 **91.7% 的连通域小于 20px**
    (82% 只有 1~5px,中位面积 1px),但它们只占前景面积的 0.1%;
    真正球体占 99.77%。这些单像素噪点会:
      * 让 max-Hausdorff 完全失真(模型没预测到某个 1px 噪点,距离直接爆表);
      * 污染形态学特征 —— complexity = P²/(4πA) 是边界长度的一阶量,
        几十个碎点能把 630px 的轮廓周长抬到 700+,使复杂度偏差 25%。
    """
    m = np.asarray(mask) > 0
    if min_px <= 0 or not m.any():
        return m
    lbl, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)
    keep = sizes >= min_px
    keep[0] = False
    return keep[lbl]


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def _surface(m):
    """1 像素宽的"内边界"带:m ^ erosion(m)。"""
    return m ^ ndimage.binary_erosion(m, border_value=0)


def _sd(a, b):
    """
    a 的每个边界像素到 b 边界的最小距离(边界到边界)。
    scipy 的 EDT(x) 是"到最近 False 像素的距离",所以 EDT(~b_s)
    = 到最近 b 边界像素的距离;在 a 的边界像素上采样即为所求。
    任一侧为空返回 None。
    """
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    if not a.any() or not b.any():
        return None
    a_s = _surface(a)
    b_s = _surface(b)
    if not a_s.any() or not b_s.any():
        return None
    d = ndimage.distance_transform_edt(~b_s)
    return d[a_s]


def assd_hd(gt, pred):
    """返回 (assd, hd95, hd),单位像素。任一侧为空返回 (nan, nan, nan)。"""
    d_gt_pred = _sd(gt, pred)
    d_pred_gt = _sd(pred, gt)
    if d_gt_pred is None or d_pred_gt is None:
        return float('nan'), float('nan'), float('nan')
    assd = (d_gt_pred.mean() + d_pred_gt.mean()) / 2.0
    both = np.concatenate([d_gt_pred, d_pred_gt])
    return float(assd), float(np.percentile(both, 95)), float(both.max())


def boundary_iou(gt, pred, d=2):
    """
    边界带 IoU:d 像素宽的轮廓带上的重叠。d=0 退化为前景 IoU。
    两侧带都为空(全空图)记为 1.0。
    """
    g = np.asarray(gt) > 0
    p = np.asarray(pred) > 0
    if not g.any() and not p.any():
        return 1.0
    g_band = (g ^ ndimage.binary_erosion(g, iterations=d, border_value=0)) \
        if g.any() else np.zeros_like(g)
    p_band = (p ^ ndimage.binary_erosion(p, iterations=d, border_value=0)) \
        if p.any() else np.zeros_like(p)
    inter = (g_band & p_band).sum()
    union = (g_band | p_band).sum()
    return float(inter) / max(union, 1)


def evaluate_boundary(gt_list, pred_list, band_d=2, min_px=0, bands=(2, 5, 10)):
    """
    多图边界指标,返回 {metric: {mean, std, values}}。

    为什么给多个带宽 d:边界带 IoU 是**严格**指标 —— d=2 时轮廓只要偏移
    2px,两个带就完全不重叠,IoU 直接归零(实测同心圆 R60 vs R58 得 0.0)。
    所以同时报 d=2 / 5 / 10,分别对应"亚像素级贴合""数像素级贴合"
    "大致贴合但边缘发虚"三种质量,单看一个数会误判。
    min_px > 0 时先对 GT 与预测都做连通域过滤(去掉碎点),再算指标 ——
    这是衡量"真球体轮廓质量"的口径;min_px=0 是原始口径(含噪点)。
    """
    bset = sorted({band_d} | set(bands))
    keys = ['assd', 'hd95', 'hd'] + [f'biou{d}' for d in bset]
    vals = {k: [] for k in keys}
    for g, p in zip(gt_list, pred_list):
        if min_px > 0:
            g = filter_components(g, min_px)
            p = filter_components(p, min_px)
        a, h95, h = assd_hd(g, p)
        vals['assd'].append(a)
        vals['hd95'].append(h95)
        vals['hd'].append(h)
        for d in bset:
            vals[f'biou{d}'].append(boundary_iou(g, p, d))
    out = {}
    for k, v in vals.items():
        arr = np.asarray(v, dtype=np.float64)
        out[k] = {'mean': float(np.nanmean(arr)),
                  'std': float(np.nanstd(arr)),
                  'values': [float(x) for x in arr]}
    return out


def size_buckets(gt_list):
    """按前景面积三分位分桶,返回 {small|medium|large: [idx]}。"""
    areas = [float(np.count_nonzero(np.asarray(g) > 0)) for g in gt_list]
    out = {'small': [], 'medium': [], 'large': []}
    if not areas:
        return out
    q = np.quantile(areas, [1 / 3, 2 / 3])
    for i, a in enumerate(areas):
        key = 'small' if a <= q[0] else ('medium' if a <= q[1] else 'large')
        out[key].append(i)
    return out


def per_bucket(metric_name, values, gt_list):
    """按大小桶汇总某个逐图指标,返回 {bucket: mean±std}。"""
    arr = np.asarray(values, dtype=np.float64)
    out = {}
    for b, idx in size_buckets(gt_list).items():
        if idx:
            v = arr[idx]
            out[b] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return out


# ---------------------------------------------------------------------------
# 损失(torch)
# ---------------------------------------------------------------------------


def boundary_weighted_bce(logits, mask, dist, sigma=3.0, alpha=5.0,
                          pos_weight=None):
    """
    边界带加权 BCE。dist: (B,1,H,W) 距离图 tensor。
    权重 w = 1 + alpha * exp(-D^2 / (2 sigma^2)),BCE 自身带 reduction='none'
    再乘 w,最后均值。
    """
    import torch
    import torch.nn.functional as F

    w = 1.0 + alpha * torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    w = w.expand_as(logits)
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, mask, weight=w, pos_weight=pos_weight, reduction='mean')


def kervadec_boundary_loss(prob, mask, dist, clip=20.0):
    """
    Kervadec et al. 2019 区域形式:
        L = mean( prob * phi ),  phi = GT 符号距离(内负外正)。

    **必须截断距离**:原始 φ_G 无界(512² 图里远处背景能到 300+ px),
    一个概率 0.01 的远处背景像素会贡献 3.0,累积起来会压垮真实边界附近的
    梯度,把训练带偏。截断到 clip 像素后,损失只关心轮廓一定范围内的
    内缩/外扩 —— 这也是原论文在实现里对距离做有界化的原因。
    """
    import torch
    phi = torch.where(mask > 0.5, -dist, dist)
    if clip and clip > 0:
        phi = torch.clamp(phi, min=-clip, max=clip)
    return (prob * phi).mean()
