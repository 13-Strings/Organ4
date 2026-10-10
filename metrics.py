"""
评估指标 —— 与 baseline (HTS_Seg-main/HTS_DeepLab.ipynb 的 metri()) 严格对齐。

baseline 原始实现:
    iou  = jaccard_score(true, pred, average='weighted')
    dice = 2*iou / (1 + iou)              # 由 IoU 反推，不是直接算的 Dice
    tp/tn/fn/fp 由整批像素求和
    recall = tp/(tp+fn);  precision = tp/(tp+fp);  accuracy = (tp+tn)/总数
本模块的 `metri()` 与之逐行一致，`dice_from_iou=True` 时 Dice 也按同样方式反推。

"""

import numpy as np
from sklearn.metrics import jaccard_score

# 论文 Table 2 的列顺序
TABLE2_ORDER = ['precision', 'recall', 'dice', 'jaccard', 'accuracy']

METRIC_NAMES = {
    'accuracy': 'Accuracy', 'jaccard': 'Jaccard index (IoU)',
    'dice': 'Dice', 'recall': 'Recall', 'precision': 'Precision',
}


# ---------------------------------------------------------------------------
# baseline 原版
# ---------------------------------------------------------------------------


def _is_binary(x):
    """元素是否全在 {0, 1}（空数组视为是）。"""
    return x.size == 0 or (x.min() >= 0 and x.max() <= 1)


def _weighted_jaccard_from_counts(tp, tn, fn, fp):
    """
    sklearn.jaccard_score(..., average='weighted', zero_division=0) 的等价实现。

    为什么不用 sklearn 原版：它内部会对**整条数组**做 np.unique 排序 +
    multilabel_confusion_matrix。逐图调用（262144 元素）勉强能忍，但 pooled
    口径要把 436 张 512x512 堆成一条 1.14e8 元素的数组，一次调用就是分钟级，
    在负载高的机器上能把整个评估拖死（实测光 np.stack 就要 40s+）。
    而二分类的 weighted Jaccard 本来只是几个计数比值：

        前景类: TP/(TP+FP+FN)      背景类: TN/(TN+FN+FP)
        权重  : 各类在真值里的出现次数（前景 TP+FN，背景 TN+FP）
        weighted = (w0*J0 + w1*J1) / (w0 + w1)

    zero_division=0 的语义是"某类分母为 0 时该类记 0" —— 而那种情况下它的
    权重也必为 0，所以直接记 0 与 sklearn 一致。
    """
    j_fg = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    j_bg = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0.0
    w_fg = tp + fn
    w_bg = tn + fp
    w = w_fg + w_bg
    return (w_fg * j_fg + w_bg * j_bg) / w if w > 0 else 0.0


def metrics_from_counts(tp, tn, fn, fp, dice_from_iou=True, eps=1e-12, iou=None):
    """
    由混淆矩阵四个计数直接出全部指标。

    与"把图堆成一条巨数组再跑一遍 metri()"逐位等价：pooled 的 tp/tn/fn/fp
    本来就是逐图求和，先堆数组再统计只是多做了一次 1.14e8 元素的搬运。
    iou 显式传入时用它（非 0/1 输入的 sklearn 回退路径）。
    """
    if iou is None:
        iou = _weighted_jaccard_from_counts(tp, tn, fn, fp)
    dice = 2.0 * iou / (1.0 + iou) if dice_from_iou \
        else 2 * tp / ((tp + fn) + (tp + fp) + eps)
    return {
        'accuracy': (tp + tn) / (tp + tn + fp + fn + eps),
        'jaccard': iou,
        'dice': dice,
        'recall': tp / (tp + fn + eps),
        'precision': tp / (tp + fp + eps),
        # 便于单独核对前景类（球体）的重叠度，不参与论文表格
        'fg_iou': tp / (tp + fp + fn + eps),
        'fg_dice': 2 * tp / (2 * tp + fp + fn + eps),
    }


def counts_of(t, p):
    """逐像素混淆矩阵 (tp, tn, fn, fp)。"""
    tb = t > 0
    pb = p > 0
    return (float(np.sum(tb & pb)), float(np.sum(~tb & ~pb)),
            float(np.sum(tb & ~pb)), float(np.sum(~tb & pb)))


def metri_core(t, p, dice_from_iou=True, eps=1e-12):
    """
    metri() 的内核：输入已拉平的数组，返回 (指标 dict, (tp, tn, fn, fp))。

    把计数一并返回，是为了让 evaluate_segmentation 能直接**逐图累加**出
    pooled 口径，省掉把 436 张图堆成一条 1.14e8 元素数组的那一步。
    """
    tp, tn, fn, fp = counts_of(t, p)
    iou = None
    if not (_is_binary(t) and _is_binary(p)):
        # 非 0/1 输入（如严重程度分级）保持 sklearn 原口径，不进快路径
        iou = float(jaccard_score(t, p, average='weighted', zero_division=0))
    m = metrics_from_counts(tp, tn, fn, fp, dice_from_iou=dice_from_iou,
                            eps=eps, iou=iou)
    return m, (tp, tn, fn, fp)


def metri(true_masks, predicted_masks, dice_from_iou=True, eps=1e-12):
    """
    与 baseline 的 metri() 完全一致。

    true_masks / predicted_masks: 任意形状的 0/1 数组（会被拉平后整体统计）。
    返回 dict。baseline 的返回顺序是 (accuracy, iou, dice, recall, precision)。
    """
    t = np.asarray(true_masks).reshape(-1)
    p = np.asarray(predicted_masks).reshape(-1)
    return metri_core(t, p, dice_from_iou=dice_from_iou, eps=eps)[0]


# _dice_direct() 已并入 metrics_from_counts（dice_from_iou=False 那一支），
# 公式相同：2*TP / ((TP+FN) + (TP+FP))。留两份容易改一处漏一处。


# ---------------------------------------------------------------------------
# 完整评估
# ---------------------------------------------------------------------------


def evaluate_segmentation(gt_list, pred_list, dice_from_iou=True):
    """
    逐图算指标再取 mean±std（对应论文 Table 2 的写法），
    同时给出 baseline 原味的"整批像素汇总"结果。

    gt_list / pred_list: 每个元素是一张图的 0/1 掩码（任意形状，尺寸需一致）。
    """
    assert len(gt_list) == len(pred_list), '真值与预测数量不一致'
    n = len(gt_list)
    if n == 0:
        return {'n_images': 0, 'per_image': {}, 'pooled': {}}

    per_image = {k: [] for k in METRIC_NAMES}
    extras = {'fg_iou': [], 'fg_dice': []}
    tot = [0.0, 0.0, 0.0, 0.0]          # tp, tn, fn, fp 逐图累加 = pooled 口径

    for gt, pred in zip(gt_list, pred_list):
        m, c = metri_core(np.asarray(gt).reshape(-1),
                          np.asarray(pred).reshape(-1),
                          dice_from_iou=dice_from_iou)
        for i in range(4):
            tot[i] += c[i]
        for k in per_image:
            per_image[k].append(m[k])
        for k in extras:
            extras[k].append(m[k])

    out = {'n_images': n, 'per_image': {}, 'pooled': {}}
    for k, vals in per_image.items():
        arr = np.asarray(vals, dtype=np.float64)
        out['per_image'][k] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                               'values': arr.tolist()}
    for k, vals in extras.items():
        arr = np.asarray(vals, dtype=np.float64)
        out['per_image'][k] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                               'values': arr.tolist()}

    # pooled：逐图计数求和 == 把所有图堆成一条再统计，结果逐位相同，
    # 但省掉了 np.stack 那 1.14e8 个元素（本机负载高时光这一步就要 40s+）。
    out['pooled'] = metrics_from_counts(*tot, dice_from_iou=dice_from_iou)
    return out


def format_table2(result, model_name='Ours'):
    """把评估结果排成与论文 Table 2 相同列顺序的 Markdown 表格行。"""
    pi = result['per_image']
    cells = []
    for k in TABLE2_ORDER:
        if k in pi:
            cells.append(f"{pi[k]['mean']:.4f} ± {pi[k]['std']:.4f}")
        else:
            cells.append('—')
    header = '| Model | ' + ' | '.join(METRIC_NAMES[k] for k in TABLE2_ORDER) + ' |'
    sep = '|' + '---|' * (len(TABLE2_ORDER) + 1)
    row = f'| {model_name} | ' + ' | '.join(cells) + ' |'
    return '\n'.join([header, sep, row])


# baseline 论文 Table 2 的参考值（用于对齐核对）
BASELINE_TABLE2 = {
    'U-Net':      {'precision': 0.9289, 'recall': 0.9516, 'dice': 0.9356,
                   'jaccard': 0.8799, 'accuracy': 0.9797},
    'FCN':        {'precision': 0.9692, 'recall': 0.9025, 'dice': 0.9345,
                   'jaccard': 0.8804, 'accuracy': 0.9789},
    'DeepLabv3':  {'precision': 0.9408, 'recall': 0.9551, 'dice': 0.9532,
                   'jaccard': 0.9117, 'accuracy': 0.9828},
    'Mask R-CNN': {'precision': 0.9380, 'recall': 0.9585, 'dice': 0.9510,
                   'jaccard': 0.9080, 'accuracy': 0.9822},
    'YOLOv12-Seg': {'precision': 0.9505, 'recall': 0.8890, 'dice': 0.9180,
                    'jaccard': 0.8520, 'accuracy': 0.9778},
}
BASELINE_BEST = 'DeepLabv3'          # 论文最优


def compare_with_baseline(result, model_name='Ours'):
    """
    与论文最优模型 DeepLabv3 逐指标对比，返回差异表。
    delta > 0 表示优于 baseline。
    """
    ref = BASELINE_TABLE2[BASELINE_BEST]
    pi = result.get('per_image', {})
    rows = []
    for k in TABLE2_ORDER:
        ours = pi.get(k, {}).get('mean', float('nan'))
        rows.append({
            'metric': METRIC_NAMES[k],
            'ours': ours,
            'baseline_deeplabv3': ref[k],
            'delta': ours - ref[k],
            'better': ours > ref[k],
        })
    return {'model': model_name, 'reference': BASELINE_BEST, 'rows': rows}
