#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_script.py —— 一键复现 HTS-Seg 测试集结果（本提交包的主运行脚本）。

跑什么：
    单模型      : weights/s3_base.pt，阈值 0.5，不做 TTA          —— 「单模型」口径
    概率集成    : weights/s3_best_seed{42,43,44}.pt，8 向 TTA，
                  阈值由验证集按 bIoU@2 调出                      —— 「集成增强」口径

评估口径与训练脚本完全一致（同一个 metrics.evaluate_segmentation /
boundary.evaluate_boundary），所以打印出的五项数字就是
`train_seg.py` 写进 train_log.jsonl 的那条 `测试集结果` 记录的定义：
    逐图算 precision/recall/dice/jaccard/accuracy，再对 436 张取算术平均。

用法:
    python test_script.py                        # 两种口径都跑
    python test_script.py --mode single          # 只跑单模型（实测约 3 分钟）
    python test_script.py --mode ensemble        # 只跑集成（实测约 42 分钟）
    python test_script.py --mode single \\
        --single-ckpt outputs/organ_v5/.../s3-.../ckpt/best.pt   # 换成自己训的权重
    python test_script.py --DEVICE_ID 3 --EVAL_BATCH_SIZE 8      # 任意 config 变量

产物:
    outputs/test_results/test_results.json    逐口径完整结果 + 逐图向量
    outputs/test_results/test_results.md      人读的汇总表
"""

import argparse
import os
import sys
import time

import config  # noqa: F401  —— 必须最先 import：config 在 import 时写 CUDA_*/PYTHONHASHSEED

import evaluate as E

ROOT = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(ROOT, 'weights')
OUT_DIR = os.path.join(ROOT, 'outputs', 'test_results')

# 已发表的口径对照（见 README「测试集结果」一节）
REFERENCE = {
    'single': {'label': '单模型（参考 run，224px / deconv 头）',
               'precision': 0.9518, 'recall': 0.9530, 'dice': 0.9845,
               'jaccard': 0.9697, 'accuracy': 0.9843},
    'ensemble': {'label': '3 种子集成 + TTA + 验证集调阈',
                 'precision': 0.9540, 'recall': 0.9607, 'dice': 0.9862,
                 'jaccard': 0.9729, 'accuracy': 0.9860},
}


def _fmt_row(name, r):
    return '| ' + name + ' | ' + ' | '.join(f'{r[m]:.4f}' for m in E.HEAD_METRICS) + ' |'


def main():
    ap = argparse.ArgumentParser(description='复现 HTS-Seg 测试集结果')
    ap.add_argument('--mode', choices=['single', 'ensemble', 'all'], default='all')
    ap.add_argument('--single-ckpt', default=os.path.join(W, 's3_base.pt'),
                    help='单模型口径用哪个权重（默认 weights/s3_base.pt）')
    ap.add_argument('--ensemble-ckpts', nargs='+', default=None,
                    help='集成口径的权重列表（默认三个种子权重）')
    ap.add_argument('--out', default=OUT_DIR)
    args, unknown = ap.parse_known_args()

    utils_argv = unknown
    E.utils.apply_cli_overrides(utils_argv)
    E.utils.seed_everything(config.SEG_SEED)

    ens_ckpts = args.ensemble_ckpts or [os.path.join(W, f's3_best_seed{s}.pt')
                                        for s in (42, 43, 44)]
    results = {}
    t_all = time.time()

    print('=' * 78)
    print(f'OSMOSIS 测试集复现   设备={config.DEVICE}  '
          f'HTS_IMG_SIZE={config.HTS_IMG_SIZE}  '
          f'测试集={config.HTS_SUBSET}')
    print('=' * 78)

    if args.mode in ('single', 'all'):
        print('\n[1/2] 单模型口径 …', flush=True)
        results['single'] = E.evaluate([args.single_ckpt], split='test',
                                       tta=False, thr=None, tune=False)

    if args.mode in ('ensemble', 'all'):
        print('\n[2/2] 概率集成口径 …', flush=True)
        missing = [c for c in ens_ckpts if not os.path.exists(c)]
        if missing:
            print('[warn] 缺少权重，跳过集成：')
            for c in missing:
                print('   ', c)
        else:
            results['ensemble'] = E.evaluate(ens_ckpts, split='test',
                                             tta=True, thr=None, tune=True)

    # ---------------- 汇总 ----------------
    if not results:
        print('\n[错误] 没有任何口径跑成功（权重是否缺失？见上面的 [warn]）。')
        return 1
    n_images = next(iter(results.values()))['n_images']
    lines = ['# HTS-Seg 测试集结果（本包实跑）', '',
             f'- 数据集划分：`{config.HTS_SUBSET}`，测试集 '
             f'{n_images} 张，'
             f'split_seed={config.SPLIT_SEED}（与 baseline 论文一致）',
             f'- 输入分辨率：{config.HTS_IMG_SIZE}',
             '- 指标定义：逐图算 precision/recall/dice/jaccard/accuracy 后取算术平均'
             '（`metrics.evaluate_segmentation`，与 baseline 的 `metri()` 同级）', '',
             '| 口径 | ' + ' | '.join(m.capitalize() for m in E.HEAD_METRICS)
             + ' | ASSD(px) | HD95(px) | 前景Dice |',
             '|---|---|---|---|---|---|---|---|---|']
    print('\n' + '=' * 78)
    print('汇总（本包实跑）')
    print('=' * 78)
    for key in ('single', 'ensemble'):
        if key not in results:
            continue
        out = results[key]
        r, b = out['result']['region'], out['result']['boundary']
        name = '单模型' if key == 'single' else f'集成({out["n_models"]} 模型)'
        note = '' if not out['tta'] else ' + TTA'
        row = (f'| {name}{note}（thr={out["threshold"]:.2f}） | '
               + ' | '.join(f'{r[m]:.4f}' for m in E.HEAD_METRICS)
               + f" | {b.get('assd', float('nan')):.3f} "
               f"| {b.get('hd95', float('nan')):.3f} "
               f"| {out['result']['region_fg_dice']:.4f} |")
        lines.append(row)
        print(row)

    print('\n对照（已发表口径）')
    print('| 口径 | ' + ' | '.join(m.capitalize() for m in E.HEAD_METRICS) + ' |')
    print('|---|---|---|---|---|---|')
    for key in ('single', 'ensemble'):
        ref = REFERENCE[key]
        row = '| ' + ref['label'] + ' | ' + ' | '.join(
            f'{ref[m]:.4f}' for m in E.HEAD_METRICS) + ' |'
        print(row)
    lines += ['', '## 对照（已发表口径）', '',
              '| 口径 | ' + ' | '.join(m.capitalize() for m in E.HEAD_METRICS) + ' |',
              '|---|---|---|---|---|---|']
    for key in ('single', 'ensemble'):
        ref = REFERENCE[key]
        lines.append('| ' + ref['label'] + ' | ' + ' | '.join(
            f'{ref[m]:.4f}' for m in E.HEAD_METRICS) + ' |')

    out_dir = E.utils.ensure_dir(args.out)
    E.utils.write_json(os.path.join(out_dir, 'test_results.json'), results)
    with open(os.path.join(out_dir, 'test_results.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n耗时 {time.time() - t_all:.0f}s')
    print(f'结果已写入 {out_dir}/test_results.json 与 test_results.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
