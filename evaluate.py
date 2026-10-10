#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py —— 统一评估入口（单模型 / 多模型概率集成；测试集 / 验证集）。

本文件是"提交包"新增的脚手架，**不修改 core 目录里 code_v5 的任何文件**：
所有指标都由 code_v5 自己的代码算 ——
    metrics.evaluate_segmentation   （区域指标，与 baseline paper 的 metri() 逐行一致）
    boundary.evaluate_boundary      （边界指标 ASSD/HD95/bIoU）
    eval_tta.full_eval_and_vectors  （区域 + 边界一次算完，附带逐图向量）
因此这里打印出的数字与 `train_seg.py` 训练结束时写进 `train_log.jsonl` 的
「测试集结果」记录、以及 `eval/table2.md`，是**同一套口径**。

用法:
    # 1) 单模型，阈值取 config.SEG_THRESHOLD(=0.5)，不做 TTA
    python evaluate.py --ckpt weights/s3_base.pt

    # 2) 三个种子做概率平均集成 + 8 向 TTA + 验证集调阈
    python evaluate.py --ckpt weights/s3_best_seed42.pt \\
                                weights/s3_best_seed43.pt \\
                                weights/s3_best_seed44.pt \\
                       --tta --tune-threshold

    # 3) 换设备 / 换分辨率 / 换分割头（任意 config 大写变量都能在命令行覆盖）
    python evaluate.py --ckpt weights/s3_base.pt --DEVICE_ID 3 --HTS_IMG_SIZE 512

注意 `--HTS_IMG_SIZE` / `--SEG_DECODER` 会改变模型结构，必须与权重训练时一致；
不传 `--decoder` 时本脚本会从权重里**自动嗅探**分割头类型（deconv / bilinear），
避免"第二个模型才报 size mismatch"这类问题。
"""

import argparse
import json
import os
import sys
import time

import config  # noqa: F401  —— 必须最先 import：config 在 import 时写 CUDA_*/PYTHONHASHSEED

import numpy as np
import torch

import utils
from data.hts import build_hts_datasets
from eval_seg import build_seg_model
from eval_tta import full_eval_and_vectors, predict_probs, tune_threshold

SPLIT_IDX = {'train': 0, 'val': 1, 'test': 2}
HEAD_METRICS = ('precision', 'recall', 'dice', 'jaccard', 'accuracy')
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'outputs', 'eval')


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------


def sniff_decoder(state):
    """从权重里判断分割头类型。

    bilinear 头：`seg_head.ups.0.1` 是 Conv2d(256,256,3,3)，weight 是 4 维；
    deconv 头  ：同一位置是 GroupNorm，weight 是 1 维 (256,)。
    混用会在**装载第二个模型时**才炸 size mismatch（strict=False 不放过形状不符），
    所以在建模型之前先判一次。
    """
    w = None
    for k in ('seg_head.ups.0.1.weight', 'seg_head.ups.0.1.bias'):
        if k in state:
            w = state[k]
            break
    if w is None:
        return None
    return 'bilinear' if getattr(w, 'dim', lambda: 0)() == 4 else 'deconv'


def explicit_decoder(argv, args):
    """用户显式指定的分割头：`--decoder` 或 `--SEG_DECODER` 任意一种。"""
    if args.decoder and args.decoder != 'auto':
        return args.decoder
    for i, a in enumerate(argv):
        if a == '--SEG_DECODER' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--SEG_DECODER='):
            return a.split('=', 1)[1]
    return None


def load_model(ckpt, device, decoder=None, verbose=True):
    """按（可能的）自动嗅探结果新建模型并载入权重。"""
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    if decoder is None:
        sniffed = sniff_decoder(state)
        if sniffed and sniffed != config.SEG_DECODER:
            if verbose:
                print(f'[decoder] 权重为 {sniffed} 头，config.SEG_DECODER='
                      f'{config.SEG_DECODER} → 已自动切换为 {sniffed}',
                      flush=True)
            config.SEG_DECODER = sniffed
    elif decoder != config.SEG_DECODER:
        config.SEG_DECODER = decoder
    model = build_seg_model(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose and (missing or unexpected):
        print(f'[warn] {os.path.basename(ckpt)}: missing={len(missing)} '
              f'unexpected={len(unexpected)}', flush=True)
    model.eval()
    return model


def probs_for(ckpt, device, split, tta, decoder=None, verbose=True):
    """返回 (probs, gt, labels)。probs 是 (N,H,W) float32 概率图。"""
    model = load_model(ckpt, device, decoder=decoder, verbose=verbose)
    ds = build_hts_datasets()[SPLIT_IDX[split]]
    loader = utils.build_loader(ds, utils.eval_batch_size(), shuffle=False,
                                seed=config.SPLIT_SEED)
    p, g, l = predict_probs(model, loader, device, tta=tta)
    del model
    torch.cuda.empty_cache()
    return p, g, l


# ---------------------------------------------------------------------------
# 评估主逻辑（test_script.py 也复用这里）
# ---------------------------------------------------------------------------


def evaluate(ckpts, split='test', tta=False, thr=None, tune=False, decoder=None,
             verbose=True):
    """对一个或多个权重做评估。多个权重时先各自出概率图再取平均（概率集成）。

    返回 dict，可直接 json.dump。
    """
    device = config.DEVICE
    out = {'split': split, 'ckpts': [os.path.abspath(c) for c in ckpts],
           'n_models': len(ckpts), 'tta': bool(tta),
           'img_size': config.HTS_IMG_SIZE,
           'split_seed': config.SPLIT_SEED, 'subset': config.HTS_SUBSET}

    probs_list, per_model = [], []
    gt = labels = None
    for i, c in enumerate(ckpts):
        t0 = time.time()
        p, g, l = probs_for(c, device, split, tta, decoder=decoder, verbose=verbose)
        gt, labels = g, l
        r, _vec = full_eval_and_vectors(p, gt, config.SEG_THRESHOLD, f'model{i}')
        r['ckpt'] = os.path.abspath(c)
        r['seconds'] = round(time.time() - t0, 1)
        per_model.append(r)
        probs_list.append(p)
        if verbose:
            print(f'  [模型 {i}] {os.path.basename(c):<28} '
                  f'Dice={r["region"]["dice"]:.4f} Jaccard={r["region"]["jaccard"]:.4f} '
                  f'Acc={r["region"]["accuracy"]:.4f} fgDice={r["region_fg_dice"]:.4f} '
                  f'({r["seconds"]}s)', flush=True)
        del p
    out['per_model'] = per_model
    out['n_images'] = int(gt.shape[0])
    # 自动嗅探可能改过分割头，落盘的是**实际使用**的那个
    out['decoder'] = config.SEG_DECODER

    # ---- 概率平均 ----
    acc = np.zeros_like(probs_list[0], dtype=np.float32)
    for p in probs_list:
        acc += p
    ens = acc / len(probs_list)
    del probs_list, acc

    # ---- 阈值 ----
    if thr is not None:
        use_thr, thr_src = float(thr), 'cli'
    elif tune:
        vp, vg, _ = (ens, gt, None) if split == 'val' else \
            _val_probs(ckpts, device, tta, decoder, verbose)
        use_thr, scan = tune_threshold(vp, vg, metric='biou2')
        out['val_threshold_scan'] = scan
        thr_src = 'val:biou2'
        if verbose:
            print(f'  [阈值] 用验证集 biou2 调出 {use_thr:.2f}', flush=True)
    else:
        use_thr, thr_src = float(config.SEG_THRESHOLD), 'config.SEG_THRESHOLD'
    out['threshold'] = use_thr
    out['threshold_source'] = thr_src

    result, vec = full_eval_and_vectors(ens, gt, use_thr,
                                        'ensemble' if len(ckpts) > 1 else 'single')
    out['result'] = result
    out['per_image_vectors'] = {k: [float(x) for x in v] for k, v in vec.items()}
    if labels is not None:
        out['labels'] = [int(x) for x in labels]

    if verbose:
        print_result(out)
    return out


def _val_probs(ckpts, device, tta, decoder, verbose):
    """调阈值需要验证集上的集成概率图（每个模型各跑一遍验证集）。"""
    acc = None
    for c in ckpts:
        p, g, _ = probs_for(c, device, 'val', tta, decoder=decoder, verbose=False)
        acc = p if acc is None else acc + p
    if verbose:
        print(f'  [阈值] 已在 {acc.shape[0]} 张验证集上取集成概率图', flush=True)
    return acc / len(ckpts), g, None


def print_result(out):
    r = out['result']['region']
    b = out['result']['boundary']
    print()
    print(f"=== {out['split']} 评估 ===")
    print(f"  划分      : {out['subset']} / {out['split']} / "
          f"{out['n_images']} 张 (split_seed={out['split_seed']})")
    print(f"  输入分辨率: {out['img_size']}    分割头: {out['decoder']}")
    print(f"  模型数    : {out['n_models']}    TTA: {'是' if out['tta'] else '否'}   "
          f"阈值: {out['threshold']:.2f} ({out['threshold_source']})")
    print()
    print('  | 指标 | ' + ' | '.join(m.capitalize() for m in HEAD_METRICS) + ' |')
    print('  |---|' + '---|' * len(HEAD_METRICS))
    print('  | 结果 | ' + ' | '.join(f'{r[m]:.4f}' for m in HEAD_METRICS) + ' |')
    print()
    print(f"  前景 Dice = {out['result']['region_fg_dice']:.4f}")
    print('  边界(px)  : ' + '  '.join(f'{k}={b[k]:.3f}'
                                       for k in ('assd', 'hd95', 'hd') if k in b))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description='OSMOSIS 统一评估入口（单模型 / 概率集成）')
    ap.add_argument('--ckpt', nargs='+', required=True, help='一个或多个权重路径')
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--tta', action='store_true', help='8 向 dihedral TTA')
    ap.add_argument('--tune-threshold', dest='tune', action='store_true',
                    help='在验证集上按 bIoU@2 调阈值（比固定 0.5 更优）')
    ap.add_argument('--thr', type=float, default=None, help='直接指定阈值')
    ap.add_argument('--decoder', choices=['auto', 'deconv', 'bilinear'],
                    default='auto', help='分割头类型，auto=从权重嗅探（默认）')
    ap.add_argument('--out', default=None, help=f'结果 json 落盘目录（默认 {DEFAULT_OUT}）')
    args, unknown = ap.parse_known_args()

    utils.apply_cli_overrides(unknown)
    utils.seed_everything(config.SEG_SEED)

    decoder = explicit_decoder(unknown, args)
    if decoder:
        config.SEG_DECODER = decoder

    print(f'设备 {config.DEVICE}（cuda:0 对应 dev_id={config.DEVICE_ID}）  '
          f'HTS_IMG_SIZE={config.HTS_IMG_SIZE}  SEG_DECODER={config.SEG_DECODER}',
          flush=True)
    out = evaluate(args.ckpt, split=args.split, tta=args.tta, thr=args.thr,
                   tune=args.tune, decoder=decoder)

    out_dir = utils.ensure_dir(args.out or DEFAULT_OUT)
    tag = 'single' if len(args.ckpt) == 1 else f'ens{len(args.ckpt)}'
    if args.tta:
        tag += '_tta'
    path = os.path.join(out_dir, f'eval_{args.split}_{tag}.json')
    utils.write_json(path, out)
    print(f'结果已写入 {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
