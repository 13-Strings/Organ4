"""
汇总报告 —— 把三个阶段的结果收拢成一份 Markdown + 若干曲线图。

三个阶段的原始日志本来就分别写在各自目录里（train_log.jsonl / metrics.json），
但那需要手工解析 jsonl 才能看出趋势。本脚本把它们汇总成：

    {OUTPUT_ROOT}/summary/report.md         一张表看完全部结果
    {OUTPUT_ROOT}/summary/curves/*.png      各阶段的 loss / 指标曲线

用法:
    $PY report.py                     # 汇总正式的三个阶段
    $PY report.py --RUN_TAG _smoke    # 汇总冒烟测试的结果
"""
import setproctitle
setproctitle.setproctitle("python3") 
import config  # noqa: F401  —— 必须最先 import

import argparse
import json
import os
import sys

import metrics as M
import utils

STAGES = [('stage1_mae', 'Stage 1  MAE 自监督预训练'),
          ('stage2_domain', 'Stage 2  域适应'),
          ('stage3_hts', 'Stage 3  HTS-Seg 任务微调')]


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def collect(out_root):
    """读回三个阶段的所有日志。缺失的阶段记为 None。"""
    # 目录路径由参数逐层拼成（s3 层还挂着 SEG_* 后缀），所以向 config 要，
    # 不要在这里自己拼 —— 否则 Stage 3 会因为名字对不上而被静默漏掉
    dirs = dict(zip(('stage1_mae', 'stage2_domain', 'stage3_hts'),
                    config.stage_dir_names()))
    data = {}
    for name, title in STAGES:
        d = os.path.join(out_root, dirs[name])
        if not os.path.isdir(d):
            continue
        data[name] = {
            'title': title,
            'dir': d,
            'log': read_jsonl(os.path.join(d, 'train_log.jsonl')),
            'metrics': utils.read_json(os.path.join(d, 'metrics.json')),
            'eval': utils.read_json(os.path.join(d, 'eval', 'test_metrics.json')),
            'table2': _read_text(os.path.join(d, 'eval', 'table2.md')),
        }
    return data


def _read_text(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# 曲线
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 曲线标题/图例里带中文，默认的 DejaVu Sans 没有 CJK 字形，会渲染成空框。
    # 列表里没有可用字体时 matplotlib 会自动退到下一个，不会报错。
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return plt


def plot_curves(data, fig_dir):
    plt = _plt()
    made = []

    # ---- Stage 1: 训练 loss + 每 epoch 验证 loss ----
    s1 = data.get('stage1_mae')
    if s1 and s1['log']:
        train = [(r['step'], r['loss']) for r in s1['log']
                 if r.get('msg') == 'train' and 'loss' in r and 'step' in r]
        val = [(r['epoch'], r['val_loss']) for r in s1['log']
               if r.get('msg') == 'epoch 结束' and 'val_loss' in r]
        fig, ax = plt.subplots(figsize=(9, 4))
        if train:
            ax.plot(*zip(*train), lw=1, label='train loss')
        if val:
            ax.plot(*zip(*val), 'o-', label='val loss (per epoch)')
        ax.set_xlabel('step / epoch')
        ax.set_ylabel('MAE recon loss')
        ax.set_title('Stage 1  RxRx1 MAE 预训练')
        ax.legend()
        ax.grid(alpha=.3)
        p = os.path.join(fig_dir, 'stage1_loss.png')
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)

    # ---- Stage 2: 各损失分量 + 留出集指标 ----
    s2 = data.get('stage2_domain')
    if s2 and s2['log']:
        keys = ['recon_src', 'recon_tgt', 'domain']
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for k in keys:
            pts = [(r['step'], r[k]) for r in s2['log']
                   if r.get('msg') == 'train' and k in r and 'step' in r]
            if pts:
                axes[0].plot(*zip(*pts), lw=1, label=k)
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title('Stage 2  训练损失分量')
        axes[0].legend()
        axes[0].grid(alpha=.3)

        vh = (s2['metrics'] or {}).get('val_history', [])
        if vh:
            ep = [r['epoch'] for r in vh]
            for k in ('recon_tgt', 'recon_src'):
                if k in vh[0]:
                    axes[1].plot(ep, [r[k] for r in vh], 'o-', label=k)
            ax2 = axes[1].twinx()
            if 'dom_acc' in vh[0]:
                ax2.plot(ep, [r['dom_acc'] for r in vh], 's--', color='crimson',
                         label='domain acc (理想=0.5)')
                ax2.set_ylabel('domain acc', color='crimson')
                ax2.axhline(0.5, color='crimson', ls=':', alpha=.5)
            axes[1].set_xlabel('epoch')
            axes[1].set_ylabel('recon loss')
            axes[1].set_title('Stage 2  留出集')
            h1, l1 = axes[1].get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            axes[1].legend(h1 + h2, l1 + l2, fontsize=8)
            axes[1].grid(alpha=.3)
        p = os.path.join(fig_dir, 'stage2_domain.png')
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)

    # ---- Stage 3: 训练 loss + 每 epoch 验证指标 ----
    s3 = data.get('stage3_hts')
    if s3 and s3['log']:
        train = [(r['step'], r['loss']) for r in s3['log']
                 if r.get('msg') == 'train' and 'loss' in r and 'step' in r]
        ep = [r for r in s3['log'] if r.get('msg') == 'epoch 结束']
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        if train:
            axes[0].plot(*zip(*train), lw=1)
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('loss')
        axes[0].set_title('Stage 3  训练损失 (BCE + Dice)')
        axes[0].grid(alpha=.3)

        if ep:
            e = [r['epoch'] for r in ep]
            for k, lab in (('val_fg_dice', 'val fg_dice'),
                           ('val_jaccard', 'val Jaccard (baseline 口径)'),
                           ('val_accuracy', 'val accuracy')):
                if k in ep[0]:
                    axes[1].plot(e, [r[k] for r in ep], 'o-', label=lab)
            axes[1].axhline(0.9117, color='gray', ls='--',
                            label='DeepLabv3 Jaccard 0.9117 (论文最优)')
            axes[1].set_xlabel('epoch')
            axes[1].set_title('Stage 3  验证集区域指标（与 baseline 同口径）')
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=.3)

            # 边界指标:区域指标看不出轮廓误差,单独画一张
            if 'val_hd95' in ep[0]:
                axes[2].plot(e, [r['val_hd95'] for r in ep], 'o-',
                             color='crimson', label='HD95 (px, 越小越好)')
                axes[2].plot(e, [r['val_assd'] for r in ep], 's-',
                             color='darkorange', label='ASSD (px, 越小越好)')
                ax2 = axes[2].twinx()
                ax2.plot(e, [r['val_biou2'] for r in ep], '^--',
                         color='teal', label='boundary IoU@2px (越大越好)')
                ax2.set_ylabel('boundary IoU', color='teal')
                axes[2].set_xlabel('epoch')
                axes[2].set_ylabel('distance (px)')
                axes[2].set_title('Stage 3  验证集边界指标')
                h1, l1 = axes[2].get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                axes[2].legend(h1 + h2, l1 + l2, fontsize=8)
                axes[2].grid(alpha=.3)
        p = os.path.join(fig_dir, 'stage3_seg.png')
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)

    return made


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def build_report(data, out_root, tag, figs):
    L = []
    L.append('# OSMOSIS 三阶段训练报告\n')
    L.append(f'- 输出根目录: `{out_root}`')
    L.append(f'- RUN_TAG: `{tag or "(无)"}`')
    L.append(f'- 随机种子: {config.SEED}   '
             f'测试集划分种子: {config.SPLIT_SEED}')
    L.append(f'- 数据集: RxRx1 / OOC / HTS-Seg(`{config.HTS_SUBSET}`)')
    L.append('')

    # ---- 一页总览 ----
    L.append('## 1. 一页总览\n')
    L.append('| 阶段 | 轮数 | 全局 step | 关键结果 | 峰值显存 |')
    L.append('|---|---|---|---|---|')
    for name, title in STAGES:
        d = data.get(name)
        if not d:
            L.append(f'| {title} | — | — | 未运行 | — |')
            continue
        m = d['metrics'] or {}
        key = ''
        if name == 'stage1_mae':
            v = m.get('best_val_loss')
            key = f'best val loss = {v:.5f}' if isinstance(v, float) else '—'
        elif name == 'stage2_domain':
            vh = m.get('val_history') or []
            if vh:
                key = (f"dom_acc={vh[-1].get('dom_acc', float('nan')):.4f}, "
                       f"recon_tgt={vh[-1].get('recon_tgt', float('nan')):.4f}")
            key = f"{config.DOM_SOURCE} -> {config.DOM_TARGET}; " + (key or '—')
        else:
            # test_metrics.json 的结构是 {split, metrics:{per_image,pooled}, ...}
            ev = ((d['eval'] or {}).get('metrics') or {}).get('per_image', {})
            if ev:
                key = (f"Dice={ev['dice']['mean']:.4f}, "
                       f"Jaccard={ev['jaccard']['mean']:.4f}, "
                       f"Acc={ev['accuracy']['mean']:.4f}")
                bg = M.BASELINE_TABLE2['DeepLabv3']['jaccard']
                key += f"（论文最优 {bg}）"
            bnd = (d['eval'] or {}).get('boundary') or {}
            if bnd:
                key += (f"; ASSD={bnd['assd']['mean']:.3f}px, "
                        f"HD95={bnd['hd95']['mean']:.3f}px, "
                        f"bIoU@2px={bnd['biou2']['mean']:.4f}")
        L.append(f"| {title} | {m.get('epochs_done', '—')} | "
                 f"{m.get('global_step', '—')} | {key} | "
                 f"{m.get('peak_gpu_gb', '—')} GB |")
    L.append('')

    # ---- 各阶段细节 ----
    L.append('## 2. 分阶段细节\n')
    for name, title in STAGES:
        d = data.get(name)
        L.append(f'### {title}\n')
        if not d:
            L.append('未运行。\n')
            continue
        L.append(f"日志目录: `{d['dir']}`\n")
        m = d['metrics']
        if m:
            drop = {'val_history'}
            kv = {k: v for k, v in m.items() if k not in drop}
            L.append('```json')
            L.append(json.dumps(kv, ensure_ascii=False, indent=2, default=str))
            L.append('```\n')
        if name == 'stage2_domain' and (m or {}).get('val_history'):
            L.append('留出集逐 epoch：\n')
            L.append('| epoch | recon_src | recon_tgt | domain | dom_acc | aux_acc |')
            L.append('|---|---|---|---|---|---|')
            for r in m['val_history']:
                L.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                    r.get('epoch', '—'), r.get('recon_src', float('nan')),
                    r.get('recon_tgt', float('nan')), r.get('domain', float('nan')),
                    r.get('dom_acc', float('nan')), r.get('aux_acc', float('nan'))))
            L.append('')
        if d['table2']:
            L.append(d['table2'])
            L.append('')

    # ---- 曲线 ----
    if figs:
        L.append('## 3. 曲线\n')
        for p in figs:
            rel = os.path.relpath(p, os.path.dirname(p))
            L.append(f'### {rel}\n')
            L.append(f'![{rel}](curves/{rel})\n')

    L.append('## 4. 文件索引\n')
    L.append('| 文件 | 内容 |')
    L.append('|---|---|')
    for name, title in STAGES:
        d = data.get(name)
        if not d:
            continue
        base = os.path.basename(d['dir'])
        L.append(f'| `{base}/train_log.jsonl` | 每步/每轮原始日志（loss、lr、显存、指标） |')
        L.append(f'| `{base}/metrics.json` | 训练摘要 |')
        L.append(f'| `{base}/config_snapshot.py` | 本次实验的完整参数快照 |')
    L.append('| `summary/report.md` | 本文件 |')
    L.append('| `summary/curves/*.png` | 各阶段曲线 |')
    L.append('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--RUN_TAG', dest='run_tag', default=None,
                    help='要汇总的实验标签，默认用 config.RUN_TAG')
    ap.add_argument('--out', default=None, help='报告输出目录')
    args, unknown = ap.parse_known_args()
    utils.apply_cli_overrides(unknown)

    tag = config.RUN_TAG if args.run_tag is None else args.run_tag
    out_root = config.output_root(tag)
    data = collect(out_root)

    if not data:
        raise SystemExit(
            f'{out_root} 下没有找到任何阶段目录，先跑训练再来汇总。\n'
            f'期望的目录名: {", ".join(config.stage_dir_names())}\n'
            f'提示: 目录名由参数拼成；若训练时用命令行覆盖过参数，'
            f'汇总时要用同一组参数（或 --RUN_TAG）。')

    summary_dir = utils.ensure_dir(args.out or os.path.join(out_root, 'summary'))
    fig_dir = utils.ensure_dir(os.path.join(summary_dir, 'curves'))

    figs = plot_curves(data, fig_dir)
    md = build_report(data, out_root, tag, figs)
    with open(os.path.join(summary_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(md)

    print(f'已汇总 {len(data)} 个阶段 -> {os.path.join(summary_dir, "report.md")}')
    for p in figs:
        print(f'  曲线: {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
