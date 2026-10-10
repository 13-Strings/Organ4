"""
把 RxRx1 预解码成内存映射缓存，解决 NFS 单文件延迟过高的问题。

为什么需要这个：
    实测本机 NFS 每个 PNG 的读取延迟约 0.5 秒，RxRx1 每个 site 要读 3 个通道：
      workers=0 -> 0.7 img/s        workers=2 -> 2.7 img/s
    按这个速度跑全量预训练，GPU 利用率会低到 1% 以下。
    把像素预先解码成一个大文件后，顺序读取可以跑满带宽。

做法：
    对每个 site 读 config.RXRX1_CHANNELS 指定的 3 个通道 -> resize 到 CACHE_SIZE
    -> 顺序写进一个 (N, 3, S, S) 的 uint8 内存映射文件。
    全量 81224 site、S=256 时约 15.9GB。

断点续训：
    进度记在 meta json 的 last_done 字段里，中断后重跑会从下一个 site 继续，
    已经写好的部分不重算。

用法:
    /data/user/miniconda3/envs/mumo/bin/python prepare_rxrx1_cache.py
    ... --WORKERS 24 --CACHE_SIZE 256 --LIMIT 1000     # 先小规模试
    ... --full                                          # 用全部 125510 个 site
"""

import config  # noqa: F401  —— 必须最先 import

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from PIL import Image

import utils

# 全局，供子进程通过 fork 继承
_ROOT = None
_CHANNELS = None
_SIZE = None
_DF = None


def _site_path(row, channel):
    return os.path.join(_ROOT, row['experiment'], f"Plate{int(row['plate'])}",
                        f"{row['well']}_s{int(row['site'])}_w{int(channel)}.png")


def _load_site(idx):
    """
    读一个 site 的 3 个通道并 resize，返回 (idx, uint8 数组 (3,S,S))。

    用 try/except 而不是 os.path.exists + open：在 NFS 上每次 exists 都是一次
    额外往返，实测白吃 37% 的开销；直接 open 失败再兜底更划算。
    """
    row = _DF.iloc[idx]
    planes = []
    for ch in _CHANNELS:
        p = _site_path(row, ch + 1)          # 文件名里 w1..w6 是 1-based
        try:
            im = Image.open(p).convert('L').resize((_SIZE, _SIZE),
                                                   Image.BILINEAR)
            planes.append(np.asarray(im))
        except (OSError, ValueError):
            planes.append(np.zeros((_SIZE, _SIZE), dtype=np.uint8))
    return idx, np.stack(planes, axis=0)     # (3, S, S)


def _init_worker(root, channels, size):
    """worker 初始化。DataFrame 通过 fork 继承，不用序列化传参。"""
    global _ROOT, _CHANNELS, _SIZE
    _ROOT, _CHANNELS, _SIZE = root, channels, size


def main():
    utils.apply_cli_overrides()
    utils.seed_everything(config.SEED)

    import pandas as pd

    size = config.RXRX1_CACHE_SIZE
    cache_dir = utils.ensure_dir(config.RXRX1_CACHE_DIR)
    tag = f"ch{'-'.join(map(str, config.RXRX1_CHANNELS))}_s{size}"
    bin_path = os.path.join(cache_dir, f'rxrx1_{tag}.u8')
    meta_path = os.path.join(cache_dir, f'rxrx1_{tag}.json')

    print(f'=== RxRx1 预解码缓存 ===\n  源: {config.RXRX1_ROOT}')
    print(f'  目标: {bin_path}\n  通道: {config.RXRX1_CHANNELS}  '
          f'尺寸: {size}x{size}  workers: {config.CACHE_WORKERS}')

    df = pd.read_csv(config.RXRX1_METADATA)
    if config.RXRX1_TRAIN_SPLIT_ONLY:
        df = df[df['dataset'] == 'train']
    df = df.reset_index(drop=True)
    n_total = len(df)

    limit = config.CACHE_LIMIT
    if limit and limit > 0:
        n_total = min(n_total, limit)
        print(f'  [限制] 只处理前 {n_total} 个 site')

    expected_bytes = n_total * 3 * size * size

    meta = utils.read_json(meta_path, default={}) or {}
    cur_split = 'train' if config.RXRX1_TRAIN_SPLIT_ONLY else 'all'
    same = (meta.get('channels') == list(config.RXRX1_CHANNELS)
            and meta.get('size') == size and meta.get('n') == n_total
            and meta.get('root') == config.RXRX1_ROOT
            and meta.get('dataset_split', 'train') == cur_split)
    if not same:
        if os.path.exists(bin_path):
            print('  参数与已有缓存不一致，重建')
            os.remove(bin_path)
        meta = {'channels': list(config.RXRX1_CHANNELS), 'size': size,
                'n': n_total, 'root': config.RXRX1_ROOT, 'last_done': -1,
                # 记录本缓存是按哪个 dataset split 建的。读取端会校验，
                # 防止拿 train 建的下标去索引 test 的样本（会静默读到训练数据）
                'dataset_split': ('train' if config.RXRX1_TRAIN_SPLIT_ONLY
                                  else 'all')}

    done = int(meta.get('last_done', -1))
    if done >= n_total - 1:
        print(f'  缓存已完整（{n_total} 个 site），无需重跑')
        _write_meta(meta_path, meta, bin_path)
        return 0

    # 预分配文件（稀疏写，不实际占用全部空间直到写入）
    if not os.path.exists(bin_path):
        with open(bin_path, 'wb') as f:
            f.truncate(expected_bytes)

    arr = np.memmap(bin_path, dtype=np.uint8, mode='r+',
                    shape=(n_total, 3, size, size))

    start = done + 1
    print(f'  已有进度: {start}/{n_total}，从第 {start} 个继续')

    global _ROOT, _CHANNELS, _SIZE, _DF
    _ROOT = config.RXRX1_ROOT
    _CHANNELS = list(config.RXRX1_CHANNELS)
    _SIZE = size
    _DF = df.iloc[:n_total].reset_index(drop=True)

    t0 = time.time()
    n_done = 0
    last_report = t0
    interrupted = False
    try:
        if config.CACHE_WORKERS <= 1:
            for idx in range(start, n_total):
                i, a = _load_site(idx)
                arr[i] = a
                _flush(arr, meta, meta_path, bin_path, i)
                n_done += 1
                if n_done % 500 == 0:
                    _report(n_done, n_total, start, t0)
        else:
            ctx = mp.get_context('fork')
            with ctx.Pool(config.CACHE_WORKERS, initializer=_init_worker,
                          initargs=(_ROOT, _CHANNELS, _SIZE)) as pool:
                # imap 保序：可以安全地按到达顺序推进 last_done
                for i, a in pool.imap(_load_site, range(start, n_total),
                                      chunksize=32):
                    arr[i] = a
                    _flush(arr, meta, meta_path, bin_path, i)
                    n_done += 1
                    now = time.time()
                    if now - last_report > 15:
                        _report(n_done, n_total, start, t0)
                        last_report = now
    except KeyboardInterrupt:
        interrupted = True
        print('\n[中断] 保存进度后退出，重跑本脚本即可续做', flush=True)

    arr.flush()
    meta['last_done'] = int(meta.get('last_done', -1))
    _write_meta(meta_path, meta, bin_path)

    elapsed = time.time() - t0
    done_now = meta['last_done'] + 1
    print(f'\n完成 {done_now}/{n_total} 个 site，本次新增 {n_done}，耗时 {elapsed/60:.1f} 分钟')
    print(f'缓存文件: {bin_path} ({os.path.getsize(bin_path)/2**30:.2f} GB)')
    return 0 if not interrupted else 130


def _flush(arr, meta, meta_path, bin_path, i):
    """按顺序推进进度；每 2000 个落一次盘，避免频繁写 json。"""
    meta['last_done'] = i
    if (i + 1) % 2000 == 0:
        arr.flush()
        _write_meta(meta_path, meta, bin_path)


def _write_meta(meta_path, meta, bin_path):
    meta = dict(meta)
    meta['bin'] = bin_path
    meta['bytes'] = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
    tmp = meta_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)


def _report(n_done, n_total, start, t0):
    el = time.time() - t0
    rate = n_done / max(el, 1e-9)
    left = (n_total - start - n_done) / max(rate, 1e-9)
    print(f'  {start + n_done}/{n_total}  {rate:.1f} site/s  '
          f'剩余约 {left/60:.1f} 分钟', flush=True)


if __name__ == '__main__':
    sys.exit(main())
