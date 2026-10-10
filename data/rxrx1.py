"""
RxRx1 数据集（Stage 1 预训练 / Stage 2 可选源域或目标域）。

原始数据: 125510 个 site，每个 site 是 6 个通道的 512x512 uint8 PNG，
文件名形如 {experiment}/Plate{plate}/{well}_s{site}_w{channel}.png。

网络输入固定为 3 通道，因此按 config.RXRX1_CHANNELS 选 3 个通道堆叠。

两条读取路径：
  1. 预解码缓存（默认）—— prepare_rxrx1_cache.py 生成的 (N,3,S,S) uint8 内存映射。
  2. 直接读 PNG —— 缓存不存在或参数不匹配时自动回落，会打印提示。
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class RxRx1Dataset(Dataset):
    """
    __getitem__ 返回 dict:
        image : (3, S, S) float32, 取值 [0,1]
        index : 该样本在索引表里的行号（便于复现与调试）

    train=True 时做 MAE 标准的 RandomResizedCrop 增强；否则中心裁剪到 S。
    """

    def __init__(self, root, metadata_csv, channels=(0, 1, 4), img_size=224,
                 train=True, train_split_only=True, subset_ratio=1.0, seed=42,
                 use_cache=None, cache_dir=None, cache_size=None,
                 dataset_split=None):
        self.root = root
        self.channels = list(channels)
        self.img_size = img_size
        self.train = train
        self._png_cache = {}

        # ---- 索引表：先按 split 过滤，再取固定随机子集 ----
        df = pd.read_csv(metadata_csv)
        self.split_used = (dataset_split if dataset_split is not None
                           else ('train' if (train_split_only and
                                             'dataset' in df.columns) else 'all'))
        if dataset_split is not None and 'dataset' in df.columns:
            # 显式指定用哪个 split（做留出集评估时用 'test'，
            # 否则会在训练过的 site 上评估，指标虚高）
            df = df[df['dataset'] == dataset_split]
        elif train_split_only and 'dataset' in df.columns:
            df = df[df['dataset'] == 'train']
        df = df.reset_index(drop=True)

        n_all = len(df)
        if subset_ratio < 1.0:
            rs = np.random.RandomState(seed)
            keep = rs.choice(n_all, size=max(1, int(n_all * subset_ratio)),
                             replace=False)
            keep.sort()
        else:
            keep = np.arange(n_all)

        # keep[i] = 第 i 个样本在原始（未取子集）索引表里的行号，
        # 同时也是它在预解码缓存里的行号 —— 两边用的是同一个 order
        self.cache_index = keep
        self.df = df.iloc[keep].reset_index(drop=True)

        # ---- 读取后端 ----
        self.memmap = self._open_memmap(use_cache, cache_dir, cache_size)
        src_size = self.memmap.shape[-1] if self.memmap is not None else 512

        # ---- 增强 ----
        if train:
            self.aug = T.RandomResizedCrop(
                img_size, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3),
                interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        else:
            self.resize = T.Compose([
                T.Resize(int(img_size * 256 / 224),
                         interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(img_size),
            ])
        self.src_size = src_size

    # ------------------------------------------------------------------
    def _open_memmap(self, use_cache, cache_dir, cache_size):
        """
        打开预解码缓存。通道 / 尺寸 / 覆盖的 site 数任何一项对不上就回落到读 PNG，
        避免"缓存与当前配置不一致却照样用"的静默错误。
        """
        import config
        if use_cache is None:
            use_cache = config.RXRX1_CACHE
        if not use_cache:
            return None

        cache_dir = cache_dir or config.RXRX1_CACHE_DIR
        cache_size = cache_size or config.RXRX1_CACHE_SIZE
        tag = f"ch{'-'.join(map(str, self.channels))}_s{cache_size}"
        bin_path = os.path.join(cache_dir, f'rxrx1_{tag}.u8')
        meta_path = os.path.join(cache_dir, f'rxrx1_{tag}.json')

        if not (os.path.exists(bin_path) and os.path.exists(meta_path)):
            print(f'[RxRx1] 未找到预解码缓存 {bin_path}，回落为逐张读 PNG '
                  f'(NFS 上很慢，建议先跑 prepare_rxrx1_cache.py)', flush=True)
            return None

        meta = json.load(open(meta_path))
        if meta.get('channels') != self.channels or \
                meta.get('size') != cache_size:
            return None
        # 缓存的样本顺序是按某个 split 过滤后的顺序，拿 train 的下标去索引
        # test 的样本会静默读到训练数据 —— split 不一致就退回读 PNG
        if meta.get('dataset_split', 'train') != self.split_used:
            print(f"[RxRx1] 缓存是按 '{meta.get('dataset_split', 'train')}' split 建的，"
                  f"当前需要 '{self.split_used}'，回落为逐张读 PNG", flush=True)
            return None

        need = int(self.cache_index.max()) + 1
        if meta.get('n', 0) < need:
            print(f"[RxRx1] 缓存只覆盖 {meta.get('n')} 个 site，"
                  f"当前需要 {need} 个，回落为逐张读 PNG", flush=True)
            return None

        print(f'[RxRx1] 使用预解码缓存 {bin_path} '
              f'({meta["n"]} site, {cache_size}x{cache_size})', flush=True)
        return np.memmap(bin_path, dtype=np.uint8, mode='r',
                         shape=(meta['n'], 3, cache_size, cache_size))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def _site_path(self, row, channel):
        return os.path.join(self.root, row['experiment'],
                            f"Plate{int(row['plate'])}",
                            f"{row['well']}_s{int(row['site'])}_w{int(channel)}.png")

    def _load_png(self, idx):
        """逐张读 3 个通道并堆叠成 (S,S,3) uint8。缺文件用全 0 兜底，不中断训练。"""
        row = self.df.iloc[idx]
        planes = []
        for ch in self.channels:
            p = self._site_path(row, ch + 1)      # 文件名里 w1..w6 是 1-based
            if os.path.exists(p):
                planes.append(np.asarray(Image.open(p).convert('L')))
            else:
                planes.append(np.zeros((512, 512), dtype=np.uint8))
        return np.stack(planes, axis=-1)

    def __getitem__(self, idx):
        if self.memmap is not None:
            arr = np.asarray(
                self.memmap[self.cache_index[idx]]).transpose(1, 2, 0)
        else:
            arr = self._png_cache.get(idx)
            if arr is None:
                arr = self._load_png(idx)
                if len(self._png_cache) < 256:     # 小 LRU，避免重复解 PNG
                    self._png_cache[idx] = arr

        img = Image.fromarray(arr)
        img = self.aug(img) if self.train else self.resize(img)
        return {'image': TF.to_tensor(img), 'index': idx}   # (3,S,S) float32 [0,1]

    def close(self):
        self._png_cache.clear()


def build_rxrx1(phase='train', **overrides):
    """按 config 构造 RxRx1 数据集。phase ∈ {'train','eval'} 决定是否做随机增强。"""
    import config
    kw = dict(root=config.RXRX1_ROOT,
              metadata_csv=config.RXRX1_METADATA,
              channels=tuple(config.RXRX1_CHANNELS),
              img_size=config.MAE_INPUT_SIZE,
              train=(phase == 'train'),
              train_split_only=config.RXRX1_TRAIN_SPLIT_ONLY,
              subset_ratio=config.RXRX1_SUBSET_RATIO,
              seed=config.SEED,
              dataset_split=None)
    kw.update(overrides)
    return RxRx1Dataset(**kw)


# ---------------------------------------------------------------------------
# 通用：把任意图像统一成 3 通道 [0,1] 张量（OOC / HTS 共用）
# ---------------------------------------------------------------------------


def load_image_tensor(path, img_size, train=False, strong_aug=False):
    """
    读图 -> 3 通道 -> resize/crop -> (3,S,S) float32 [0,1]。

    注意：**所有分支都必须输出恰好 (img_size, img_size)**。
    OOC 数据集的原始图像尺寸并不统一（实测有 1536x2048、1542x2056 等），
    一旦漏掉 resize，DataLoader 的 collate 就会因为尺寸不一致直接报错。
    """
    img = Image.open(path).convert('RGB')
    if train:
        # 训练时统一用 RandomResizedCrop，既做增强又保证输出尺寸固定
        scale = (0.5, 1.0) if strong_aug else (0.8, 1.0)
        img = T.RandomResizedCrop(
            img_size, scale=scale, ratio=(3 / 4, 4 / 3),
            interpolation=T.InterpolationMode.BILINEAR, antialias=True)(img)
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
    else:
        # 短边缩放到 256 再中心裁剪，对任意长宽比都成立
        img = TF.resize(img, int(img_size * 256 / 224),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True)
        img = TF.center_crop(img, img_size)
    return TF.to_tensor(img)
