"""
OOC 明场数据集（Stage 2 的源域 / 目标域，3072 张）。

目录结构: {root}/{split}/{good|bad}/*.png，split ∈ {train, val, test}
原始标签 'Decision 1/2 (good/bad)' 在目录名里: good=0, bad=1。

域适应是无监督的，主用途是"读图 -> 算重建损失 / 域判别损失"；
good/bad 标签只在 DOM_AUX_CLS=True 时作为辅助分类任务使用。
"""

import os

import numpy as np
from torch.utils.data import Dataset

import utils
from .rxrx1 import load_image_tensor

LABEL_NAMES = ['good', 'bad']          # 目录名 -> 标签 0/1
_IMG_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def list_split_images(root, split, refresh=False):
    """
    返回 (paths, labels)。labels: good=0, bad=1。

    真实目录结构是 {split}/{good,bad}/cell_type_XXX/{0-1,2-3,4,4+}_days/*.png，
    所以必须递归；用 utils.list_files 的 scandir + 磁盘缓存版本，
    pathlib.rglob 在这台机器的 NFS 上慢到不可用。
    """
    paths, labels = [], []
    for lab, name in enumerate(LABEL_NAMES):
        sub = os.path.join(root, split, name)
        if not os.path.isdir(sub):
            continue
        files = utils.list_files(sub, exts=_IMG_EXT, refresh=refresh)
        paths.extend(files)
        labels.extend([lab] * len(files))
    return paths, labels


class OOCDataset(Dataset):
    """
    __getitem__ 返回 dict:
        image     : (3,S,S) float32 [0,1]
        aux_label : long, good=0 / bad=1（无标签时为 -1）
        index     : 全局行号
    """

    def __init__(self, paths, labels, img_size=224, train=True, strong_aug=False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.img_size = img_size
        self.train = train
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_tensor(self.paths[idx], self.img_size,
                                train=self.train, strong_aug=self.strong_aug)
        lab = self.labels[idx] if self.labels[idx] is not None else -1
        return {'image': img, 'aux_label': int(lab), 'index': idx}


def build_ooc(splits=('train',), img_size=224, train=True):
    """按 config 构造 OOC 数据集（多 split 合并成一个）。"""
    import config
    paths, labels = [], []
    for sp in splits:
        p, l = list_split_images(config.OOC_ROOT, sp)
        paths.extend(p)
        labels.extend(l)
    if not paths:
        raise FileNotFoundError(
            f'OOC 目录下没找到图像: {config.OOC_ROOT}/{{{",".join(splits)}}}/*/')
    return OOCDataset(paths, labels, img_size=img_size, train=train)
