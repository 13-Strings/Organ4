"""
Stage 1 的「纯器官芯片」预训练语料:OOC 芯片明场图 + HTS-Seg 芯片球体 tile。

为什么要有这个模块
------------------
原 Stage1 用的是 RxRx1(2D 细胞培养的 6 通道荧光图),它**不是器官芯片数据**。
本赛道要求数据与器官芯片相关,所以需要一个"完全不碰非芯片数据"的对照臂:

    OOC       3,072 张芯片明场(6 细胞系,good/bad 质量标签 —— 这里只用图像)
    HTS-Seg   2,180 张芯片球体 tile(control/treatment —— 只用图像,不用掩码)
    ---------------------------------------------------------------
    合计      5,252 张,全部来自器官芯片

两臂(PxRx1 预训练 vs 纯 OoC 预训练)在 Stage2/Stage3 用完全相同的配置与数据,
差异只来自预训练语料,因此能直接回答:大规模非芯片数据到底还需不需要。

接口与 data/rxrx1.py 的 RxRx1Dataset 一致({'image', 'index'}),train_mae.py
只换 builder,其余训练逻辑一行不动。
"""

import os

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

import utils

_IMG_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _collect_paths():
    """收集两个器官芯片数据集的全部图像路径,排序保证可复现。"""
    import config
    paths = []

    # OOC:OOC_image_dataset/{train,val,test}/{good,bad}/cell_type/days/*.png
    for sp in ('train', 'val', 'test'):
        for lab in ('good', 'bad'):
            sub = os.path.join(config.OOC_ROOT, sp, lab)
            if os.path.isdir(sub):
                paths.extend(utils.list_files(sub, exts=_IMG_EXT))

    # HTS-Seg:cleaned_tiled_images/images/{control,treatment}/*.bmp
    hts_img = os.path.join(config.HTS_ROOT, config.HTS_SUBSET, 'images')
    if os.path.isdir(hts_img):
        paths.extend(utils.list_files(hts_img, exts=_IMG_EXT))

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f'器官芯片预训练语料为空。检查 OOC_ROOT={config.OOC_ROOT} 与 '
            f'HTS_ROOT/{{HTS_SUBSET}}/images={hts_img}')
    return paths


class OoCPretrainDataset(Dataset):
    """
    __getitem__ 返回 {'image': (3,S,S) float32 [0,1], 'index': int}。

    增强与 RxRx1Dataset 保持一致(训练 RandomResizedCrop、评估短边缩放+中心裁剪),
    这样两臂的差异只剩"预训练数据来源"这一个变量。
    """

    def __init__(self, paths, img_size=512, train=True):
        self.paths = list(paths)
        self.img_size = img_size
        self.train = train
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # OOC 的原始图尺寸不一(实测有 1536x2048、1542x2056 等),HTS tile 是 512;
        # 两条分支都保证输出恰好 (S,S)。
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.aug(img) if self.train else self.resize(img)
        return {'image': TF.to_tensor(img), 'index': idx}


def build_ooc_pretrain(phase='train', img_size=None, **overrides):
    """按 config 构造纯器官芯片预训练数据集。phase ∈ {'train','eval'}。"""
    import config
    img_size = img_size or config.MAE_INPUT_SIZE
    ds = OoCPretrainDataset(_collect_paths(), img_size=img_size,
                            train=(phase == 'train'))
    if overrides.get('verbose', True):
        print(f'[OoCPretrain] {len(ds)} 张器官芯片图像 '
              f'(OOC + HTS-Seg), img_size={img_size}, train={phase == "train"}',
              flush=True)
    return ds
