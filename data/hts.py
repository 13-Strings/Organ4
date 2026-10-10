"""
HTS-Seg 数据集（Stage 3 球体分割）。

目录结构:
    {root}/{tiled_images|cleaned_tiled_images}/images/{control,treatment}/*.bmp
    {root}/{tiled_images|cleaned_tiled_images}/labels/{control,treatment}/*.tif

真值: uint8 二值图，0=芯片区域(背景)，1=球体区域(正类)，与论文一致。

数据集划分严格复刻 baseline（HTS_Seg-main 的 notebook）:
    imagePaths = sorted(list_images(tiled_images/images))   # 递归列出全部文件
    train, test = train_test_split(imagePaths, test_size=0.2, random_state=42)
因此本模块的测试集与 baseline 完全一致，指标可直接对齐。
"""

import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

import utils

_IMG_EXT = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------


def list_images_recursive(root, refresh=False):
    """
    等价于 imutils.paths.list_images：递归列出目录下全部文件（含子目录），已排序。
    用 os.scandir + 磁盘缓存实现 —— pathlib.rglob 在这台机器的 NFS 上慢到不可用。
    """
    return utils.list_files(root, exts=None, refresh=refresh)


def baseline_split(image_dir, mask_dir, test_ratio=0.2, val_ratio=0.1, seed=42,
                   mode='random'):
    """
    返回 (train_idx, val_idx, test_idx, img_paths, mask_paths)。

    mode='random'    : 复刻 baseline —— 对有序列表用
                       train_test_split(test_size, random_state=seed)，再切 10% 验证。
                       依赖图像/掩码两个列表下标对齐（这里显式断言 stems 一致）。
    mode='condition' : 训练集只用 control/，**测试集只用 treatment/**（'condition_rev'
                       反过来）。用于跨条件（药物处理）泛化测试，见 config 的注释。
    """
    from sklearn.model_selection import train_test_split

    img_paths = list_images_recursive(image_dir)
    mask_paths = list_images_recursive(mask_dir)
    if len(img_paths) != len(mask_paths):
        raise RuntimeError(f'图像/掩码数量不一致: {len(img_paths)} vs {len(mask_paths)}')

    for ip, mp in zip(img_paths, mask_paths):
        if os.path.splitext(os.path.basename(ip))[0] != \
           os.path.splitext(os.path.basename(mp))[0]:
            raise RuntimeError(f'图像与掩码未对齐:\n  {ip}\n  {mp}')

    n = len(img_paths)
    if mode in ('condition', 'condition_rev'):
        train_tag = f'{os.sep}control{os.sep}' if mode == 'condition' \
            else f'{os.sep}treatment{os.sep}'
        test_tag = f'{os.sep}treatment{os.sep}' if mode == 'condition' \
            else f'{os.sep}control{os.sep}'
        pool = np.array([i for i, p in enumerate(img_paths) if train_tag in p])
        te = np.array([i for i, p in enumerate(img_paths) if test_tag in p])
        if len(pool) == 0 or len(te) == 0:
            raise RuntimeError(
                f'按条件划分失败: {mode} 下训练池 {len(pool)} 张、测试集 {len(te)} 张。'
                f'请确认 {image_dir} 下有 control/treatment 子目录。')
        tr, va = train_test_split(pool, test_size=val_ratio, random_state=seed)
        return np.sort(tr), np.sort(va), np.sort(te), img_paths, mask_paths

    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_ratio, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_ratio, random_state=seed)
    return tr, va, te, img_paths, mask_paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HTSSegDataset(Dataset):
    """
    __getitem__ 返回 dict:
        image : (3,S,S) float32 [0,1]
        mask  : (1,S,S) float32 {0,1}   —— 1 表示球体区域（正类）
        label : long，0=control, 1=treatment（取自所在子目录名）
        index : 该样本在 split 内的行号
    """

    def __init__(self, img_paths, mask_paths, split_idx, img_size=224, train=False,
                 with_dist=False):
        self.img_paths = [img_paths[i] for i in split_idx]
        self.mask_paths = [mask_paths[i] for i in split_idx]
        self.img_size = img_size
        self.train = train
        self.with_dist = with_dist
        self._dist_cache_dir = None
        if with_dist:
            import config
            self._dist_cache_dir = os.path.join(
                config.OUTPUT_ROOT, '_cache', 'distmaps')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ipath = self.img_paths[idx]
        # 图像必须与掩码用同一套几何：整张 tile 直接缩到 img_size。
        # 不能走 load_image_tensor 的 eval 分支（短边缩到 256 再 center_crop）——
        # 那样图像只看到中心 (img_size*224/256)² 的区域，而掩码覆盖整张 tile，
        # 两者线性尺度差 256/224（实测前景占比 0.110 vs 0.145，比值 0.762），
        # 监督目标与输入对不齐。baseline 对 image/mask 用的是同一个 Resize((224,224))。
        image = TF.to_tensor(TF.resize(
            Image.open(ipath).convert('RGB'), [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.BILINEAR, antialias=True))

        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)
        mask = (np.asarray(mask) > 0).astype(np.float32)   # 严格二值化
        mask = torch.from_numpy(mask).unsqueeze(0)         # (1,S,S)

        dist = self._load_dist(self.mask_paths[idx]) if self.with_dist else None

        if self.train:
            if self.with_dist:
                image, mask, dist = _joint_augment3(image, mask, dist)
            else:
                image, mask = _joint_augment(image, mask)

        label = 1 if f'{os.sep}treatment{os.sep}' in ipath else 0
        out = {'image': image, 'mask': mask, 'label': label, 'index': idx}
        if self.with_dist:
            out['dist'] = dist
        return out

    def _load_dist(self, mask_path):
        """加载(或计算并落盘)掩码的欧氏距离图,返回 (1,S,S) float32。"""
        import boundary
        stem = os.path.splitext(os.path.basename(mask_path))[0]
        f = None
        if self._dist_cache_dir:
            os.makedirs(self._dist_cache_dir, exist_ok=True)
            f = os.path.join(self._dist_cache_dir, f'{stem}.npy')
            if os.path.exists(f):
                d = np.load(f)
                return torch.from_numpy(d).unsqueeze(0).float()
        m = np.asarray(Image.open(mask_path).convert('L')) > 0
        if m.shape[0] != self.img_size or m.shape[1] != self.img_size:
            m = np.asarray(TF.resize(Image.fromarray(m.astype(np.uint8)),
                                     [self.img_size, self.img_size],
                                     interpolation=TF.InterpolationMode.NEAREST)) > 0
        d = boundary.gt_dist_map(m).astype(np.float32)
        if f is not None:
            with open(f + '.tmp', 'wb') as fh:
                np.save(fh, d)
            os.replace(f + '.tmp', f)
        return torch.from_numpy(d).unsqueeze(0)

    @property
    def labels(self):
        return [1 if f'{os.sep}treatment{os.sep}' in p else 0 for p in self.img_paths]

    @property
    def stems(self):
        return [os.path.splitext(os.path.basename(p))[0] for p in self.img_paths]


def _joint_augment(image, mask):
    """
    图像与掩码同步的几何增强：水平翻转 + 垂直翻转 + 90° 整数倍旋转。
    全部用 torch RNG（已被 worker_init_fn 固定），因此可复现。
    """
    if torch.rand(1).item() < 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if torch.rand(1).item() < 0.5:
        image, mask = TF.vflip(image), TF.vflip(mask)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        image = torch.rot90(image, k, dims=(1, 2))
        mask = torch.rot90(mask, k, dims=(1, 2))
    return image, mask


def _joint_augment3(image, mask, dist):
    """
    三张图共享同一组几何变换（图像/掩码/距离图严格同步）。
    距离图是距离场,flip/rot90 等距变换下与掩码同步即可,无需重算。
    """
    if torch.rand(1).item() < 0.5:
        image, mask, dist = TF.hflip(image), TF.hflip(mask), TF.hflip(dist)
    if torch.rand(1).item() < 0.5:
        image, mask, dist = TF.vflip(image), TF.vflip(mask), TF.vflip(dist)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        image = torch.rot90(image, k, dims=(1, 2))
        mask = torch.rot90(mask, k, dims=(1, 2))
        dist = torch.rot90(dist, k, dims=(1, 2))
    return image, mask, dist


# ---------------------------------------------------------------------------
# 构造入口
# ---------------------------------------------------------------------------


def build_hts_datasets(img_size=None):
    """返回 (train_ds, val_ds, test_ds, meta)。meta 里带完整划分信息，便于落盘复现。"""
    import config

    img_size = img_size or config.HTS_IMG_SIZE
    base = os.path.join(config.HTS_ROOT, config.HTS_SUBSET)
    image_dir = os.path.join(base, 'images')
    mask_dir = os.path.join(base, 'labels')

    tr, va, te, img_paths, mask_paths = baseline_split(
        image_dir, mask_dir,
        test_ratio=config.TEST_RATIO, val_ratio=config.VAL_RATIO,
        seed=config.SPLIT_SEED, mode=getattr(config, 'SEG_SPLIT_MODE', 'random'))

    # 数据效率实验:只用一部分训练集(验证/测试集保持不变,便于跨比例对比)
    frac = float(getattr(config, 'SEG_TRAIN_FRACTION', 1.0))
    n_full = len(tr)
    if frac < 1.0:
        rs = np.random.RandomState(config.SPLIT_SEED)
        perm = rs.permutation(n_full)
        n_keep = max(1, int(round(n_full * frac)))
        tr = np.sort(np.asarray(tr)[perm[:n_keep]])
    else:
        n_keep = n_full

    with_dist = 'bnd' in config.SEG_LOSS
    train_ds = HTSSegDataset(img_paths, mask_paths, tr, img_size, train=True,
                             with_dist=with_dist)
    val_ds = HTSSegDataset(img_paths, mask_paths, va, img_size, train=False,
                           with_dist=with_dist)
    test_ds = HTSSegDataset(img_paths, mask_paths, te, img_size, train=False,
                            with_dist=with_dist)

    meta = {
        'subset': config.HTS_SUBSET,
        'image_dir': image_dir,
        'img_size': img_size,
        'split_mode': getattr(config, 'SEG_SPLIT_MODE', 'random'),
        'n_total': len(img_paths),
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'train_fraction': frac, 'n_train_full': n_full,
        'split_seed': config.SPLIT_SEED,
        'test_ratio': config.TEST_RATIO,
        'test_stems': test_ds.stems,
        'test_labels': test_ds.labels,
    }
    return train_ds, val_ds, test_ds, meta
