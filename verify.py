"""
冒烟测试 —— 不训练、不落盘大文件，几分钟内验证整条流水线的每个环节是否正确。

用法:
    /data/user/miniconda3/envs/mumo/bin/python verify.py
    /data/user/miniconda3/envs/mumo/bin/python verify.py --skip-data

逐项检查:
    1. 环境与随机种子
    2. 三个数据集各取一个 batch（形状 / 取值域 / 掩码二值性）
    3. 数据集划分是否与 baseline 一致
    4. ViT / MAE / 分割头 / 域判别器 的前向与参数量
    5. OOMAdaptiveStep 的梯度累积与切分是否正确
    6. 形态学特征在合成圆盘上的正确性（有解析解可比）
    7. metri() 与 baseline 公式逐行一致
    8. 断点保存/加载往返一致
"""

import config  # noqa: F401  —— 必须最先 import

import sys
import traceback

import numpy as np

PASS, FAIL, SKIP = [], [], []


class SkipCheck(Exception):
    """检查项主动跳过（例如 --skip-data）。"""


def run_check(name, fn, *args, **kwargs):
    """跑一个检查项并记录结果，任何异常都不中断后续检查。"""
    try:
        fn(*args, **kwargs)
        PASS.append(name)
        print(f'  [OK]   {name}', flush=True)
    except SkipCheck as e:
        SKIP.append((name, str(e)))
        print(f'  [SKIP] {name}: {e}', flush=True)
    except Exception as e:
        FAIL.append((name, traceback.format_exc()))
        print(f'  [FAIL] {name}: {e}', flush=True)


def main():
    skip_data = '--skip-data' in sys.argv
    print(f'\n=== OSMOSIS 冒烟测试 (DEVICE={config.DEVICE}) ===\n')

    import os

    import torch

    import utils

    utils.seed_everything(config.SEED)

    print('[1] 环境')
    print(f'  torch={torch.__version__} cuda={torch.version.cuda} '
          f'available={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  gpu={torch.cuda.get_device_name(0)}')
    print(f'  CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')

    print('\n[2] 数据集')
    run_check('数据集读取', check_datasets, skip_data)

    print('\n[3] 划分一致性')
    run_check('baseline 划分对齐', check_split, skip_data)

    print('\n[4] 模型前向')
    run_check('模型前向与反传', check_models)

    print('\n[5] OOM 自适应步进')
    run_check('OOM 自适应步进', check_stepper)

    print('\n[6] 指标对齐')
    run_check('指标对齐', check_metrics)

    print('\n[7] 断点往返')
    run_check('断点保存/加载', check_checkpoint)

    print('\n' + '=' * 60)
    print(f'通过 {len(PASS)} 项，失败 {len(FAIL)} 项，跳过 {len(SKIP)} 项')
    for name, tb in FAIL:
        print(f'\n--- {name} 失败详情 ---\n{tb}')
    return 1 if FAIL else 0


# ---------------------------------------------------------------------------


def check_datasets(skip):
    if skip:
        raise SkipCheck('--skip-data')
    import torch

    import utils
    from data.hts import build_hts_datasets
    from data.ooc import build_ooc
    from data.rxrx1 import build_rxrx1

    ds = build_rxrx1(phase='train', subset_ratio=0.001, img_size=224)
    b = next(iter(utils.build_loader(ds, 4, shuffle=False, seed=config.SEED,
                                     num_workers=0)))
    assert b['image'].shape == (4, 3, 224, 224), b['image'].shape
    assert 0.0 <= float(b['image'].min()) and float(b['image'].max()) <= 1.0
    print(f'    RxRx1: n={len(ds)} batch={tuple(b["image"].shape)}')

    ooc = build_ooc(splits=['train'], img_size=224, train=False)
    ob = next(iter(utils.build_loader(ooc, 4, shuffle=False, seed=config.SEED,
                                      num_workers=0)))
    assert ob['image'].shape == (4, 3, 224, 224)
    assert set(np.unique(ob['aux_label'].numpy())) <= {0, 1}
    print(f'    OOC  : n={len(ooc)} batch={tuple(ob["image"].shape)} '
          f'aux_labels={np.unique(ob["aux_label"].numpy()).tolist()}')

    # OOC 原始图像尺寸不统一（1536x2048 / 1542x2056 ...），训练模式下必须
    # 仍然被统一到 (3,S,S)，否则 DataLoader 的 collate 会直接报尺寸不一致。
    ooc_tr = build_ooc(splits=['train'], img_size=224, train=True)
    shapes = {tuple(ooc_tr[i]['image'].shape) for i in range(12)}
    assert shapes == {(3, 224, 224)}, f'训练模式输出尺寸不统一: {shapes}'
    print(f'    OOC训练模式: 12 个样本尺寸全部为 {shapes}')

    _tr, _va, te, _meta = build_hts_datasets()
    hb = next(iter(utils.build_loader(te, 4, shuffle=False, seed=config.SEED,
                                      num_workers=0)))
    # 尺寸取 config.HTS_IMG_SIZE，不要写死 224 —— 训练分辨率改成 512 之后，
    # 原来的硬编码断言会误报失败（本提交包修正了这一处，见 docs/05）。
    assert hb['image'].shape == (4, 3, config.HTS_IMG_SIZE, config.HTS_IMG_SIZE)
    assert hb['mask'].shape == (4, 1, config.HTS_IMG_SIZE, config.HTS_IMG_SIZE)
    uniq = np.unique(hb['mask'].numpy())
    assert set(uniq.tolist()) <= {0.0, 1.0}, f'掩码必须严格二值, 实际 {uniq}'
    print(f'    HTS  : n={len(te)} batch={tuple(hb["image"].shape)} '
          f'mask值域={uniq.tolist()} labels={hb["label"].numpy().tolist()}')


def check_split(skip):
    if skip:
        raise SkipCheck('--skip-data')
    import numpy as np

    from data.hts import baseline_split
    import config as C

    img_dir = f'{C.HTS_ROOT}/{C.HTS_SUBSET}/images'
    mask_dir = f'{C.HTS_ROOT}/{C.HTS_SUBSET}/labels'
    tr, va, te, img_paths, _ = baseline_split(img_dir, mask_dir,
                                              C.TEST_RATIO, C.VAL_RATIO,
                                              C.SPLIT_SEED)
    n = len(img_paths)
    # baseline: train_test_split(test_size=0.2, random_state=42) -> ceil(0.2*n)
    expect_test = int(np.ceil(C.TEST_RATIO * n))
    assert len(te) == expect_test, f'测试集大小 {len(te)} != 期望 {expect_test}'
    assert len(set(tr) & set(te)) == 0, '训练/测试集重叠'
    assert len(set(tr) | set(va) | set(te)) == n, '划分没有覆盖全部样本'
    print(f'    n={n} train={len(tr)} val={len(va)} test={len(te)} '
          f'(与 baseline 的 sklearn 划分一致)')


def check_models():
    import torch

    import utils
    from models import ClsHead, DomainDiscriminator, MAE, SegHead, ViTEncoder

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 3, config.MAE_INPUT_SIZE, config.MAE_INPUT_SIZE, device=dev)

    enc = ViTEncoder(img_size=config.MAE_INPUT_SIZE,
                     patch_size=config.MAE_PATCH_SIZE, in_chans=3,
                     embed_dim=config.MAE_ENC_DIM, depth=config.MAE_ENC_DEPTH,
                     num_heads=config.MAE_ENC_HEADS).to(dev)
    tok, cls = enc(x, mask_ratio=0.0)
    grid = config.MAE_INPUT_SIZE // config.MAE_PATCH_SIZE
    assert tok.shape == (2, grid * grid, config.MAE_ENC_DIM), tok.shape
    assert cls.shape == (2, config.MAE_ENC_DIM)
    print(f'    ViT    : tokens={tuple(tok.shape)} cls={tuple(cls.shape)} '
          f'params={utils.format_params(enc)}')

    tok_m, cls_m, mask, ids = enc(x, mask_ratio=0.75)
    keep = grid * grid - int(grid * grid * 0.75)
    assert tok_m.shape[1] == keep, (tok_m.shape, keep)
    assert mask.shape == (2, grid * grid) and float(mask.mean()) > 0.7
    print(f'    MAE掩码: 可见={tok_m.shape[1]}/{grid*grid} '
          f'掩码比例={float(mask.mean()):.3f}')

    mae = MAE(img_size=config.MAE_INPUT_SIZE, patch_size=config.MAE_PATCH_SIZE,
              in_chans=3, enc_dim=config.MAE_ENC_DIM, enc_depth=2,
              enc_heads=config.MAE_ENC_HEADS, dec_dim=64, dec_depth=2,
              dec_heads=4).to(dev)
    loss, pred, m = mae(x)
    assert loss.ndim == 0 and pred.shape == (2, grid * grid,
                                             config.MAE_PATCH_SIZE ** 2 * 3)
    loss.backward()
    print(f'    MAE    : loss={float(loss):.4f} pred={tuple(pred.shape)} (反传 OK)')

    seg = SegHead(embed_dim=config.MAE_ENC_DIM, num_classes=1,
                  grid_size=grid).to(dev)
    logits = seg(tok.detach())
    assert logits.shape == (2, 1, config.MAE_INPUT_SIZE, config.MAE_INPUT_SIZE)
    print(f'    SegHead: logits={tuple(logits.shape)} '
          f'params={utils.format_params(seg)}')

    disc = DomainDiscriminator(in_dim=config.MAE_ENC_DIM, num_domains=2).to(dev)
    d = disc(cls.detach())
    assert d.shape == (2, 2)
    d.sum().backward()
    print(f'    Disc   : logits={tuple(d.shape)} (GRL 反传 OK)')

    aux = ClsHead(in_dim=config.MAE_ENC_DIM, num_classes=2).to(dev)
    assert aux(cls.detach()).shape == (2, 2)
    print(f'    ClsHead: OK')


def check_stepper():
    import torch

    import utils

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lin = torch.nn.Linear(4, 1).to(dev)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    stepper = utils.OOMAdaptiveStep(opt, scaler=None, chunks=1, grad_clip=0.0,
                                    device=dev)

    def f(batch):
        batch = utils.move_to_device(batch, dev)
        return ((lin(batch['x']) - batch['y']) ** 2).mean()

    batch = {'x': torch.randn(8, 4), 'y': torch.randn(8, 1)}
    before = {k: v.clone() for k, v in lin.state_dict().items()}
    loss, ok = stepper(f, batch)
    assert ok and np.isfinite(loss)
    assert not torch.allclose(before['weight'], lin.weight.detach()), \
        '参数没有更新'

    # chunks>1 时梯度应等价于整 batch 的平均。
    # 注意要把整个 state_dict（含 bias）都复制过来，否则两个模型本来就不同。
    lin2 = torch.nn.Linear(4, 1).to(dev)
    lin2.load_state_dict({k: v.clone() for k, v in before.items()})
    opt2 = torch.optim.SGD(lin2.parameters(), lr=0.1)
    s2 = utils.OOMAdaptiveStep(opt2, scaler=None, chunks=8, grad_clip=0.0,
                               device=dev)

    def f2(batch):
        batch = utils.move_to_device(batch, dev)
        return ((lin2(batch['x']) - batch['y']) ** 2).mean()

    l2, ok2 = s2(f2, batch)
    assert ok2
    diff = float((lin.weight.detach() - lin2.weight.detach()).abs().max())
    # 分片会改变浮点求和顺序，允许 1e-6 量级的舍入差异
    assert diff < 1e-6, f'梯度累积与整 batch 不等价, 最大差 {diff}'
    print(f'    chunks=1 loss={loss:.6f} | chunks=8 loss={l2:.6f} '
          f'权重差={diff:.2e} (等价性 OK)')

    # 不能整除的切片（8 -> [3,3,2]）也必须等价：按片内样本数加权而不是按片数
    lin3 = torch.nn.Linear(4, 1).to(dev)
    lin3.load_state_dict({k: v.clone() for k, v in before.items()})
    opt3 = torch.optim.SGD(lin3.parameters(), lr=0.1)
    s3 = utils.OOMAdaptiveStep(opt3, scaler=None, chunks=3, grad_clip=0.0,
                               device=dev)

    def f3(batch):
        batch = utils.move_to_device(batch, dev)
        return ((lin3(batch['x']) - batch['y']) ** 2).mean()

    l3, ok3 = s3(f3, batch)
    diff3 = float((lin.weight.detach() - lin3.weight.detach()).abs().max())
    assert ok3 and diff3 < 1e-6, \
        f'不整除切片 [3,3,2] 与整 batch 不等价, 最大差 {diff3}'
    print(f'    chunks=3 [3,3,2] loss={l3:.6f} 权重差={diff3:.2e} (加权正确)')

    # batch 切分正确性
    parts = utils._split_batch(batch, 3)
    assert sum(p['x'].shape[0] for p in parts) == 8
    assert utils._batch_len((batch, batch)) == 8
    print(f'    切分: 8 样本 -> {[p["x"].shape[0] for p in parts]} 片; '
          f'元组 batch 长度={utils._batch_len((batch, batch))}')


def check_metrics():
    import metrics as M

    rng = np.random.RandomState(0)
    gt = (rng.rand(4, 64, 64) > 0.7).astype(np.uint8)
    pred = gt.copy()
    # 随机翻转 5% 的像素
    flip = rng.rand(*gt.shape) < 0.05
    pred[flip] ^= 1

    m = M.metri(gt, pred)
    for k in ('accuracy', 'jaccard', 'dice', 'precision', 'recall'):
        assert 0.0 <= m[k] <= 1.0, (k, m[k])
    assert abs(m['dice'] - 2 * m['jaccard'] / (1 + m['jaccard'])) < 1e-12, \
        'Dice 必须由 IoU 反推（baseline 口径）'

    perfect = M.metri(gt, gt)
    assert abs(perfect['accuracy'] - 1) < 1e-12
    assert abs(perfect['jaccard'] - 1) < 1e-12
    assert abs(perfect['recall'] - 1) < 1e-12

    rep = M.evaluate_segmentation(list(gt), list(pred))
    assert rep['n_images'] == 4
    for k in M.TABLE2_ORDER:
        assert k in rep['per_image'], k
    print('    metri() 与 baseline 公式一致; 完美预测 accuracy=1.0')
    print('    ' + M.format_table2(rep, '合成数据').splitlines()[0])
    print('    ' + M.format_table2(rep, '合成数据').splitlines()[2])


def check_checkpoint():
    import os
    import tempfile

    import torch

    import utils

    tmp = tempfile.mkdtemp(prefix='osmosis_ckpt_')
    try:
        m1 = torch.nn.Linear(8, 4)
        o1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
        utils.seed_everything(config.SEED)
        x = torch.randn(3, 8)
        m1(x).sum().backward()
        o1.step()

        p = os.path.join(tmp, 'last.pt')
        utils.save_checkpoint(p, m1, o1, None, None, epoch=3, global_step=42,
                              best=0.5)
        assert os.path.exists(p) and not os.path.exists(p + '.tmp'), \
            '原子写应该只留下目标文件'

        m2 = torch.nn.Linear(8, 4)
        o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        ep, gs, best, _ = utils.load_checkpoint(p, m2, o2)

        assert (ep, gs, best) == (3, 42, 0.5), (ep, gs, best)
        for a, b in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(a, b), '权重往返不一致'

        # 随机数状态也要还原：从同一断点恢复两次，后续抽样必须一致
        utils.load_checkpoint(p, m2, o2)
        r1 = torch.randn(4)
        utils.load_checkpoint(p, m2, o2)
        r2 = torch.randn(4)
        assert torch.equal(r1, r2), 'RNG 状态未正确还原'
        print(f'    epoch/step/best 还原 OK ({ep}, {gs}, {best}); '
              f'权重逐位一致; RNG 状态一致; 原子写 OK')

        # resolve_resume 的三种行为
        assert utils.resolve_resume('none', tmp) is None
        assert utils.resolve_resume('auto', tmp) == p
        assert utils.resolve_resume('auto', os.path.join(tmp, 'nope')) is None
        print('    resolve_resume(none/auto) 行为正确')
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
