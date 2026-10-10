"""
下游任务头：分割头（Stage 3）、域判别器（Stage 2）、分类头。

分割头把 ViT 的 14x14 token 网格逐步反卷积回原图分辨率，输出单通道 logits。
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 梯度反转层（DANN）
# ---------------------------------------------------------------------------


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 前向恒等，反向取负并乘强度 -> 让编码器学到"域不可分"的特征
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


# ---------------------------------------------------------------------------
# 分割头
# ---------------------------------------------------------------------------


class SegHead(nn.Module):
    """
    (B, L, D) token -> (B, 1, H, W) logits。

    L 必须等于 grid*grid。上采样次数 = log2(patch_size)：
    因为 img_size = grid_size * patch_size，从 grid 还原到 img_size 需要翻
    patch_size 倍，也就是 log2(patch_size) 次 2x 上采样。
    例如 patch=16, grid=14 -> 4 次 -> 14→28→56→112→224。

    mode='deconv' 用 ConvTranspose2d(k2,s2)（原实现）；
    mode='bilinear' 用双线性插值 + 3x3 卷积（无棋盘伪影，边界更平滑，
    消融实验用）。

    归一化用 GroupNorm 而不是 BatchNorm：小 batch 下更稳，且与 batch 内统计无关。
    """

    def __init__(self, embed_dim=768, num_classes=1, grid_size=14,
                 patch_size=16, hidden=256, norm_groups=32, mode='deconv'):
        super().__init__()
        assert mode in ('deconv', 'bilinear')
        self.grid_size = grid_size
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=1),
            nn.GroupNorm(norm_groups, hidden),
            nn.GELU(),
        )

        num_ups = _log2(patch_size)
        blocks = []
        for _ in range(num_ups):
            if mode == 'deconv':
                blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(hidden, hidden, kernel_size=2, stride=2),
                    nn.GroupNorm(norm_groups, hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.GroupNorm(norm_groups, hidden),
                    nn.GELU(),
                ))
            else:
                blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.GroupNorm(norm_groups, hidden),
                    nn.GELU(),
                ))
        self.ups = nn.ModuleList(blocks)
        self.head = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, tokens):
        B, L, D = tokens.shape
        g = self.grid_size
        assert L == g * g, f'token 数 {L} 与 grid {g}x{g} 不匹配'
        x = tokens.transpose(1, 2).reshape(B, D, g, g)
        x = self.stem(x)
        for blk in self.ups:
            x = blk(x)
        return self.head(x)


def _log2(n):
    k = 0
    while 2 ** k < n:
        k += 1
    assert 2 ** k == n, f'必须是 2 的幂，收到 {n}'
    return k


# ---------------------------------------------------------------------------
# 域判别器
# ---------------------------------------------------------------------------


class DomainDiscriminator(nn.Module):
    """
    输入编码器的全局表征，输出"来自哪个域"的 logits。
    前面挂梯度反转层，与编码器对抗训练。
    """

    def __init__(self, in_dim=768, hidden=256, num_domains=2, lambd=1.0):
        super().__init__()
        self.lambd = lambd
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, num_domains),
        )

    def forward(self, x, lambd=None):
        return self.net(grad_reverse(x, self.lambd if lambd is None else lambd))


# ---------------------------------------------------------------------------
# 分类头
# ---------------------------------------------------------------------------


class ClsHead(nn.Module):
    """全局表征 -> 类别 logits（Stage 2 的 OOC good/bad 辅助分类任务）。"""

    def __init__(self, in_dim=768, num_classes=2, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
