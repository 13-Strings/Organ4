"""
自研 ViT 编码器（MAE 风格），不依赖 timm，避免版本行为差异。

结构: PatchEmbed -> +pos_embed -> (可选随机掩码) -> [CLS] + Blocks -> LayerNorm
输出保留 14x14 的空间网格信息，供 Stage 3 的分割头反卷积使用。
"""

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """把 (B,3,H,W) 切成 patch 并线性映射到 embed_dim，输出 (B, L, D)。"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, 'img_size 必须能被 patch_size 整除'
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, D, g, g)
        x = x.flatten(2).transpose(1, 2)       # (B, L, D)
        return x


class Attention(nn.Module):
    """标准多头自注意力，qkv 融合成一个 Linear。"""

    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 用 PyTorch 原生 SDPA：显存更省，且在确定性模式下有专属 kernel
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """pre-norm Transformer block，可选 LayerScale（MAE 微调时用）。"""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 init_values=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.gamma_1 = (nn.Parameter(init_values * torch.ones(dim))
                        if init_values is not None else None)
        self.gamma_2 = (nn.Parameter(init_values * torch.ones(dim))
                        if init_values is not None else None)

    def forward(self, x):
        h = self.norm1(x)
        h = self.attn(h)
        x = x + (self.gamma_1 * h if self.gamma_1 is not None else h)
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + (self.gamma_2 * h if self.gamma_2 is not None else h)
        return x


class ViTEncoder(nn.Module):
    """
    输出:
      tokens: (B, L, D)  —— 保留空间网格（Stage 3 分割头用）
      cls:    (B, D)     —— 全局表征（Stage 2 域判别用）

    掩码只发生在 MAE 预训练里（forward 的 mask_ratio 参数），下游阶段传 0。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 init_values=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop, init_values)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_module)

    def forward(self, x, mask_ratio=0.0, return_tokens=True):
        """
        mask_ratio>0 时按 MAE 做随机掩码（仅预训练用），额外返回 ids_restore 与 mask。
        """
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, L, D)
        x = x + self.pos_embed[:, 1:, :]

        ids_restore = None
        mask = None
        if mask_ratio > 0:
            x, mask, ids_restore = random_masking(x, mask_ratio)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        x = torch.cat([cls, x], dim=1)                   # (B, 1+keep, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        tokens = x[:, 1:]
        if not return_tokens:
            tokens = None
        if mask_ratio > 0:
            return tokens, cls_out, mask, ids_restore
        return tokens, cls_out


def random_masking(x, mask_ratio):
    """
    MAE 随机掩码：按噪声排序取前 (1-ratio) 个作为可见 token。
    返回 (x_visible, mask, ids_restore)，mask 中 1 表示被遮住的位置。
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(N, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)            # 还原到原始顺序
    return x_visible, mask, ids_restore


def _init_module(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """可选的正弦位置编码（默认不用，MAE 原文用可学习 pos_embed）。"""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    emb_h = _sincos_1d(embed_dim // 2, grid[0])
    emb_w = _sincos_1d(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1).reshape(grid_size ** 2, embed_dim)


def _sincos_1d(dim, pos):
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
