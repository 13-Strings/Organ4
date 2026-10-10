"""
MAE (Masked Autoencoder) —— Stage 1 自监督预训练。

编码器只处理可见 patch（25%），解码器在全部 patch 上重建被遮住的像素。
预训练结束后只保留编码器给 Stage 2/3 使用（config.MAE_SAVE_DECODER 控制是否也存解码器）。
"""

import torch
import torch.nn as nn

from .vit import Attention, Mlp, ViTEncoder


class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 enc_dim=768, enc_depth=12, enc_heads=12,
                 dec_dim=512, dec_depth=8, dec_heads=16,
                 mask_ratio=0.75, norm_pix_loss=True, mlp_ratio=4.0):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size,
                                  in_chans=in_chans, embed_dim=enc_dim,
                                  depth=enc_depth, num_heads=enc_heads,
                                  mlp_ratio=mlp_ratio)
        num_patches = self.encoder.patch_embed.num_patches

        # ---- 轻量解码器 ----
        self.decoder_embed = nn.Linear(enc_dim, dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dec_dim))
        self.decoder_blocks = nn.ModuleList([
            _DecoderBlock(dec_dim, dec_heads, mlp_ratio) for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, patch_size ** 2 * in_chans, bias=True)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(_init_decoder)

    # ------------------------------------------------------------------
    def patchify(self, imgs):
        """(B,C,H,W) -> (B, L, p*p*C)，每个 patch 内像素拉平。"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H % p == 0 and W % p == 0
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(B, h * w, p * p * C)

    def forward(self, imgs):
        """返回 (loss, pred, mask)。mask 中 1 表示被遮住、参与损失的位置。"""
        tokens, _cls, mask, ids_restore = self.encoder(
            imgs, mask_ratio=self.mask_ratio)
        pred = self.forward_decoder(tokens, ids_restore)
        return self.compute_loss(pred, imgs, mask), pred, mask

    def compute_loss(self, pred, imgs, mask):
        """MSE 重建损失，只在被遮住的 patch 上计算。"""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                       # (B, L)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def forward_decoder(self, tokens, ids_restore):
        """可见 token -> 补上 mask token -> 还原顺序 -> 解码器 -> 像素预测。"""
        x = self.decoder_embed(tokens)                 # (B, keep, dec_dim)
        B, L_keep, D = x.shape
        L = ids_restore.shape[1]

        mask_tokens = self.mask_token.expand(B, L - L_keep, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))

        cls = self.decoder_pos_embed[:, :1, :].expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]                             # 丢掉解码器的 cls

    # ------------------------------------------------------------------
    def encoder_state_dict(self):
        """只导出编码器权重（下游 Stage 2/3 直接用）。"""
        return {k.replace('encoder.', ''): v
                for k, v in self.state_dict().items()
                if k.startswith('encoder.')}


class _DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _init_decoder(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def load_pretrained_encoder(encoder, ckpt_path, strict=True, verbose=True):
    """
    把 MAE 预训练权重（train_mae.py 导出的 encoder.pt，或完整 mae 断点）灌进编码器。
    返回 (missing, unexpected) 供人工检查。
    """
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if isinstance(state, dict) and 'model' in state:
        state = state['model']                       # 完整断点
    if isinstance(state, dict) and any(k.startswith('encoder.') for k in state):
        state = {k.replace('encoder.', ''): v for k, v in state.items()
                 if k.startswith('encoder.')}

    missing, unexpected = encoder.load_state_dict(state, strict=strict)
    if verbose:
        print(f'[预训练权重] 载入 {ckpt_path}', flush=True)
        print(f'  missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    return missing, unexpected
