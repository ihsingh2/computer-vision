import math
import torch
import torch.nn.functional as F
from torch import nn


def get_sinusoidal_positional_embedding(n_positions, dim):
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, emb_dim, H/patch_size, W/patch_size]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        return x, (H, W)


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.lambda_init = 0.8

        # Differential projections
        self.W_q = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        self.W_k = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        self.W_v = nn.Linear(emb_dim, 2 * emb_dim, bias=False)

        # Learnable lambda‑params (per head)
        self.lq1 = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.lk1 = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.lq2 = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.lk2 = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        # Per-head GroupNorm over the 2 * head_dim channels
        self.group_norm = nn.GroupNorm(1, 2 * self.head_dim, affine=True)

        # Final projection
        self.fc_out = nn.Linear(2 * emb_dim, emb_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, emb_dim]
        B, N, _ = x.size()

        # Project and split into heads and (Q1, Q2), (K1, K2), V
        #   -> raw_q: [B, N, 2 * emb_dim] -> reshape -> [B, num_heads, N, 2 * head_dim]
        raw_q = self.W_q(x).view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        raw_k = self.W_k(x).view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        raw_v = self.W_v(x).view(B, N, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)

        # Split into two streams for differential attention
        q1, q2 = raw_q.chunk(2, dim=-1)    # each [B, num_heads, N, head_dim]
        k1, k2 = raw_k.chunk(2, dim=-1)    # each [B, num_heads, N, head_dim]
        v = raw_v                          # [B, num_heads, N, 2 * head_dim]

        # Scaled dot product
        A1 = (q1 @ k1.transpose(-2, -1)) / math.sqrt(self.head_dim)   # [B, num_heads, N, N]
        A2 = (q2 @ k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
        a1 = F.softmax(A1, dim=-1)
        a2 = F.softmax(A2, dim=-1)

        # Compute lambda per head
        lam1 = torch.exp((self.lq1 * self.lk1).sum(dim=1))
        lam2 = torch.exp((self.lq2 * self.lk2).sum(dim=1))
        lam  = lam1 - lam2 + self.lambda_init
        lam  = lam.view(1, self.num_heads, 1, 1)

        # Differential attention
        attn = a1 - lam * a2
        attn = self.dropout(attn)
        out = attn @ v # [B, num_heads, N, 2 * head_dim]

        # Per‑head GroupNorm: Merge B and heads as batch, move channels to dim=1
        Bh, C = B * self.num_heads, 2 * self.head_dim
        out = out.view(Bh, N, C).transpose(1, 2)   # [B * num_heads, C, N]
        out = self.group_norm(out)                               # [B * num_heads, C, N]
        out = out.transpose(1, 2).view(B, self.num_heads, N, C)  # [B, num_heads, N, 2 * head_dim]

        # Projection
        out = out.permute(0, 2, 1, 3).reshape(B, N, 2 * self.emb_dim)
        out = self.fc_out(out)  # [B, N, emb_dim]

        return out, attn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadDifferentialAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, attn = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn


class DifferentialVisionTransformer(nn.Module):
    def __init__(self,
                 image_size=32,
                 in_channels=3,
                 num_classes=10,
                 emb_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_hidden_dim=512,
                 patch_size=4,
                 pos_embed_type="1d",  # Options: "none", "1d", "2d", "sinusoidal"
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim)
        grid_h = grid_w = image_size // patch_size
        n_patches = grid_h * grid_w
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed_type = pos_embed_type
        self.dropout = nn.Dropout(dropout)

        if pos_embed_type == "none":
            # fixed zero buffer
            self.register_buffer("pos_embed", torch.zeros(1, n_patches + 1, emb_dim))
        elif pos_embed_type == "sinusoidal":
            pe = get_sinusoidal_positional_embedding(n_patches + 1, emb_dim)
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        elif pos_embed_type == "1d":
            # learned per-position
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, emb_dim))
        elif pos_embed_type == "2d":
            # separate row/col embeddings + CLS
            self.grid_size = (grid_h, grid_w)
            self.row_embed = nn.Parameter(torch.randn(1, grid_h, emb_dim // 2))
            self.col_embed = nn.Parameter(torch.randn(1, grid_w, emb_dim // 2))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, emb_dim))
        else:
            raise ValueError(f"Unknown pos_embed_type {pos_embed_type}")

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_hidden_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x, (H, W) = self.patch_embed(x)  # [B, num_patches, emb_dim]
        num_patches = x.shape[1]

        # Prepare [CLS] token and concat to patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, emb_dim]

        if self.pos_embed_type in ("none", "sinusoidal", "1d"):
            pos = self.pos_embed
        else:  # "2d"
            gh, gw = self.grid_size
            # [1, gh, gw, emb_dim//2]
            rx = self.row_embed.unsqueeze(2).repeat(1, 1, gw, 1)
            cx = self.col_embed.unsqueeze(1).repeat(1, gh, 1, 1)
            patch_pe = torch.cat([rx, cx], dim=-1).flatten(1, 2)  # [1, n_patches, emb_dim]
            pos = torch.cat([self.cls_pos, patch_pe], dim=1)     # [1, n_patches+1, emb_dim]

        x = x + pos.to(x.device)
        x = self.dropout(x)

        attentions = []  # to record attention maps for visualization
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attentions.append(attn)  # each attn: [B, num_heads, N, N]

        x = self.norm(x)
        cls_out = x[:, 0]  # classification based on CLS token
        logits = self.head(cls_out)
        return logits, attentions
