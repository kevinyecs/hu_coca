import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum

class ViTConfig():
    def __init__(self,
                 emb_dim : int = 768,
                 num_labels : int = 1,
                 image_size : int = 224,
                 patch_size : int = 16,
                 depth : int = 12,
                 num_attn_heads : int = 12,
                 attn_head_dim : int = 64,
                 ffn_dim : int = 3072,
                 num_channels : int = 3,
                 attn_dropout : float = 0.0,
                 emb_dropout : float = 0.0):

        self.emb_dim = emb_dim
        self.num_labels = num_labels
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.attn_head_dim = attn_head_dim
        self.ffn_dim = ffn_dim
        self.num_channels = num_channels
        self.attn_dropout = attn_dropout
        self.emb_dropout = emb_dropout

class PositionwiseFFN(nn.Module):
    def __init__(self,
                 dim : int,
                 hidden_dim : int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)
        return x

class DualPatchNorm(nn.Module):
    def __init__(self,
                 patch_size: int,
                 patch_dim: int,
                 num_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_dim

        self.norm1 = nn.LayerNorm(patch_size * patch_size * num_channels)
        self.proj = nn.Linear(patch_size * patch_size * num_channels, patch_dim)
        self.norm2 = nn.LayerNorm(patch_dim)

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = self.norm1(x)
        x = self.proj(x)
        x = self.norm2(x)
        return x

class Attn(nn.Module):
    def __init__(self,
                 dim: int,
                 out_dim: int,
                 num_heads: int = 8,
                 head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, head_dim * num_heads)
        self.k_proj = nn.Linear(dim, head_dim * num_heads)
        self.v_proj = nn.Linear(dim, head_dim * num_heads)
        self.o_proj = nn.Linear(head_dim * num_heads, out_dim)

        self.scale = head_dim ** -0.5

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.num_heads)

        scores = einsum(q, k, 'b h n d, b h m d -> b h n m')
        attention = F.softmax(scores * self.scale, dim = -1)

        o = einsum(attention, v, 'b h n m, b h m d -> b h n d')
        o = rearrange(o, 'b h n d -> b n (h d)')

        return self.o_proj(o)

class ViTBlock(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.attn_norm = nn.LayerNorm(config.emb_dim)
        self.attn = Attn(
            dim = config.emb_dim,
            num_heads = config.num_attn_heads,
            head_dim = config.attn_head_dim,
            out_dim = config.emb_dim)

        self.ffn_norm = nn.LayerNorm(config.emb_dim)
        self.ffn = PositionwiseFFN(
            dim = config.emb_dim,
            hidden_dim = config.ffn_dim)

    def forward(self, x):

        x_norm = self.attn_norm(x)
        x = self.attn(x_norm) + x

        x_norm = self.ffn_norm(x)
        x = self.ffn(x_norm) + x

        return x

class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.model_config = config

        self.to_patch = DualPatchNorm(config.patch_size, config.emb_dim)
        self.blocks = nn.ModuleList([ ViTBlock(config) for _ in range(config.depth) ])

        self.cls_token = nn.Parameter(torch.randn(config.emb_dim, ))
        self.pos_embs = nn.Parameter(torch.randn(1 + (config.image_size // config.patch_size) ** 2, config.emb_dim))

        self.norm = nn.LayerNorm(config.emb_dim)
        self.to_labels = nn.Linear(config.emb_dim, config.num_labels)

    def forward(self,
                pixel_values: torch.Tensor):

        ## Convert 'pixel_values' to patch_embeddings
        x = self.to_patch(pixel_values)

        ## Concatenate 'cls_token' at the beginning of each batch
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = x.shape[0])
        x = torch.cat((cls_tokens, x), dim = 1)

        ## Positional embeddings
        x += self.pos_embs[None, :]

        ## Blocks
        for block in self.blocks:
            x = block(x)

        ## Separate the cls token
        x = x[:, 0]

        ## Classifier head
        x = self.norm(x)
        x = self.to_labels(x)

        return x
