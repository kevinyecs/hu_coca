import torch
import torch.nn as nn
from einops import einsum, rearrange

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward
from language_encoder import SwiGLU
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = context_dim # or dim

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale
        q = q * self.scale
        # get key / values
        k, v = self.to_kv(context).chunk(2, dim=-1)
        # query / key similarity
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        '''
        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)
        '''
        return out