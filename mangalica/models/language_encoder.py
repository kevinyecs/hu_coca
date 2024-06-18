import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.checkpoint import checkpoint as cp
from einops import einsum, rearrange
from typing import Optional

class LlamaConfig():
    """
    Transformer configuration class of DoT

    Args:
        n_tokens (int, defaults to 32000):
            Defines the number of different tokens the embedding layer holds.

        d_model (int, defaults to 1024):
            Dimension of the embedding representation.

        depth (int, defaults to 1):
            Number of blocks the model built upon.

        exp_factor (int, defaults to 4):
            SwiGLU hidden dimension multiplier (defaults to d_model * exp_factor)

        q_heads (int, defaults to 16):
            Number of query heads in the Grouped-Query Attention.

        kv_heads (int, defaults to 8):
            Number of key/value heads in the Grouped-Query Attention.

        norm_eps (float, defaults to 1e-6):
            Small value to handle numerical stability (prevents division with zero).

        max_seq_len (int, defaults to 2048):
            Context length the model was pretrained on.

        attn_dropout (float, defaults to 0.0)
            Dropout probability for attention.

        pad_token_id (int, defaults to None):
            The padding token id from tokenizer. We use the 'eos_token' as 'pad_token' to automatically mask out the attention matrix.
    """

    def __init__(self,
                 n_tokens: int = 32000,
                 d_model: int = 1024,
                 depth: int = 1,
                 exp_factor: int = 4,
                 q_heads: int = 16,
                 kv_heads: int = 4,
                 norm_eps: float = 1e-5,
                 max_seq_len: int = 2048,
                 attn_dropout: float = 0.0,
                 pad_token_id: Optional[int] = None):

        self.n_tokens = n_tokens
        self.d_model = d_model
        self.depth = depth
        self.exp_factor = exp_factor
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.attn_dropout = attn_dropout
        self.pad_token_id = pad_token_id

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm)
    https://arxiv.org/pdf/1910.07467.pdf

    A well-known explanation of the success of LayerNorm is its re-centering
    and re-scaling invariance property. However RMSNorm only focuses on
    re-scaling invariance and regularizes the summed inputs simply according
    to the root mean square statistic.

    Intuitively, RMSNorm simplifies LayerNorm by totally removing the
    mean statistic at the cost of sacrificing the invariance that mean
    normalization affords.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    References:
        https://github.com/facebookresearch/llama (Credit)
    """

    def __init__(self, dim: int, eps: Optional[float] = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).to(x.dtype)
        return self.scale * x

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([ -x2, x1 ], dim = -1)

def apply_rotary_embs(q, k, cos, sin):
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    https://arxiv.org/pdf/2104.09864.pdf

    Combines the concept of absolute and relative position embeddings.
    RoPE naturally incorporates relative position information through rotation
    matrix product instead of altering terms in the expanded formulation of
    additive position encoding when applied with self-attention.

    ## Length Extrapolatable Rotary Embeddings
    https://arxiv.org/pdf/2212.10554.pdf

    ## Position Interpolation
    https://arxiv.org/pdf/2306.15595.pdf

    Args:
        dim (int): Input dimension size.
        max_pos_embs (int): Default context size which was the model trained on.
        base (int): Base value for inverse frequency.
        scaling_factor (float): Interpolation value. Defaults to 1.0 (2.0 would be 2 * context_size interpolation)

    References:
        https://github.com/facebookresearch/llama (Credit)

    Notes:
        Inverse frequency is precomputed when module is created.
        Inverse frequency is recomputed after the sequence length exceeds the
        maximum sequence length.
    """

    def __init__(self,
                 dim: int,
                 max_seq_len: int = 2048,
                 base: int = 10000,
                 scaling_factor: float = 1.0,
                 device = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))

    def forward(self, pos_ids: torch.Tensor):
        seq_len = torch.max(pos_ids) + 1

        if seq_len > self.max_seq_len:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_seq_len) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2) / self.dim))

        freqs = torch.einsum('..., f -> ...f', pos_ids, self.inv_freq.to(pos_ids.device))
        emb = torch.cat(( freqs, freqs ), dim = -1)
        cos, sin = torch.cos(emb), torch.sin(emb)

        return (cos, sin)

class GQA(nn.Module):
    """
    Grouped-Query Attention (GQA)
    https://arxiv.org/pdf/2305.13245.pdf

    Only uses a single key-value head, drastically speeds up decoder inference.
    GQA divides the query heads into groups, each of which shares a single key
    and value head. The number of heads 'n_heads' should be choosed to meet
    the condition for 'n_heads' % 'n_kv_heads' == 0.

    Args: (unpacked from config)
        dim (int): Input dimension size.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of attention key/value heads.
        attn_dropout (float): Dropout probability for attention.

    Einsum Notations:
        b - batch size
        n - sequence length
        h - head dimension
        d - embedding dimension

    References:
        https://github.com/facebookresearch/llama (Credit)

    Future:
        [ ] Key/Value cache
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.q_heads = config.q_heads
        self.kv_heads = config.kv_heads if config.kv_heads is not None else config.q_heads
        self.head_dim = config.d_model // config.q_heads
        self.attn_dropout = config.attn_dropout

        self.q_proj = nn.Linear(config.d_model, config.q_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(config.q_heads * self.head_dim, config.d_model, bias = False)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor],
                rotary_embs: tuple[torch.Tensor, torch.Tensor]):

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.q_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.kv_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.kv_heads)

        cos, sin = rotary_embs
        q, k = apply_rotary_embs(q, k, cos, sin)

        scores = einsum(q + 1.0, k + 1.0, 'b h n d, b g s d -> b h n s')

        ## Mask out upper triangular to prevent look-ahead
        if attn_mask is not None:
            scores = scores + attn_mask

        ## Padding tokens are zero vectors by default (padding_idx set to pad_token_id) so their product is also 0,
        ## this way we could just change them to -inf and they will remain zero in the rest of the computation. But
        ## if we mask out any correlation with the padding tokens it would corrupt the attention matrix. To prevent
        ## it we just have to add 1.0 to the query and key vectors and their (padding tokens) product will be equal
        ## the length of the embedding vectors (head_dim), and it can be easily masked out to -inf. As for the rest
        ## in the attention matrix we have NaN vectors at the padding occurences, setting it to 0.0 will fix it.
        ## Also the normalization with sqrt(head_dim) can be moved after the masking process.
        scores = scores.where(scores != float(self.head_dim), float('-inf'))
        scores = (scores - 1.0) / math.sqrt(self.head_dim)
        attention = F.softmax(scores.float(), dim = -1).nan_to_num(0.0).to(x.dtype)
        attention = F.dropout(attention, p = self.attn_dropout, training = self.training)

        o = einsum(attention, v, 'b h n s, b g s d -> b h n d')
        o = rearrange(o, 'b h n d -> b n (h d)')

        return self.o_proj(o)

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU)
    https://arxiv.org/pdf/2002.05202v1.pdf

    This can be through of as a multiplicative skip connection
    which helps gradients flow across the layers.

    Args:
        dim (int): Input dimension size.
        exp_factor (float): Hidden dimension multiplier. Defaults to 2.
    """

    def __init__(self,
                 dim: int,
                 exp_factor: Optional[int] = 2):
        super().__init__()
        self.dim = dim
        self.exp_factor = exp_factor

        hidden_dim = dim * exp_factor
        self.uv_proj = nn.Linear(dim, hidden_dim * 2, bias = False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias = False)
        self.act = nn.SiLU()

    def forward(self, x):
        u, v = torch.chunk(self.uv_proj(x), 2, dim = -1)
        return self.out_proj(self.act(u) * v)

class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.dim = config.d_model

        self.attn = GQA(config)
        self.ffn = SwiGLU(
            dim = config.d_model,
            exp_factor = config.exp_factor
        )

        self.attn_norm = RMSNorm(config.d_model, eps = config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps = config.norm_eps)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor],
                rotary_embs: tuple[torch.Tensor, torch.Tensor]):

        x_norm = self.attn_norm(x)
        x = self.attn(x_norm, attn_mask, rotary_embs) + x

        x_norm = self.ffn_norm(x)
        x = self.ffn(x_norm) + x

        return x

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model_config = config
        self.n_tokens = config.n_tokens
        self.depth = config.depth

        self.embs = nn.Embedding(config.n_tokens, config.d_model, padding_idx = config.pad_token_id)
        self.blocks = nn.ModuleList([ LlamaBlock(config) for _ in range(config.depth) ])
        self.norm = RMSNorm(dim = config.d_model)
        self.to_logits = nn.Linear(config.d_model, config.n_tokens, bias = False)
        self.rope = RoPE(config.d_model // config.q_heads, config.max_seq_len)

        self.gradient_checkpointing = True

    def forward(self, x: torch.Tensor):
        _, seq_len = x.shape
        device = x.device

        pos_ids = torch.arange(seq_len).to(device)
        rotary_embs = self.rope(pos_ids)

        ## Maybe move attn_mask to Wrapper and pass here as argument (attn_mask)
        attn_mask = None
        if seq_len > 1:
            attn_mask = torch.full((seq_len, seq_len), float('-inf')).triu(diagonal = 1)
            attn_mask = attn_mask.to(device)

        x = self.embs(x)

        for block in self.blocks:
            if self.gradient_checkpointing:
                x = cp(block, x, attn_mask, rotary_embs, use_reentrant = False)
            else:
                x = block(x, attn_mask, rotary_embs)

        x = self.norm(x)
        logits = self.to_logits(x)
        return logits
