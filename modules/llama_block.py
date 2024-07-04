import torch
from torch import nn, Tensor, LongTensor

from modules import RMSNorm, MultiheadGQA_v2

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int,
        multiple_of: int = 64,
        ffn_dim_multiplier: float|int = None
    ):
        '''
        Args:
            dim: Input dimension.
            hidden_dim: Hidden dimension of the feedforward layer.
            multiple_of: Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier: Custom multiplier for hidden dimension. Defaults to None.
        '''
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, dim, bias = False)
        self.w3 = nn.Linear(dim, hidden_dim, bias = False)
    def forward(self, x):
        x = self.w2(nn.functional.silu(self.w1(x))*self.w3(x))
        return x

class LlamaBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: torch.device|str = None,
        dtype: torch.dtype = None,
        pos_emb_type: str = 'rope',
        rope_config: dict[str, any] = None,
        abs_config: dict[str, any] = None,
        hidden_dim: int = 4*4096,
        multiple_of: int = 256,
        ffn_dim_multiplier: float|int = None
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim=embed_dim)
        self.GQA = MultiheadGQA_v2(
            embed_dim=embed_dim,
            query_heads=query_heads,
            kv_heads=kv_heads,
            dropout=dropout,
            bias=bias,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
            pos_emb_type=pos_emb_type,
            rope_config=rope_config,
            abs_config=abs_config,
        )
        self.norm2 = RMSNorm(dim=embed_dim)
        self.FFN = FeedForward(
            dim=embed_dim,
            hidden_dim=hidden_dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier
        )

    def forward(
        self, 
        x: Tensor, 
        attn_mask: Tensor = None, 
        is_causal: bool = False,
        **kwargs
    ):
        '''
        args:
            x: (batch_size, seq_len, embed_dim)
        returns:
            tensor like x
        '''
        residual1 = x
        x1 = self.norm1(x)
        x1, _ = self.GQA(x1,x1,x1, attn_mask=attn_mask, is_causal=is_causal)

        x2 = residual1 + x1
        residual2 = x2
        x3 = self.norm2(x2)
        x3 = self.FFN(x3)

        out = residual2 + x3
        return out


