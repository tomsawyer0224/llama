from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor, nn

#from modules import RotaryEmbedding, SinusoidalPositionalEncoding
from .rope import RotaryEmbedding
from .spe import SinusoidalPositionalEncoding
def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    force_grouped: bool = False,
):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        # Separate the query heads into 'num_head_groups' chunks, and fold the group
        # dimension into the batch dimension.  This allows us to compute the attention
        # for each head in parallel, then sum over all of the groups at the end.
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        # If the number of query/key heads is equal, we can skip grouping the queries,
        # and just use the standard sdot product attention.
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    if is_causal:
        # Mask out the upper triangular portion of the attention matrix. This prevents
        # the model from attending to tokens in the future.
        mask = torch.ones(
            (bq, nq, nk),
            device=query.device,
            dtype=torch.bool,
        ).tril_()

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        # If mask is 2D, assume that it is applied to the key/value sequence dimension.
        # Else if mask is 3D, assume that it is applied to the query/key/value sequence
        # dimension for all attention heads.
        #
        # Users could also provide a 4D mask, which is applied to the query/key/value
        # sequence dimension for each attention head (though I don't have a particular
        # use case in mind for that).
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        # Mask similarity values by setting them to negative infinity.  This guarantees
        # that they will not contribute to the softmax computation below.
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b h n d -> b n h d")

    attn_weights: Optional[Tensor] = None
    if need_weights:
        # Move the sequence dimensions back to positions 1, 2.  Move the head dimension
        # to position 3.  This more closely matches the return shape of the attention
        # output: (b, n, h, d).
        attn_weights = rearrange(attention, "b h n s -> b n s h")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights


class MultiheadGQA(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

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
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by "
                f"kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        # Query projection layer is the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        # Key/value projection layers have a smaller output dimension, so that
        # the we have fewer key/value attention heads after reshaping.
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                kv_embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        # Grouped attention output will have the same embedding dimension as the
        # key/value Tensors.  So the output projection layer needs to accept the
        # same dimension (kv_embed_dim).
        self.out_proj = nn.Linear(
            kv_embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        # TODO
        # attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x, attn = scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            # TODO
            # mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            force_grouped=False,
        )
        x = rearrange(x, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'!  We follow the MAGNETO
        # architecture (https://arxiv.org/pdf/2210.06423.pdf), which applies an extra
        # layer norm before the linear output projection.  The cross-attention layer in
        # the MAGNETO decoder does not include this layer norm, so users have the
        # option to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, attn

def scaled_dot_product_gqa_v2(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    force_grouped: bool = False,
    relative_score: Tensor = None
):
    """Scaled dot product attention with support for grouped queries.
    differs from orginal version: adding relative_score, repeat kv_head to query_head

    Einstein notation:
    - b: batch size
    - n / s: sequence length, n = target, s = source
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
            entry[i,j] = False means qi does not pay attention with kj
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)
        relative_score: if provided, tensor of shape (b, h, n, s)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        # Separate the query heads into 'num_head_groups' chunks, and fold the group
        # dimension into the batch dimension.  This allows us to compute the attention
        # for each head in parallel, then sum over all of the groups at the end.

        #query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        #similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
        key = repeat(key, 'b hk s d -> b g hk s d', g = num_head_groups)
        key = rearrange(key, 'b g hk s d -> b (g hk) s d')
        value = repeat(value, 'b hv s d -> b g hv s d', g = num_head_groups)
        value = rearrange(value, 'b g hv s d -> b (g hv) s d')
    #else:
        # If the number of query/key heads is equal, we can skip grouping the queries,
        # and just use the standard sdot product attention.
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    if is_causal:
        # Mask out the upper triangular portion of the attention matrix. This prevents
        # the model from attending to tokens in the future.
        mask = torch.ones(
            (bq, nq, nk),
            device=query.device,
            dtype=torch.bool,
        ).tril_()

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        # If mask is 2D, assume that it is applied to the key/value sequence dimension.
        # Else if mask is 3D, assume that it is applied to the query/key/value sequence
        # dimension for all attention heads.
        #
        # Users could also provide a 4D mask, which is applied to the query/key/value
        # sequence dimension for each attention head (though I don't have a particular
        # use case in mind for that).
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        # Mask similarity values by setting them to negative infinity.  This guarantees
        # that they will not contribute to the softmax computation below.
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    if relative_score is not None:
        attention = F.softmax(similarity / scale + relative_score, dim=-1)
    else:
        attention = F.softmax(similarity / scale, dim=-1)

    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b h n d -> b n h d")

    attn_weights: Optional[Tensor] = None
    if need_weights:
        # Move the sequence dimensions back to positions 1, 2.  Move the head dimension
        # to position 3.  This more closely matches the return shape of the attention
        # output: (b, n, h, d).
        attn_weights = rearrange(attention, "b h n s -> b n s h")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights


class MultiheadGQA_v2(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

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
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        pos_emb_type: str = 'rope',
        rope_config: dict[str, any] = None,
        abs_config: dict[str, any] = None
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by "
                f"kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        # Query projection layer is the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        # Key/value projection layers have a smaller output dimension, so that
        # the we have fewer key/value attention heads after reshaping.
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        # Grouped attention output will have the same embedding dimension as the
        # key/value Tensors.  So the output projection layer needs to accept the
        # same dimension (kv_embed_dim).
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        assert pos_emb_type in ['rope', 'abs', 'abs_rel'], \
        "pos_emb_type should be in ['rope', 'abs', 'abs_rel']"
        # abs = absolute positional encoding
        # abs_rel = absolute positional encoding with adding relative score in attention
        if pos_emb_type == 'rope':
            #assert abs_config is None, 'dont provide abs_config when using rope'
            if rope_config is None:
                rope_config = dict(
                    #dim = head_dim,
                    #seq_before_head_dim = True,
                    #use_xpos = False
                )
            if isinstance(rope_config, dict):
                rope_config['dim'] = head_dim # to ensure proper config is provided
                rope_config['seq_before_head_dim'] = True
                rope_config['use_xpos'] = False
                self.rope = RotaryEmbedding(**rope_config)
            else: # if is object
                self.rope = rope_config
        else: # 'abs' or 'abs_rel'
            if abs_config is None:
                abs_config = dict(
                    #d_query = embed_dim,
                    #d_key = kv_embed_dim
                )
            if isinstance(abs_config, dict):
                abs_config['d_query'] = embed_dim # to ensure proper config is provided
                abs_config['d_key'] = kv_embed_dim 
                self.spe = SinusoidalPositionalEncoding(**abs_config)
            else: # is object
                self.spe = abs_config
            if pos_emb_type == 'abs_rel':
                self.relative_weight = nn.Parameter(torch.ones((query_heads,)))
        self.pos_emb_type = pos_emb_type
        self._reset_parameters()
    def _get_relative_score(self, q, k, relative_weight):
        '''
        gets relative score from q(target) and k(source); more nearer, more stronger
        args:
            q, k: (b,seq_len,h,d)
        returns:
            tensor of shape (b,h,q_seq_len,k_seq_len)
        '''
        b, q_seq_len, h, d = q.shape
        k_seq_len = k.shape[-3]
        M = max(q_seq_len, k_seq_len)
        A = torch.arange(0,k_seq_len).repeat((q_seq_len,1))
        B = torch.arange(0,q_seq_len)[:,None]
        C = A - B
        D = torch.where(C >= 0, M - C, M + C) # (q_seq_len, k_seq_len) = (n, s)
        W = repeat(relative_weight, 'h -> () h () ()')
        relative_score = repeat(D, 'n s -> b h n s', b = b, h = h).to(W.device)
        
        return (relative_score/M)*W # (b, h, q_seq_len, k_seq_len)

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        # TODO
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query) # (b, n, embed_dim)
        k: Tensor = self.k_proj(key) # (b, s, kv_embed_dim)
        v: Tensor = self.v_proj(value) # (b, s, kv_embed_dim)

        if self.pos_emb_type in ['abs', 'abs_rel']:
            q, k = self.spe(q, k)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.

        if self.pos_emb_type == 'rope':
            #q, k = self.rope.rotate_queries_and_keys(q, k)
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)
        if self.pos_emb_type == 'abs_rel':
            relative_score = self._get_relative_score(q, k, self.relative_weight)
        else:
            relative_score = None
        x, attn = scaled_dot_product_gqa_v2(
            query=q,
            key=k,
            value=v,
            # TODO
            mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            force_grouped=False,
            relative_score=relative_score
        )
        x = rearrange(x, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'!  We follow the MAGNETO
        # architecture (https://arxiv.org/pdf/2210.06423.pdf), which applies an extra
        # layer norm before the linear output projection.  The cross-attention layer in
        # the MAGNETO decoder does not include this layer norm, so users have the
        # option to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, attn










