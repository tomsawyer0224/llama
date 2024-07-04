import torch
from torch import Tensor, LongTensor, nn
import transformers
from transformers import LlamaForCausalLM, LlamaConfig

from modules import (
    FeedForward, 
    LlamaBlock, 
    RMSNorm,
    RotaryEmbedding,
    SinusoidalPositionalEncoding
)

class Llama(nn.Module):
    def __init__(
        self,
        d_model: int = 4096,
        vocab_size: int = 32000,
        n_layers: int = 32,
        max_seq_len: int = 2048,
        query_heads: int = 32,
        kv_heads: int = 8,
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
        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model
        )

        if pos_emb_type == 'rope':
            #assert abs_config is None, 'dont provide abs_config when using rope'
            head_dim = d_model // query_heads
            if rope_config is None:
                rope_config = dict(
                    #dim = head_dim,
                    #seq_before_head_dim = True,
                    #use_xpos = False
                )
            rope_config['dim'] = head_dim # to ensure proper config is provided
            rope_config['seq_before_head_dim'] = True
            rope_config['use_xpos'] = False
            rope_config = RotaryEmbedding(**rope_config)
        else: # 'abs' or 'abs_rel'
            kv_embed_dim = (d_model // query_heads) * kv_heads
            if abs_config is None:
                abs_config = dict(
                    #d_query = d_model,
                    #d_key = kv_embed_dim,
                    #max_len = max_seq_len
                )
            abs_config['d_query'] = d_model # to ensure proper config is provided
            abs_config['d_key'] = kv_embed_dim 
            abs_config['max_len'] = max_seq_len
            abs_config = SinusoidalPositionalEncoding(**abs_config)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(
                LlamaBlock(
                    embed_dim=d_model,
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
                    hidden_dim=hidden_dim,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier
                )
            )
        self.norm = RMSNorm(dim=d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias = False)
    def forward(
        self, 
        x: LongTensor,
        attn_mask: Tensor = None,
        is_causal: bool = None,
        **kwargs
    ):
        '''
        args:
            x: (batch_size, seq_len)
        returns:
            tensor of shape (batch_size, seq_len, embed_dim)
        '''
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, is_causal=is_causal)
        x = self.norm(x)
        logit = self.linear(x)
        return logit
    @property
    def n_params(self):
        model_params = sum([p.numel() for p in self.parameters()])
        model_buffers = sum([b.numel() for b in self.buffers()])
        n_ps = round(model_params/1e6,2)
        n_bs = round(model_buffers/1e6,2)
        total = n_ps + n_bs
        return f'{total} (M total params): {n_ps} (params) + {n_bs} (buffers)'
    def accuracy(self, y_true, y_pred):
        '''
        args:
            y_true: LongTensor, shape (batch_size, seq_len)
            y_pred: LongTensor of shape (batch_size, seq_len) or
                    FloatTensor of shape (batch_size, seq_len, vocab_size)
        returns:
            accuracy
        '''
        if y_pred.ndim == 3:
            y_pred = y_pred.argmax(-1)
        acc = (y_true == y_pred).sum()/y_true.numel()
        return acc.item()
    def loss_fn(self, inp, tgt, ignore_index=-1):
        '''
        args:
            inp: input, shape (batch_size, seq_len, vocab_size)
            tgt: target (label), shape (batch_size, seq_len)
        
        loss = nn.functional.cross_entropy(
            inp.transpose(1,2), 
            tgt, 
            ignore_index = ignore_index,
            reduction = 'none'
        ).mean()
        '''
        b,s,v = inp.shape
        loss = nn.functional.cross_entropy(
            inp.view(-1, v), 
            tgt.view(-1), 
            ignore_index = ignore_index,
            #reduction = 'none'
        )
        return loss

class LlamaHF(nn.Module):
    def __init__(
        self,
        d_model: int = 4096, # hidden_size
        vocab_size: int = 32000, # vocab_size
        n_layers: int = 32, # num_hidden_layers
        max_seq_len: int = 2048, # max_position_embeddings
        query_heads: int = 32, # num_attention_heads
        kv_heads: int = 8, # num_key_value_heads
        dropout: float = 0.0, # attention_dropout
        hidden_dim: int = 4*4096, # refer to intermediate_size
        use_cache: bool = False,
    ):
        super().__init__()
        # convert hidden_dim to intermediate_size
        hidden_dim = int(2 * hidden_dim / 3)
        intermediate_size = 256 * ((hidden_dim + 256 - 1) // 256)
        
        config = LlamaConfig(
            vocab_size = vocab_size,
            hidden_size = d_model,
            num_hidden_layers = n_layers,
            max_position_embeddings = max_seq_len,
            num_attention_heads = query_heads,
            num_key_value_heads = kv_heads,
            attention_dropout = dropout,
            intermediate_size = intermediate_size,
            use_cache = use_cache
        )
        self.causal_model = LlamaForCausalLM(config)
    def forward(
        self, 
        x: LongTensor,
        attn_mask: Tensor = None, # attention_mask - padding mask
        is_causal: bool = None, # not neccesary
        **kwargs
    ):
        '''
        args:
            x: (batch_size, seq_len)
        returns:
            logit (batch_size, seq_len, vocab_size)
        '''
        output = self.causal_model(
            input_ids = x,
            attention_mask = attn_mask,
        )
        return output['logits']
    @property
    def n_params(self):
        model_params = sum([p.numel() for p in self.parameters()])
        model_buffers = sum([b.numel() for b in self.buffers()])
        n_ps = round(model_params/1e6,2)
        n_bs = round(model_buffers/1e6,2)
        total = n_ps + n_bs
        return f'{total} (M total params): {n_ps} (params) + {n_bs} (buffers)'
    def accuracy(self, y_true, y_pred):
        '''
        args:
            y_true: LongTensor, shape (batch_size, seq_len)
            y_pred: LongTensor of shape (batch_size, seq_len) or
                    FloatTensor of shape (batch_size, seq_len, vocab_size)
        returns:
            accuracy
        '''
        if y_pred.ndim == 3:
            y_pred = y_pred.argmax(-1)
        acc = (y_true == y_pred).sum()/y_true.numel()
        return acc.item()
    def loss_fn(self, inp, tgt, ignore_index=-1):
        '''
        args:
            inp: input, shape (batch_size, seq_len, vocab_size)
            tgt: target (label), shape (batch_size, seq_len)
        
        loss = nn.functional.cross_entropy(
            inp.transpose(1,2), 
            tgt, 
            ignore_index = ignore_index,
            reduction = 'none'
        ).mean()
        '''
        b,s,v = inp.shape
        loss = nn.functional.cross_entropy(
            inp.view(-1, v), 
            tgt.view(-1), 
            ignore_index = ignore_index,
            #reduction = 'none'
        )
        return loss


