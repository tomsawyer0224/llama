import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch

from modules import (
    MultiheadGQA,
    MultiheadGQA_v2, 
    RMSNorm, 
    RotaryEmbedding, 
    SinusoidalPositionalEncoding
)

class TestModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.embed_dim = 512
        self.n_heads = 16
        self.head_dims = self.embed_dim // self.n_heads
        self.q_heads = self.n_heads
        self.d_query = self.head_dims
        self.kv_heads = 8
        self.d_key = self.d_query
    def test_RMSNorm(self):
        print('---test_RMSNorm---')
        x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim
        )
        rms_norm = RMSNorm(dim = self.embed_dim)
        out = rms_norm(x)
        print(f'x: {x.shape}')
        print(f'out: {out.shape}')
        print()
    ''''''
    def test_RotaryEmbedding(self):
        print('---test_RotaryEmbedding---')
        q = torch.randn(
            self.batch_size, self.seq_len, self.n_heads, self.head_dims
        )
        k = torch.randn(
            self.batch_size, self.seq_len, self.kv_heads, self.head_dims
        )
        rope = RotaryEmbedding(
            dim = self.head_dims,
            seq_before_head_dim = True,
            use_xpos = True
        )
        rotated_q, rotated_k = rope.rotate_queries_and_keys(q, k)
        print(f'q: {q.shape}')
        print(f'k: {k.shape}')
        print(f'rotated_q: {rotated_q.shape}')
        print(f'rotated_k: {rotated_k.shape}')
        print()
    def test_SinusoidalPositionalEncoding(self):
        print(f'---test_SinusoidalPositionalEncoding---')
        q = torch.randn(
            self.batch_size, self.seq_len, self.d_query
        )
        k = torch.randn(
            self.batch_size, self.seq_len, self.d_key
        )
        spe = SinusoidalPositionalEncoding(
            d_query = self.d_query,
            d_key = self.d_key
        )
        encoded_q, encoded_k = spe(q, k)
        print(f'q: {q.shape}')
        print(f'k: {k.shape}')
        print(f'encoded_q: {encoded_q.shape}')
        print(f'encoded_k: {encoded_k.shape}')
        print()
    
    def test_MultiheadGQA_v2(self):
        print('---test_MultiheadGQA_v2---')
        rope_config = dict(
            dim = self.head_dims,
            custom_freqs = None,
            freqs_for = 'lang',
            theta = 10000,
            max_freq = 10,
            num_freqs = 1,
            learned_freq = False,
            use_xpos = False,
            xpos_scale_base = 512,
            interpolate_factor = 1.,
            theta_rescale_factor = 1.,
            seq_before_head_dim = True,
            cache_if_possible = True
        )
        GQA_rope = MultiheadGQA_v2(
            embed_dim = self.embed_dim,
            query_heads = self.q_heads,
            kv_heads = self.kv_heads,
            pos_emb_type = 'rope',
            rope_config = rope_config
        )
        n_kv_heads = self.q_heads // self.kv_heads
        kv_embed_dim = self.embed_dim // n_kv_heads
        abs_config = dict(
            d_query = self.embed_dim, 
            d_key = kv_embed_dim, 
            dropout = 0.1, 
            max_len = 1024
        )
        GQA_abs = MultiheadGQA_v2(
            embed_dim = self.embed_dim,
            query_heads = self.q_heads,
            kv_heads = self.kv_heads,
            pos_emb_type = 'abs',
            abs_config = abs_config
        )
        GQA_abs_rel = MultiheadGQA_v2(
            embed_dim = self.embed_dim,
            query_heads = self.q_heads,
            kv_heads = self.kv_heads,
            pos_emb_type = 'abs_rel',
            abs_config = abs_config
        )

        x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim
        )

        out_rope, _ = GQA_rope(x,x,x, is_causal = True)
        out_abs, _ = GQA_abs(x,x,x, is_causal = True)
        out_abs_rel, _ = GQA_abs_rel(x,x,x, is_causal = True)
        print(f'x: {x.shape}')
        print(f'out_rope: {out_rope.shape}')
        print(f'out_abs: {out_abs.shape}')
        print(f'out_abs_rel: {out_abs_rel.shape}')
        print()

if __name__=="__main__":
    unittest.main()

