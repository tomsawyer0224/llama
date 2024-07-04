import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch

from models import Llama

class Test_Llama(unittest.TestCase):
    def setUp(self):
        rope_config = dict(
            dim = 2,
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
        abs_config = dict(
            d_query = 512, 
            d_key = 1, 
            dropout = 0.1, 
            max_len = 2048
        )
        self.model_config = dict(
            d_model = 512, #4096,
            vocab_size = 10000, #32000,
            n_layers = 8, #32,
            max_seq_len = 2048,
            query_heads = 8, #32,
            kv_heads = 4, #8,
            dropout = 0.0,
            bias = True,
            layer_norm = True,
            layer_norm_eps = 1e-5,
            gamma_init = 1.0,
            device = None,
            dtype = None,
            pos_emb_type = 'rope',
            #pos_emb_type = 'abs',
            #pos_emb_type = 'abs_rel',
            rope_config = rope_config,
            abs_config = abs_config,
            hidden_dim = 4*512,
            multiple_of = 256,
            ffn_dim_multiplier = None
        )
        self.Llama_model = Llama(**self.model_config)
    def test_forward(self):
        print('---test_forward---')
        x = torch.randint(0,100,(2,6))
        attn_mask = torch.randint(0,2,(2,6,6)).bool()
        logit1 = self.Llama_model(x)
        logit2 = self.Llama_model(x, attn_mask = None, is_causal = True)
        logit3 = self.Llama_model(x, attn_mask = attn_mask, is_causal = None)
        print(f'x: {x.shape}\n{x}')
        print(f'logit1: {logit1.shape}\n{logit1}')
        print(f'logit2: {logit2.shape}\n{logit2}')
        print(f'logit3: {logit3.shape}\n{logit3}')
        print()
    def test_n_params(self):
        print('---test_n_params---')
        print(self.Llama_model.n_params)
    def test_loss_fn_and_accuracy(self):
        print('---test_loss_fn and accuracy---')
        #inp = torch.randint(0,20, (4,10,512))
        x = torch.randint(0,20,(4,100))
        out = self.Llama_model(x)
        loss = self.Llama_model.loss_fn(out,x)
        acc = self.Llama_model.accuracy(x,out)
        print(f'x: {x.shape}')
        print(f'out: {out.shape}')
        print(f'loss: {loss}')
        print(f'acc: {acc}')
        print()


        

if __name__=="__main__":
    unittest.main()
        
