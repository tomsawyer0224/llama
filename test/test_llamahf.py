import sys
if not '.' in sys.path:
    sys.path.append('.')
import unittest
import torch

from models import LlamaHF

class Test_Llama(unittest.TestCase):
    def setUp(self):
        self.model_config = dict(
            d_model = 512, # hidden_size
            vocab_size = 10000, # vocab_size
            n_layers = 8, # num_hidden_layers
            max_seq_len = 2048, # max_position_embeddings
            query_heads = 8, # num_attention_heads
            kv_heads = 4, # num_key_value_heads
            dropout = 0.0, # attention_dropout
            hidden_dim = 4*512, # refer to intermediate_size
            use_cache = False,
        )
        self.Llama_model = LlamaHF(**self.model_config)
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
        
