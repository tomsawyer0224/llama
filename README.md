This is a personal project, for educational purpose only!

Based on: \
  https://arxiv.org/abs/2302.13971 \
  https://arxiv.org/abs/2307.09288 \
  https://arxiv.org/abs/2108.12409 \
  https://github.com/Meta-Llama/llama \
  https://github.com/lucidrains/rotary-embedding-torch \
  https://huggingface.co/docs/transformers/main/en/model_doc/llama

About this project:
  1. Llama is a family of LLMs developed by Meta with some replacements: \
     LayerNorm >> RMSNorm \
     Positional Encoding >> Rotary Position Embedding \
     Multihead Attention >> Grouped Query Attention \
     ReLU >> SwiGLU

  2. On this project, I built 4 Llama models (without KV cache) to test the impacts of positional encoding to the model \
     LlamaHF: Llama model from HuggingFace (with ROPE) \
     Llama_abs: Llama with absolute positional encoding \
     Llama_abs_rel: Llama with absolute positional encoding and relative positional encoding (inspired by AliBi model) \
     Llama_rope: Llama with rotary position embedding

  3. Trained on tiny shakespeare dataset, see the "results" folder for more details. \
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/3ee9e3f9-a09a-47c7-9e7f-9f76d3aa9ff3)
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/55d77136-d19e-4335-8b67-5b9ccfd5ed9a)
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/c629c01a-b903-47a8-85de-99452d61f12f)
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/9c9c5591-6604-4439-85de-d1a3534c0409) \
     After training on 100 epochs, the training loss of Llama_rope is best, Llama_HF is worst. In constrast, the perplexity of Llama_rope is worst, Llama_HF is best. Llama_abs_rel is better than Llama_abs. Overfitting is occurred here, maybe caused by model complexity or lack of data to train. Because the training process is performed on the tiny dataset and tiny model, so the conclusions may be not exact and need to do more experiments.



     

     
     
  
