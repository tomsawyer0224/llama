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

  2. On this project, I built 4 Llama models (without KV cache) to test the impacts of positional encoding to the model
     LlamaHF: Llama model from HuggingFace (with ROPE)
     Llama_abs: Llama with absolute positional encoding
     Llama_abs_rel: Llama with absolute positional encoding and relative positional encoding (inspired by AliBi model)
     Llama_rope: Llama with rotaty position embedding

  3. Trained on tiny shakespeare dataset
     
  
