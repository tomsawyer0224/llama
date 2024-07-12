# This is a personal project, for educational purposes only!

# Based on:
  https://arxiv.org/abs/2302.13971 \
  https://arxiv.org/abs/2307.09288 \
  https://arxiv.org/abs/2108.12409 \
  https://github.com/Meta-Llama/llama \
  https://github.com/lucidrains/rotary-embedding-torch \
  https://huggingface.co/docs/transformers/main/en/model_doc/llama

# About this project:
  1. Llama is a family of LLMs developed by Meta with some replacements: \
     LayerNorm >> RMSNorm \
     Positional Encoding >> Rotary Position Embedding \
     Multihead Attention >> Grouped Query Attention \
     ReLU >> SwiGLU

  2. On this project, I built 4 Llama models (without KV cache) to test the impacts of positional encoding on the model. \
     LlamaHF: Llama model from HuggingFace (with ROPE) \
     Llama_abs: Llama with absolute positional encoding \
     Llama_abs_rel: Llama with absolute positional encoding and relative positional encoding (inspired by AliBi model) \
     Llama_rope: Llama with rotary position embedding

  3. Trained on the tiny_shakespeare dataset, see the "results" folder for more details. \
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/3ee9e3f9-a09a-47c7-9e7f-9f76d3aa9ff3)
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/55d77136-d19e-4335-8b67-5b9ccfd5ed9a)
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/c629c01a-b903-47a8-85de-99452d61f12f)
     ![image](https://github.com/tomsawyer0224/llama/assets/130035084/9c9c5591-6604-4439-85de-d1a3534c0409) \
     After training on 100 epochs, the training loss of Llama_rope is the best, Llama_HF is the worst. In contrast, the perplexity of Llama_rope is worst, Llama_HF is the best. Overfitting occurred here (the performance is good at training but bad at validation), maybe caused by model complexity or a lack of data to train. Because the training process is performed on a tiny dataset and a tiny model, the conclusions may not be exact and need to do more experiments.

# How to use:
  1. Clone this repo, cd to llama
  2. Install the requirements: pip install -q -r requirements.txt
  3. Training the tokenizer: run the below command, the tokenizer is located in the root directory
```
     python train_tokenizer.py \
      --corpus './data/tinyshakespeare.txt' \
      --vocab_size 8192 \
      --model_name 'tinyshakespeare' \
      --model_type 'bpe'
```
  5. Traning Llama: edit the config file (configs/llama_rope.yaml,...), then run the command
```
     !python train.py \
      --config_file './configs/llama_abs.yaml' \
      --max_epochs 100 \
      --ckpt_path './results/llama_abs/checkpoints/epoch=49-step=5500.ckpt' # when resume the training
```
  7. After training, logs and checkpoints will be saved to "results" folder \

Note: This project was built on Google Colab, it may not work on other platforms.
     

     

     
     
  
