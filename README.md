# LLaMA - Large Languare Model Meta AI
**LLaMA** is a family of LLMs developed by Meta with some replacements:
- LayerNorm ⟶ RMSNorm.
- Positional Encoding ⟶ Rotary Position Embedding.
- Multihead Attention ⟶ Grouped Query Attention.
- ReLU ⟶ SwiGLU.
# About this project:
1. This is a personal project, for educational purposes only!
2. This project is built to compare three position encoding methods: Rotary Position Embeddings, Absolute Position Encodings and Relative Position Encodings. There are four models:
   - Llama_rope: Llama with rotary position embeddings.
   - Llama_abs_rel: Llama with absolute position encoding and relative position encoding (inspired by the AliBi model).
   - Llama_abs: Llama with absolute position encodings.
   - LlamaHF: Llama model from Hugging Face (with RoPE).
3. Experiment:
   - Model size: ~33M params.
   - Dataset: [tiny_shakespeare](data/tinyshakespeare.txt).
   - Number of epochs: 100.
   - Platform: Google Colab free (T4 GPU).
4. Result:
   - Performance on the training phase: Llama_rope > Llama_abs_rel > Llama_abs > LlamaHF.
   - Performance on the validation phase: Llama_rope < Llama_abs_rel < Llama_abs < LlamaHF.
   - Overfitting occurred here (the model is too complex or the dataset is too small).
   ![image](results/epoch_100/train_loss_epoch.png) \
   
   ![image](https://github.com/tomsawyer0224/llama/assets/130035084/3ee9e3f9-a09a-47c7-9e7f-9f76d3aa9ff3)
   ![image](https://github.com/tomsawyer0224/llama/assets/130035084/55d77136-d19e-4335-8b67-5b9ccfd5ed9a)
   ![image](https://github.com/tomsawyer0224/llama/assets/130035084/c629c01a-b903-47a8-85de-99452d61f12f)
   ![image](https://github.com/tomsawyer0224/llama/assets/130035084/9c9c5591-6604-4439-85de-d1a3534c0409) \
# How to use:
1. Clone this repo, cd to llama.
2. Install the requirements: pip install -q -r requirements.txt.
3. Train the tokenizer: run the below command, the pre-trained tokenizer is located in the root directory.
```
python train_tokenizer.py \
--corpus './data/tinyshakespeare.txt' \
--vocab_size 8192 \
--model_name 'tinyshakespeare' \
--model_type 'bpe'
```
4. Train Llama: modify the config file (configs/llama_rope.yaml,...), then run the below command:
```
!python train.py \
--config_file './configs/llama_abs.yaml' \
--max_epochs 100 \
--ckpt_path './results/llama_abs/checkpoints/epoch=49-step=5500.ckpt' # when resume the training
```
# Based on:
https://arxiv.org/abs/2302.13971 \
https://arxiv.org/abs/2307.09288 \
https://arxiv.org/abs/2108.12409 \
https://github.com/Meta-Llama/llama \
https://github.com/lucidrains/rotary-embedding-torch \
https://huggingface.co/docs/transformers/main/en/model_doc/llama
