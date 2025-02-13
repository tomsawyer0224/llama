from sentencepiece import SentencePieceTrainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--model_name", type=str, default="m")
    parser.add_argument("--model_type", type=str, default="bpe")

    args = parser.parse_args()

    SentencePieceTrainer.train(
        input=args.corpus,
        model_prefix=args.model_name,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
    print("finish training the tokenizer")
