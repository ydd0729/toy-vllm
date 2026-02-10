"""
Tokenize text using HuggingFace tokenizers and dump token IDs.

Usage:
    python tokenize.py "The capital of France is" --model meta-llama/Llama-3.2-1B
    python tokenize.py --decode --ids 791 6864 315 9822 374 --model meta-llama/Llama-3.2-1B

The C++ engine reads/writes plain text files of token IDs.
"""

import argparse
import json
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to tokenize")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--decode", action="store_true", help="Decode mode: IDs → text")
    parser.add_argument("--ids", nargs="+", type=int, help="Token IDs to decode")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.decode:
        if not args.ids:
            parser.error("--decode requires --ids")
        text = tokenizer.decode(args.ids)
        print(text)
    else:
        if not args.text:
            parser.error("Provide text to tokenize")
        ids = tokenizer.encode(args.text)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(ids, f)
            print(f"Wrote {len(ids)} tokens to {args.output}")
        else:
            print(f"Tokens ({len(ids)}): {ids}")
            print(f"Decoded: {[tokenizer.decode([t]) for t in ids]}")


if __name__ == "__main__":
    main()
