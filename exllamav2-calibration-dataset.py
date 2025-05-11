import os
import sys
import random
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import sentencepiece as spm
import json
import tqdm

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Build calibration dataset for ExLlamaV2 quantization")
parser.add_argument("model_path", type=str, help="Path to model directory containing config.json")
parser.add_argument("--output", type=str, default=None, help="Output .parquet file path")
parser.add_argument("--num_tokens", type=int, default=None, help="Total number of tokens to collect")
parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to collect (overrides --num_tokens)")
parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length (tokens per row)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling rows")
parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated", help="Hugging Face dataset to use")
parser.add_argument("--dry_run", action="store_true", help="Preview configuration without running")
parser.add_argument("--no_confirm", action="store_true", help="Skip confirmation prompt")
parser.add_argument("--preview", action="store_true", help="Decode and print sample tokens from first row")
args = parser.parse_args()

# --- Resolve paths and config ---
model_path = args.model_path if args.model_path == "/" else args.model_path.rstrip("/")
model_name = os.path.basename(model_path)
output_parquet = args.output or f"{model_name}.parquet"

# --- Load model config manually ---
config_path = os.path.join(model_path, "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"[!] config.json not found in {model_path}")

with open(config_path, "r") as f:
    config_data = json.load(f)

default_seq_len = config_data.get("text_config", {}).get("max_position_embeddings")
if default_seq_len is None:
    raise ValueError("[!] Could not find 'text_config.max_position_embeddings' in config.json")

seq_len = args.seq_len or default_seq_len
if seq_len <= 0:
    raise ValueError("[!] Sequence length must be positive")

# --- Load tokenizer ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer_type = "hf"
    print(f"[+] Loaded Hugging Face tokenizer from {model_path}")
except Exception:
    if os.path.isdir(model_path) and "tokenizer.model" in os.listdir(model_path):
        tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(model_path, "tokenizer.model"))
        tokenizer_type = "sp"
        print("[+] Using SentencePiece tokenizer.model")
    else:
        raise ValueError("Could not load tokenizer from model path")

# --- Set num_tokens and num_rows ---
if args.num_rows is not None:
    num_rows = args.num_rows
    num_tokens = num_rows * seq_len
else:
    num_tokens = args.num_tokens or seq_len * 100
    num_rows = num_tokens // seq_len
actual_tokens = num_rows * seq_len

# --- Summary ---
print("=" * 60)
print(f"Model: {model_name}")
print(f"Tokenizer type: {tokenizer_type}")
print(f"Sequence length (used per row): {seq_len}")
print(f"Target total tokens (will collect): {actual_tokens:,}")
print(f"Target rows to collect: {num_rows}")
print(f"Dataset: {args.dataset}")
print(f"Output file: {output_parquet}")
print("=" * 60)
if args.dry_run:
    sys.exit(0)

if not args.no_confirm:
    confirm = input("Proceed with dataset generation? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

# --- Tokenize dataset ---
dataset = load_dataset(args.dataset, split="train", streaming=True)
rows = []

for sample in tqdm.tqdm(dataset, desc="Tokenizing"):
    text = sample.get("text") or sample.get("content")
    if not text:
        continue

    try:
        tokens = (
            tokenizer.encode(text, add_special_tokens=False)
            if tokenizer_type == "hf"
            else tokenizer.encode(text, out_type=int)
        )
    except Exception:
        continue

    if not tokens:
        continue

    for i in range(0, len(tokens), seq_len):
        chunk = tokens[i:i + seq_len]
        if len(chunk) < seq_len:
            continue
        rows.append(chunk)
        if len(rows) >= num_rows:
            break
    if len(rows) >= num_rows:
        break

# --- Ensure required rows were collected ---
if len(rows) < num_rows:
    raise RuntimeError(
        f"Only {len(rows)} rows collected (expected {num_rows}). "
        "Try increasing --num_tokens or using a dataset with more diverse or longer samples."
    )

# --- Shuffle rows ---
random.seed(args.seed)
random.shuffle(rows)

# --- Preview sample ---
if args.preview:
    print("[i] Sample decoded tokens (first row):")
    if tokenizer_type == "hf":
        print(tokenizer.decode(rows[0][:256]))
    else:
        print("[Skipped: SentencePiece decode not available]")

# --- Save to Parquet ---
df = pd.DataFrame({"tokens": rows})
df.to_parquet(output_parquet, index=False)

# --- Done ---
print("[✓] Done.")
print(f"Saved {len(df)} rows of {seq_len} tokens each.")
print(f"Total tokens: {len(df) * seq_len:,}")
