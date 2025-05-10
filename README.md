# Calibration Dataset Builder for ExLlamaV2

This script generates a `.parquet` file containing full-length tokenized rows for use as a calibration dataset in ExLlamaV2 quantization. It automatically infers the model's context length from `config.json`, uses the associated tokenizer, and ensures a fixed number of complete rows are collected.

## Features

- Reads `max_position_embeddings` from `config.json`
- Supports Hugging Face or SentencePiece tokenizers
- Streams and tokenizes from a Hugging Face dataset (default: The Pile deduplicated)
- Ensures exactly N full rows of tokens (default: 100)
- Skips short or empty token outputs
- Raises an error if too few rows are collected, with suggestions
- Shuffles rows before writing
- Optionally decodes and displays a preview row
- Outputs `.parquet` compatible with `convert.py -c`

## Usage

```bash
python build_calib_dataset.py /path/to/model \
  --num_tokens 13107200 \
  --output mistral_calib.parquet
```

## Optional Flags

- `--output`: Path to output `.parquet` file
- `--num_tokens`: Total tokens to collect (default = `seq_len × 100`)
- `--seq_len`: Override the model’s configured context length
- `--dataset`: Hugging Face dataset to stream from
- `--seed`: Random seed for row shuffling (default: 42)
- `--dry_run`: Show configuration and exit
- `--no_confirm`: Skip the interactive confirmation prompt
- `--preview`: Decode and print the first row of tokens (Hugging Face tokenizer only)

## Example

```bash
python build_calib_dataset.py ./mistral-model \
  --num_tokens 13107200 \
  --output mistral_calib.parquet \
  --no_confirm \
  --preview
```

## Output

A `.parquet` file with 100 full rows of token sequences, suitable for use with:

```bash
convert.py -c mistral_calib.parquet
```

If the required number of full rows cannot be collected, the script will exit with an error and suggest increasing `--num_tokens` or choosing a dataset with longer samples.
