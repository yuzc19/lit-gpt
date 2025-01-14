import argparse
from litdata.streaming import StreamingDataset, TokensLoader, StreamingDataLoader
import torch
import numpy as np
from tqdm import tqdm
import os
from litgpt import Tokenizer
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, Features, Sequence, Value
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset shards.")
    parser.add_argument("--shard", type=int, required=True, help="Shard index")
    parser.add_argument("--num_shards", type=int, required=True, help="Total number of shards")
    return parser.parse_args()

def main():
    args = parse_args()
    shard_idx = args.shard
    num_shards = args.num_shards

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )

    # Initialize the dataset
    ds = StreamingDataset(
        input_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    ds_len = len(ds)
    shard_size = ds_len // num_shards
    start_idx = shard_idx * shard_size
    end_idx = (shard_idx + 1) * shard_size if shard_idx < num_shards - 1 else ds_len
    print(f"Processing shard {shard_idx} of size {end_idx - start_idx} from {start_idx} to {end_idx}")

    ds = ds[start_idx:end_idx]
    dataloader = DataLoader(ds, batch_size=64)
    data_iter = iter(dataloader)

    out = []
    for batch in tqdm(data_iter):
        decoded_batch = pythia_tokenizer.batch_decode(batch, skip_special_tokens=True)
        for text in decoded_batch:
            out.append({"text": text})

    features = Features({"text": Value("string")})
    processed_ds = Dataset.from_list(out, features=features)
    # processed_ds.save_to_disk(f"data/test_decode_fineweb/{shard_idx}")
    processed_ds.save_to_disk(f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train_decode/0/{shard_idx}")

if __name__ == "__main__":
    main()