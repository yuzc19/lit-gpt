import argparse
import os
import logging

from functools import partial
import random
import torch

import datasets
from datasets import Dataset, load_dataset
from litdata import optimize
from tokenizer import Tokenizer
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_fineweb.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()
    rank = args.split


    start = rank * 10
    end = (rank + 1) * 10
    logger.info(f"loading data from {start}% to {end}%")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        num_proc=62,
        name="sample-350BT",
        split=f"train[{start}%:{end}%]",
    )
    # dataset = dataset.select(range(10000)) # test
    total_samples = len(dataset)
    logger.info(f"Total examples: {total_samples}")
    # total_samples = 518631063
    train_samples = int(total_samples * 0.997)
    logger.info(f"Train examples: {train_samples} Val examples: {total_samples - train_samples}")
    
    tokenizer = Tokenizer("checkpoints/EleutherAI/pythia-1b")

    #optimize train
    optimize(
        fn=lambda index: tokenizer.encode(dataset[index]["text"], eos=True),
        inputs=list(range(train_samples)),
        output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/{rank}",
        # output_dir=f"/data/datasets/hf_cache/test/0",
        num_workers=16,
        chunk_bytes="200MB",
    )
    
    #optimize val
    optimize(
        fn=lambda index: tokenizer.encode(dataset[index]["text"], eos=True),
        inputs=list(range(train_samples, total_samples)),
        output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/val/{rank}",
        # output_dir=f"/data/datasets/hf_cache/test/1",
        num_workers=16,
        chunk_bytes="200MB",
    )
