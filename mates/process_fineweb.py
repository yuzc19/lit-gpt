import argparse
import os
from functools import partial

from datasets import Dataset, load_dataset
from litdata import optimize
from tokenizer import Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--split", type=int, default=0)

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        num_proc=(os.cpu_count() - 1),
        name="sample-350BT",
        split="train",
    )
    print("Total examples:", len(dataset))

    # Split the data in training and validation
    split_dataset = dataset.train_test_split(test_size=0.003, seed=42, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    tokenizer = Tokenizer("checkpoints/EleutherAI/pythia-1b")

    def tokenize(data: Dataset, index: int):
        yield tokenizer.encode(data[index]["text"], eos=True)

    # Split the data into 10 shards
    base = len(split_dataset["train"]) // 10
    rank = args.split
    print(rank, base)
    optimize(
        fn=partial(tokenize, split_dataset["train"]),
        inputs=list(
            range(
                rank * base,
                (rank + 1) * base if rank < 9 else len(split_dataset["train"]),
            )
        ),
        output_dir=f"{args.base_dir}/data/fineweb/sample-350BT/train/{rank}",
        num_workers=8,
        chunk_bytes="200MB",
    )
    if rank == 0:
        optimize(
            fn=partial(tokenize, split_dataset["val"]),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=f"{args.base_dir}/data/fineweb/sample-350BT/val",
            num_workers=8,
            chunk_bytes="200MB",
        )
