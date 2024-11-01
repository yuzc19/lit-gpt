import argparse
import os
from functools import partial

import datasets
import numpy as np
import torch
from datasets import Dataset
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader
from tokenizer import Tokenizer


def mates_select(dataset_size, selection_size, args):
    dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(f"{args.output_dir}/{i}") for i in range(8)]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(metrics.shape)

    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise

    indices = np.argsort(metrics)
    indices = np.concatenate([indices[:40000], indices[-40000:]])

    return indices


def random_select(dataset_size, selection_size, args):
    rng = np.random.default_rng()
    return rng.choice(dataset_size, size=(selection_size,), replace=False)


METHODS = {
    "random": random_select,
    "mates": mates_select,
}


def get_indices(dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    select_it = METHODS[args.method]
    ls = select_it(dataset_size, selection_size, args)
    indices = list(map(int, ls))
    return indices


def are_tensors_equal(list1: list[torch.Tensor], list2: list[torch.Tensor]) -> bool:
    if len(list1) != len(list2):
        return False

    for tensor1, tensor2 in zip(list1, list2):
        if not torch.equal(tensor1, tensor2):
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="mates")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--use_doc", action="store_true")

    args = parser.parse_args()
    print(args)

    args.output_dir = f"/data/users/zichunyu/out/{args.model_name}/fineweb/sample-100BT/{args.ckpt}-data_influence_model-flan-prediction"
    # args.output_dir = f"/data/users/zichunyu/out/{args.model_name}/fineweb/sample-350BT/train/0/{args.ckpt}-data_influence_model-flan-bs-1-prediction"
    tokenizer = Tokenizer("checkpoints/EleutherAI/pythia-1b")
    dataset = StreamingDataset(
        input_dir="/data/users/zichunyu/data/fineweb/sample-100BT/train",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    shard_size = int(1e6)
    for i in range(8):
        sanity_check_dataset = (
            dataset[i * shard_size : i * shard_size + 5]
            + dataset[(i + 1) * shard_size - 5 : (i + 1) * shard_size]
        )
        store_dataset = torch.load(args.output_dir + f"/{i}/sanity_check.pt")
        print(i, are_tensors_equal(sanity_check_dataset, store_dataset))

    dataset_size = len(dataset)
    print(f">> Dataset size: {dataset_size}")
    indices = get_indices(dataset_size, 0, args)
    print(f">> Max index: {max(indices)}")

    optimize(
        fn=lambda index: dataset[index],
        inputs=indices,
        output_dir=f"/data/users/zichunyu/data/fineweb/sample-100BT/{args.model_name}/{args.method}/{args.ckpt}/bs-1-sample",
        num_workers=(os.cpu_count() // 8),
        chunk_bytes="200MB",
    )
