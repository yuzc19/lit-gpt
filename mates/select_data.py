import argparse
import os
from functools import partial

import datasets
import numpy as np
import torch
from datasets import Dataset
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader


def select(dataset_size, selection_size, args):
    if args.method == "mates":
        dataset = datasets.concatenate_datasets(
            [
                datasets.load_from_disk(f"{args.output_dir}/{i}")
                for i in range(args.shard_num)
            ]
        )
        metrics = np.array(dataset["prediction"]).reshape(-1)
    else:
        metrics = np.zeros(dataset_size)
    print(">> Metrics shape:", metrics.shape)
    metrics = metrics / args.temp
    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise
    return np.argpartition(metrics, selection_size)[:selection_size]


def get_indices(dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    ls = select(dataset_size, selection_size, args)
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
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--shard_num", type=int, default=18)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--warmup", action="store_true")

    args = parser.parse_args()
    print(args)

    args.output_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-350BT/train/{args.split}/{args.ckpt}-data_influence_model-flan-prediction"
    dataset = StreamingDataset(
        input_dir=f"{args.base_dir}/data/fineweb/sample-350BT/train/{args.split}",
        item_loader=TokensLoader(block_size=2048 + 1),
    )

    if args.method == "mates":
        shard_size = len(dataset) // args.shard_num
        for i in range(args.shard_num):
            if i == args.shard_num - 1:
                sanity_check_dataset = (
                    dataset[i * shard_size : i * shard_size + 5]
                    + dataset[len(dataset) - 5 : len(dataset)]
                )
            else:
                sanity_check_dataset = (
                    dataset[i * shard_size : i * shard_size + 5]
                    + dataset[(i + 1) * shard_size - 5 : (i + 1) * shard_size]
                )
            store_dataset = torch.load(args.output_dir + f"/{i}/sanity_check.pt")
            print(i, are_tensors_equal(sanity_check_dataset, store_dataset))

    dataset_size = len(dataset)
    print(f">> Dataset size: {dataset_size}")
    # Hard coding to be fixed
    selection_size = 5120000
    indices = get_indices(dataset_size, selection_size, args)
    print(f">> Max index: {max(indices)}")

    if args.warmup:
        # warmup
        args.ckpt = 0
    optimize(
        fn=lambda index: dataset[index],
        inputs=indices,
        output_dir=f"{args.base_dir}/data/fineweb/sample-350BT/train/{args.split}/{args.model_name}/{args.method}/{args.ckpt}/{selection_size}",
        num_workers=(os.cpu_count() // 8),
        chunk_bytes="200MB",
    )
