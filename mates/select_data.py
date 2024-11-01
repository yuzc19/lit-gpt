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
        [
            datasets.load_from_disk(f"{args.output_dir}/{i}")
            for i in range(args.shard_num)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)
    metrics = metrics / args.temp
    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise
    return np.argpartition(metrics, selection_size)[:selection_size]


def edu_select(dataset_size, selection_size, args):
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(f"{args.output_dir}/{i}")
            for i in range(args.shard_num)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)
    print(">> Metrics mean:", metrics.mean())
    return np.argpartition(-metrics, selection_size)[:selection_size]


def fasttext_select(dataset_size, selection_size, args):
    dataset = datasets.load_from_disk(args.output_dir)
    metrics = np.array(dataset["prediction"]).reshape(-1)
    return np.argpartition(-metrics, selection_size)[:selection_size]


def random_select(dataset_size, selection_size, args):
    rng = np.random.default_rng()
    return rng.choice(dataset_size, size=(selection_size,), replace=False)


METHODS = {
    "random": random_select,
    "fasttext-oh-eli5": fasttext_select,
    "fineweb-edu": edu_select,
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
    parser.add_argument("--model_name", type=str, default="pythia-410m")
    parser.add_argument("--method", type=str, default="random")
    # 11.5 for the fineweb-edu
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--use_doc", action="store_true")

    args = parser.parse_args()
    print(args)

    tokenizer = Tokenizer("checkpoints/EleutherAI/pythia-1b")
    if not args.use_doc:
        # args.output_dir = f"../manifold/scaling_mates/tc_out/step-10000/pythia-1b"
        # args.output_dir = f"/data/users/zichunyu/out/fineweb/sample-100BT/fasttext-oh-eli5-prediction"
        args.output_dir = f"../out/fineweb/sample-350BT/train/0/fineweb-edu-prediction"
        # args.output_dir = f"/data/users/zichunyu/out/{args.model_name}/fineweb/sample-350BT/train/1/{args.ckpt}-data_influence_model-flan-bs-1-prediction"
        # args.output_dir = f"../manifold/scaling_mates/out/{args.model_name}/fineweb/sample-350BT/train/0/{args.ckpt}-data_influence_model-flan-prediction"
        dataset = StreamingDataset(
            input_dir="/data/users/zichunyu/data/fineweb/sample-350BT/train/0",
            item_loader=TokensLoader(block_size=2048 + 1),
        )
        args.shard_num = 8

        if args.method == "mates":
            args.shard_num = 18
            shard_size = int(1e6)
            # dataset = dataset[: 8 * shard_size]
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
        # selection_size = dataset_size // args.ratio
        selection_size = 5120000
        indices = get_indices(dataset_size, selection_size, args)
        print(f">> Max index: {max(indices)}")

        f = open("o.txt", "w")
        for i in range(5):
            # print(tokenizer.decode(dataset[indices[i]]))
            f.write(tokenizer.decode(torch.tensor(dataset[indices[i]]))[:512])
            f.write("\n---------------\n")

        optimize(
            fn=lambda index: dataset[index],
            inputs=indices,
            output_dir=f"../data/fineweb/sample-350BT/train/0/{args.method}/{selection_size}",
            # output_dir=f"/data/users/zichunyu/data/fineweb/sample-350BT/val/{args.model_name}/{args.method}/{args.ckpt}-ora/{selection_size}",
            # output_dir=f"../manifold/scaling_mates/data/fineweb/sample-350BT/train/0/{args.model_name}/{args.method}/{args.ckpt}/{selection_size}/{args.temp}",
            num_workers=(os.cpu_count() // 8),
            chunk_bytes="200MB",
        )
    else:
        args.output_dir = "/data/users/zichunyu/out/fineweb/sample-10BT/fasttext-oh-eli5-doc-prediction"
        dataset = datasets.load_dataset(
            "HuggingFaceFW/fineweb",
            num_proc=os.cpu_count() // 2,
            name="sample-10BT",
            cache_dir="/data/users/zichunyu/data/hf_cache",
            split="train",
        )

        dataset_size = len(dataset)
        print(f">> Dataset size: {dataset_size}")
        # Hard coding to be fixed
        selection_size = dataset_size // args.ratio
        indices = get_indices(dataset_size, selection_size, args)
        print(f">> Max index: {max(indices)}")

        f = open("o.txt", "w")
        for i in range(5):
            # print(tokenizer.decode(dataset[indices[i]]))
            f.write(dataset[indices[i]]["text"][:512])
            f.write("\n---------------\n")

        def tokenize(data: Dataset, index: int):
            yield tokenizer.encode(data[index]["text"], eos=True)

        optimize(
            fn=partial(tokenize, dataset),
            inputs=indices,
            output_dir=f"/data/users/zichunyu/data/fineweb/sample-10BT/{args.model_name}/{args.method}-doc/{args.ckpt}/{selection_size}",
            num_workers=(os.cpu_count() // 8),
            chunk_bytes="200MB",
        )
