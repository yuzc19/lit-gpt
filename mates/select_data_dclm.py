import argparse
import concurrent
import os
from multiprocessing import Pool
from pathlib import Path

import datasets
import numpy as np
from datasets import Dataset
from file_utils import read_jsonl, write_jsonl
from tqdm import tqdm


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


def process_jsonl(file_dir):
    return [d for d in read_jsonl(file_dir)]


def load_dataset(data_dir, shard_names, max_workers=None):
    dataset = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Using executor.map ensures order is preserved
        shards = list(
            tqdm(
                executor.map(process_jsonl, [data_dir.format(n) for n in shard_names]),
                total=len(shard_names),
            )
        )
        for shard in shards:
            dataset.extend(shard)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--shard_num", type=float, default=18)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    args.output_dir = f"../manifold/scaling_mates/out/{args.model_name}/refinedweb_01_0/fasttext/fasttext_filter/{args.ckpt}-data_influence_model-prediction"

    data_dir = f"../manifold/scaling_mates/dclm/output/refinedweb_01_0/fasttext/fasttext_filter/processed_data"
    file_list = [
        os.path.abspath(os.path.join(data_dir, f))
        for f in os.listdir(data_dir)
        if not f.startswith(".")
    ]
    shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
    shard_size = len(file_list) // args.shard_num
    data_dir = "../tmp/output/refinedweb_01_0/fasttext/fasttext_filter/processed_data/{}_processed.jsonl.zstd"

    out_dir = Path(
        f"../tmp/dclm/output/refinedweb_01_0/fasttext/fasttext_filter/2/10000-data_influence_model/processed_data",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # def count_examples(file):
    #     cnt = 0
    #     for json_line in read_jsonl(file):
    #         cnt += 1
    #     return cnt

    # for i in range(18):
    #     total_examples_num = sum(
    #         [
    #             count_examples(data_dir.format(shard_names[j]))
    #             for j in range(
    #                 i * shard_size,
    #                 ((i + 1) * shard_size if i + 1 < 18 else len(shard_names)),
    #             )
    #         ]
    #     )
    #     store_examples_num = len(datasets.load_from_disk(f"{args.output_dir}/{i}"))
    #     print(i, total_examples_num, store_examples_num)

    # dataset = datasets.load_dataset(
    #     "json",
    #     data_files=[data_dir.format(n) for n in tqdm(shard_names[100:])],
    #     # split="train",
    #     # field="text",
    #     # cache_dir="/data/users/zichunyu/data/hf_cache",
    #     features=features,
    # )

    # for n in tqdm(shard_names[:20]):
    #     file_dir = data_dir.format(n)
    #     shard = [d for d in read_jsonl(file_dir)]
    #     print(len(shard))

    dataset = load_dataset(data_dir, shard_names, max_workers=16)
    dataset_size = len(dataset)
    # dataset = Dataset.from_list(dataset)
    print(f">> Dataset size: {dataset_size}")
    selection_size = dataset_size // args.ratio
    indices = get_indices(dataset_size, selection_size, args)
    print(f">> Max index: {max(indices)}")

    # indices = [i for i in indices if i < dataset_size]
    # dataset = dataset.select(indices)
    with Pool(128) as pool:
        # Create a list of arguments for each process
        process_args = [
            (
                # dataset.shard(8, i, contiguous=True),
                [dataset[i] for i in indices[i::128]],
                f"../tmp/dclm/output/refinedweb_01_0/fasttext/fasttext_filter/2/10000-data_influence_model/processed_data/{i}_processed.jsonl.zstd",
            )
            for i in range(128)
        ]

        pool.starmap(write_jsonl, process_args)
        pool.close()
        pool.join()
