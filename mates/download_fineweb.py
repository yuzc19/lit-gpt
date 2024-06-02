import datasets
import os

datasets.load_dataset(
    "HuggingFaceFW/fineweb",
    num_proc=os.cpu_count() // 2,
    name="sample-10BT",
    cache_dir="/data/users/zichunyu/data/hf_cache",
    split="train",
)
