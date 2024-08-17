import os

import datasets
import fasttext
import torch
from datasets import Dataset
from litdata.streaming import StreamingDataset, TokensLoader
from transformers import AutoTokenizer

use_seq = False

if use_seq:
    dataset = StreamingDataset(
        input_dir="/data/users/zichunyu/data/fineweb/sample-100BT/train",
        item_loader=TokensLoader(block_size=2048 + 1),
        drop_last=True,
    )
    dataset = dataset[: int(2e6)]
    output_dir = (
        "/data/users/zichunyu/out/fineweb/sample-100BT/fasttext-oh-eli5-prediction"
    )
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        dataset[:5] + dataset[-5:],
        output_dir + "/sanity_check.pt",
    )
    dataset = Dataset.from_list([{"ori_input_ids": d[:2048]} for d in dataset])

    print("Total number of examples:", len(dataset))

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("checkpoints/EleutherAI/pythia-1b")
    # Finally, we observe that using a fairly strict threshold, which keeps the top-10% of examples, helps
    # over more permissive top-15% and top-20% thresholds. Larger -> Better
    model = fasttext.load_model(
        "/data/users/zichunyu/out/fasttext-oh-eli5/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
    )

    def process_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["ori_input_ids"],
            skip_special_tokens=True,
        )
        texts = [t.replace("\n", " ") for t in texts]
        pred = []
        for l, s in zip(*model.predict(texts, k=-1)):
            pred_map = {}
            pred_map[l[0]] = s[0]
            pred_map[l[1]] = s[1]
            pred.append(pred_map["__label__hq"])
        return {"prediction": pred}

else:
    dataset = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        num_proc=os.cpu_count() // 2,
        name="sample-10BT",
        cache_dir="/data/users/zichunyu/data/hf_cache",
        split="train",
    )
    output_dir = (
        "/data/users/zichunyu/out/fineweb/sample-10BT/fasttext-oh-eli5-doc-prediction"
    )

    print("Total number of examples:", len(dataset))

    model = fasttext.load_model(
        "/data/users/zichunyu/out/fasttext-oh-eli5/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
    )

    def process_data(examples):
        texts = [t.replace("\n", " ") for t in examples["text"]]
        pred = []
        for l, s in zip(*model.predict(texts, k=-1)):
            pred_map = {}
            pred_map[l[0]] = s[0]
            pred_map[l[1]] = s[1]
            pred.append(pred_map["__label__hq"])
        return {"prediction": pred}


dataset = dataset.map(
    process_data,
    batched=True,
    batch_size=1024,
    num_proc=8,
    remove_columns=dataset.column_names,
)
print("After scoring: Total number of examples:", len(dataset))

print(f"Saving to {output_dir}")
dataset.save_to_disk(output_dir)
