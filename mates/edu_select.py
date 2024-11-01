import argparse
import os

import torch
from datasets import Dataset
from litdata.streaming import StreamingDataset, TokensLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelAnnotator:
    def __init__(self, model_name, device_batch_size):
        self.model_name = model_name
        self.device_batch_size = device_batch_size

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # cache_dir="../manifold/scaling_mates/data/hf_cache",
        )
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")
        self.model.to(self.device)

    def __getstate__(self):
        return {
            "model_name": self.model_name,
            "device_batch_size": self.device_batch_size,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    @torch.no_grad()
    def __call__(self, example, indices):
        output = {"index": indices}

        outputs = self.model(
            torch.tensor(example["input_ids"], device=self.device),
            attention_mask=torch.tensor(example["attention_mask"], device=self.device),
            token_type_ids=torch.tensor(example["token_type_ids"], device=self.device),
        )
        output["prediction"] = outputs.logits.detach().float().cpu().numpy()

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/users/zichunyu")
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    num_proc = 8
    # 50M examples in total, 100B Tokens
    dataset = StreamingDataset(
        input_dir=f"{args.base_dir}/data/fineweb/sample-350BT/train/1",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    # 3M examples/GPU for 10k steps
    shard_size = int(1e6)
    # shard_size = len(dataset) // args.shard[1]
    dataset = dataset[
        args.base
        + args.shard[0]
        * shard_size : (
            args.base + (args.shard[0] + 1) * shard_size
            if args.shard[0] + 1 < args.shard[1]
            else len(dataset)
        )
    ]
    dataset = Dataset.from_list([{"ori_input_ids": d[:2048]} for d in dataset])

    print("Total number of examples:", len(dataset))

    # Load tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("checkpoints/EleutherAI/pythia-1b")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/fineweb-edu-classifier",
        # cache_dir="../manifold/scaling_mates/data/hf_cache",
    )

    def preprocess_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["ori_input_ids"],
            skip_special_tokens=True,
        )
        encoding = tokenizer.batch_encode_plus(
            texts,
            padding="longest",
            truncation=True,
        )
        return encoding

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    print("After tokenization: Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator("HuggingFaceTB/fineweb-edu-classifier", args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    output_dir = (
        f"{args.base_dir}/out/fineweb/sample-350BT/train/1/fineweb-edu-prediction"
    )
    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
