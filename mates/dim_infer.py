from datasets import load_from_disk, load_dataset, concatenate_datasets
from gte.gte_bert import BertForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import datasets
import argparse
import torch
import os

# Load tokenizer
tokenizer_kwargs = {"cache_dir": "/data/user_data/zichunyu/.cache/doremi"}
pythia_tokenizer = AutoTokenizer.from_pretrained(
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
    **tokenizer_kwargs,
)
pythia_tokenizer.model_max_length = 1024
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    max_length=1024,
    padding="max_length",
    cache_dir="/data/datasets/hf_cache",
)


class ModelAnnotator:
    def __init__(self, model_name, device_batch_size):
        self.model_name = model_name
        self.device_batch_size = device_batch_size

        self.model = BertForSequenceClassification.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            cache_dir="/data/datasets/hf_cache",
            problem_type="regression",
            num_labels=1,
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
    parser.add_argument(
        "--output",
        type=str,
        default="/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-1b-baseline/iter-080000-ckpt/0-100/bert-base_lambada_openai_annotate",
    )

    parser.add_argument("-F", "--data_files", type=str, nargs="+", default=[])
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        default="/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-1b-baseline/iter-080000-ckpt/bert-base-finetuned-lambada_openai",
    )
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)
    parser.add_argument("-w", "--num_workers", type=int, default=1)

    args = parser.parse_args()
    print(args)

    num_proc = os.cpu_count() // 16
    dataset = datasets.load_from_disk(
        "/data/datasets/hf_cache/c4-pythia/bert-dsdm-candidate-c4/0-100"
        + f"/{args.shard[0]}"
    )
    print("After tokenization: Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator(args.model, args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print(
        "After annotation: Total number of examples:", len(dataset)
    )  # 5hours in A6000

    print(f"Saving to {args.output}")
    dataset.save_to_disk(args.output + f"/{args.shard[0]}")
