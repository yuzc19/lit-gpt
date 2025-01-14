import argparse
import os

import torch
from datasets import Dataset,Features, Sequence, Value, load_from_disk
from litdata.streaming import StreamingDataset, TokensLoader
from modeling_data_influence_model import BertForSequenceClassification
from transformers import AutoTokenizer


class ModelAnnotator:
    def __init__(self, model_name, device_batch_size):
        self.model_name = model_name
        self.device_batch_size = device_batch_size

        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
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
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    model_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-350BT/{args.ckpt}-data_influence_model-flan"
    output_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-350BT/train/{args.split}/{args.ckpt}-data_influence_model-flan-prediction"

    num_proc = 8

    dataset = load_from_disk(f"{args.base_dir}/data/fineweb/sample-350BT/train_decode/{args.split}/{args.shard[0]}")
    print(dataset)
    print("Processed dataset loaded from disk")

    print("Total number of examples:", len(dataset))

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained("checkpoints/EleutherAI/pythia-1b")
    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints/bert-base-uncased",
        max_length=2048,
        padding="max_length",
    )

    def preprocess_data(examples):
        encoding = tokenizer.batch_encode_plus(
            examples["text"],
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        return encoding

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    print("After tokenization: Total number of examples:", len(dataset))

    dataset = dataset.map(
        ModelAnnotator(model_dir, args.device_batch_size),
        batched=True,
        with_indices=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
