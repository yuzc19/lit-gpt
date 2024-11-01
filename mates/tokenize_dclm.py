import argparse
import os

import torch
from datasets import Dataset
from file_utils import list_dir, read_jsonl
from litdata.streaming import StreamingDataset, TokensLoader
from modeling_data_influence_model import BertForSequenceClassification
from tqdm import tqdm
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
    parser.add_argument("--base_dir", type=str, default="/data/users/zichunyu")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    num_proc = 8
    # data_dir = "s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_1_of_10"
    data_dir = f"{args.base_dir}/tmp/output/refinedweb_01_1/fasttext/fasttext_filter/processed_data/"
    file_list = list_dir(data_dir)

    # dataset = dataset[
    #     args.base
    #     + args.shard[0]
    #     * shard_size : (
    #         args.base·+ (args.shard[0] + 1) * shard_size
    #         if args.shard[0] + 1 < args.shard[1]
    #         else len(dataset)
    #     )
    # ]

    model_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-100BT/{args.ckpt}-data_influence_model-flan"
    output_dir = f"{args.base_dir}/out/{args.model_name}/DCLM-refinedweb_01_1/{args.ckpt}-data_influence_model-flan-prediction"

    # shard_size = int(1e6)
    # shard_size = len(dataset) // args.shard[1]

    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints/bert-base-uncased",
        max_length=2048,
        padding="max_length",
    )

    for file in tqdm(file_list[5:]):
        shard_name = file.split("/")[-1].split(".")[0]
        # output_dir = f"{args.base_dir}/data/DCLM-refinedweb/global-shard_01_of_10/local-shard_1_of_10/{shard_name.replace('processed', 'tokenized')}"
        output_dir = f"{args.base_dir}/manifold/scaling_mates/dclm/output/refinedweb_01_1/fasttext/fasttext_filter/processed_data/{shard_name.replace('processed', 'bert_tokenized')}"

        dataset = []
        for json_line in read_jsonl(file):
            dataset.append(" ".join(json_line["text"].strip().splitlines()))
        dataset = Dataset.from_list([{"text": d} for d in dataset])
        print("Total number of examples:", len(dataset))

        def preprocess_data(examples):
            texts = examples["text"]
            encoding = tokenizer.batch_encode_plus(
                texts,
                max_length=2048,
                padding="max_length",
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

        dataset.save_to_disk(output_dir)

    # dataset = dataset.map(
    #     ModelAnnotator(model_dir, args.device_batch_size),
    #     batched=True,
    #     with_indices=True,
    #     batch_size=args.device_batch_size,
    #     remove_columns=dataset.column_names,
    # )
    # print("After annotation: Total number of examples:", len(dataset))

    # print(f"Saving to {output_dir}")
    # dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
