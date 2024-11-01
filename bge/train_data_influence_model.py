import argparse
import os
from dataclasses import dataclass
from re import S

import datasets
import numpy as np
import torch
from datasets import Features, Sequence, Value
from litdata.streaming import StreamingDataset, TokensLoader
from modeling_seq_data_influence_model import BiEncoderModel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    def __call__(self, features):
        query = [f["query"] for f in features]
        passage = [f["passages"] for f in features]
        score = [f["score"] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer.batch_encode_plus(
            query,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # d_collated = self.tokenizer.batch_encode_plus(
        #     passage,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # )
        return {
            "query": q_collated,
            # "passage": d_collated,
            "label": torch.tensor(score, dtype=torch.float32),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b", required=False)
    parser.add_argument("--ckpt", type=int, default=10000, required=False)

    args = parser.parse_args()
    print(args)

    model = BiEncoderModel(temperature=1)

    args = TrainingArguments(
        "test",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        # gradient_accumulation_steps=8,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        warmup_ratio=0.03,
        logging_steps=5,
        eval_steps=50,
        save_steps=10000,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # metric_for_best_model="spearman",
        bf16=True,
        report_to="wandb",
        run_name="dependent-data-influence-model_bge-base_dot-product",
        remove_unused_columns=False,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # train_dataset = TrainDatasetForEmbedding(tokenizer=tokenizer)

    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    data_pool = StreamingDataset(
        input_dir="/data/users/zichunyu/data/fineweb/sample-350BT/val",
        item_loader=TokensLoader(block_size=2048 + 1),
    )
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(f"../out-litgpt/step-10000-comb-16/pythia-1b/{i}")
            for i in range(16)
        ]
    )
    mean_value = np.mean([s[1] for s in np.array(dataset["scores"])])
    std_value = np.std([s[1] for s in np.array(dataset["scores"])])
    print(mean_value, std_value)

    def preprocess_data(examples):
        queries = [data_pool[index] for index in examples["probe_index"]]
        passages = [
            [data_pool[index] for index in passage_indices]
            for passage_indices in examples["train_indices"]
        ]
        queries = pythia_tokenizer.batch_decode(queries, skip_special_tokens=True)
        # queries = tokenizer.batch_encode_plus(
        #     queries,
        #     padding="max_length",
        #     truncation=True,
        # )
        passages = [
            pythia_tokenizer.batch_decode(passage, skip_special_tokens=True)
            for passage in passages
        ]
        # passages = [
        #     tokenizer.batch_encode_plus(
        #         passage,
        #         padding="max_length",
        #         truncation=True,
        #     )
        #     for passage in passages
        # ]
        scores = [(s[1] - mean_value) / std_value for s in examples["scores"]]

        examples["query"] = queries
        examples["passages"] = passages
        examples["score"] = scores

        return {"query": queries, "passages": passages, "score": scores}

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(test_size=0.1, seed=1234, shuffle=True)
    train_dataset = dataset["train"]
    print("Training data size:", len(train_dataset))
    eval_dataset = dataset["test"]

    data_collator = EmbedCollator(tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions[0]
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    # trainer.save_model()

    # predictions = trainer.predict(train_dataset.select(range(1000)))
    # predicted_scores = predictions.predictions[0]
    # print(spearmanr(predicted_scores, train_dataset["score"][:1000]))
    # Evaluate the best model
    # eval_results = trainer.evaluate()

    # Print the evaluation results
    # print("Best evaluation results:", eval_results)
