import argparse
import os
from dataclasses import dataclass
from re import S

import datasets
import numpy as np
import torch
from datasets import Features, Sequence, Value
from litdata.streaming import StreamingDataset, TokensLoader
from modeling_data_influence_model import BertForSequenceClassification
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, Trainer, TrainingArguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-1b", required=False)
    parser.add_argument("--ckpt", type=int, default=10000, required=False)

    args = parser.parse_args()
    print(args)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="regression",
        num_labels=1,
    )

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

    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        max_length=2048,
        padding="max_length",
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
    mean_value = np.mean([s[0] - s[1] for s in np.array(dataset["scores"])])
    std_value = np.std([s[0] - s[1] for s in np.array(dataset["scores"])])
    print(mean_value, std_value)

    def preprocess_data(examples):
        queries = [data_pool[index] for index in examples["probe_index"]]
        queries = pythia_tokenizer.batch_decode(queries, skip_special_tokens=True)
        enc = tokenizer.batch_encode_plus(
            queries,
            max_length=2048,
            padding="max_length",
            truncation=True,
        )
        # Convert the labels to float for regression
        scores = examples["scores"]
        enc["labels"] = [(s[0] - s[1] - mean_value) / std_value for s in scores]
        return enc

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

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions[:, 0]
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
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
