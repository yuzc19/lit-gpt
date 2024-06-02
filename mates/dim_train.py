from transformers import AutoModelForSequenceClassification
from gte_bert import BertForSequenceClassification
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from datasets import load_dataset
import numpy as np
import datasets
import torch
import os

DATASET_SIZE = 40000


def load_datasets_dsdm():
    data_files = [f"data/train-{str(i).zfill(5)}-of-00891*" for i in range(100)]
    dataset = datasets.load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=os.cpu_count() // 8,
        data_files=data_files,
        cache_dir="/data/datasets/hf_cache",
        verification_mode="no_checks",
    )["train"]

    rng = np.random.default_rng()
    indices = rng.choice(len(dataset), size=(DATASET_SIZE,), replace=False)
    eval_size = DATASET_SIZE // 10

    dsdm_scores = (
        torch.load("/data/user_data/zichunyu/out/DsDm/data/dsdm/squad.pt")
        + torch.load("/data/user_data/zichunyu/out/DsDm/data/dsdm/lambada.pt")
        + torch.load("/data/user_data/zichunyu/out/DsDm/data/dsdm/jeopardy.pt")
    ) / 3
    # dsdm_scores = torch.load("/data/user_data/zichunyu/out/DsDm/data/dsdm/lambada.pt")
    # dsdm_scores = (dsdm_scores - dsdm_scores.mean()) / dsdm_scores.std()
    # dsdm_scores /= 2.0

    train_dataset = dataset.select(indices[eval_size:])
    eval_dataset = dataset.select(indices[:eval_size])
    print(dsdm_scores.mean(), dsdm_scores.std())
    train_dataset = train_dataset.add_column(
        "scores", dsdm_scores[indices[eval_size:]].numpy().tolist()
    )
    train_dataset = train_dataset.rename_column("input_ids", "ori_input_ids")
    eval_dataset = eval_dataset.add_column(
        "scores", dsdm_scores[indices[:eval_size]].numpy().tolist()
    )
    eval_dataset = eval_dataset.rename_column("input_ids", "ori_input_ids")
    return train_dataset, eval_dataset


def load_datasets_tar():
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-1b-baseline/c4_trial_and_error_lambada_openai_080000/{i}"
            )
            for i in range(8)
        ]
    )

    dataset = dataset.train_test_split(test_size=0.1, seed=1234, shuffle=True)
    train_dataset = dataset["train"].rename_column("input_ids", "ori_input_ids")
    print(len(train_dataset))
    # train_dataset = train_dataset.select(range(DATASET_SIZE))
    eval_dataset = dataset["test"].rename_column("input_ids", "ori_input_ids")
    return train_dataset, eval_dataset


train_dataset, eval_dataset = load_datasets_tar()
mean_value = np.mean(np.array(train_dataset["scores"])[:, 0])
# lambada, arce, squad, sciq
# 2.758474349975586, 4.679286003112793, 3.8528645038604736, 5.960725784301758
# mean_value = 5.960725784301758
std_value = np.std(np.array(train_dataset["scores"])[:, 0])
# 4.994476930430366 0.8155154451946361
print(np.array(train_dataset["scores"])[:, 0].shape, mean_value, std_value)

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


# Preprocess data function
def preprocess_data(examples):
    # Encode sentence pairs
    texts = pythia_tokenizer.batch_decode(
        examples["ori_input_ids"], skip_special_tokens=True
    )
    encoding = tokenizer.batch_encode_plus(
        texts,
        max_length=1024,
        padding="max_length",
        truncation=True,
    )
    # Convert the labels to float for regression
    encoding["labels"] = [
        (float(score[0]) - mean_value) / std_value for score in examples["scores"]
    ]
    return encoding


# Process and encode the datasets
train_dataset = train_dataset.map(
    preprocess_data,
    batched=True,
    num_proc=os.cpu_count() // 8,
    remove_columns=["ori_input_ids", "scores"],
)
eval_dataset = eval_dataset.map(
    preprocess_data,
    batched=True,
    num_proc=os.cpu_count() // 8,
    remove_columns=["ori_input_ids", "scores"],
)
# train_dataset = train_dataset.select([i for i in range(80000)])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# Load model for sequence classification with a regression head
model = BertForSequenceClassification.from_pretrained(
    "/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-410m-baseline/iter-040000-ckpt/bert-base-finetuned-lambada_openai",
    # "prajjwal1/bert-medium",
    cache_dir="/data/datasets/hf_cache",
    problem_type="regression",
    num_labels=1,
)

# Freeze all parameters in the model
# for param in model.bert.parameters():
#     param.requires_grad = False

# Optionally, ensure the classifier head is unfrozen
# for param in model.classifier.parameters():
#     param.requires_grad = True

# Training arguments
batch_size = 16

# 5e-5 or 6e-5
args = TrainingArguments(
    f"/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-1b-baseline/iter-080000-ckpt/bert-base-finetuned-lambada_openai",
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    logging_steps=10,
    eval_steps=10,
    save_steps=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="spearman",
    bf16=True,
)


# Define regression metrics
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


# Initialize Trainer
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
trainer.save_model()

# Evaluate the best model
eval_results = trainer.evaluate()

# Print the evaluation results
print("Best evaluation results:", eval_results)
