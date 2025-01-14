import argparse
import os

import datasets
import numpy as np
from modeling_data_influence_model import BertForSequenceClassification
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, Trainer, TrainingArguments

oracle_dirs = [
    "/data/datasets/hf_cache/out/pythia-1b/step143000/fineweb/final/oracle",
    "/data/datasets/hf_cache/out/pythia-1b/step50000/fineweb/final/oracle",
]

def load_oracle_dataset(oracle_dir):
    try:
        dataset = datasets.concatenate_datasets(
            [datasets.load_from_disk(f"{oracle_dir}/{i}") for i in sorted(os.listdir(oracle_dir), key=int)]
        )
    except:
        dataset = datasets.load_from_disk(f"{oracle_dir}/0")
    return dataset

# for oracle_dir in oracle_dirs:
#     try:
#         dataset = datasets.concatenate_datasets(
#             [datasets.load_from_disk(f"{oracle_dir}/{i}") for i in range(8)]
#         )
#     except:
#         dataset = datasets.load_from_disk(f"{oracle_dir}/0")
#     mean_value = np.mean(np.array(dataset["scores"])[:, 0])
#     std_value = np.std(np.array(dataset["scores"])[:, 0])
#     print(oracle_dir)
#     print(np.array(dataset["scores"])[:, 0].shape, mean_value, std_value)

# Load datasets from both oracle directories
dataset1 = load_oracle_dataset(oracle_dirs[0])
dataset2 = load_oracle_dataset(oracle_dirs[1])

# Extract scores from both datasets
scores1 = np.array(dataset1["scores"])[:, 0]
scores2 = np.array(dataset2["scores"])[:, 0]

# print(dataset1["input_ids"][296])
# print(dataset2["input_ids"][296])
# print("EQL",dataset1["input_ids"][0] == dataset2["input_ids"][0])

# Calculate mean and standard deviation for both datasets
mean_value1 = np.mean(scores1)
std_value1 = np.std(scores1)
mean_value2 = np.mean(scores2)
std_value2 = np.std(scores2)

# Print mean and standard deviation
print(oracle_dirs[0])
print(scores1.shape, mean_value1, std_value1)
print(oracle_dirs[1])
print(scores2.shape, mean_value2, std_value2)

# Calculate Spearman rank correlation coefficient
spearman_corr, _ = spearmanr(scores1, scores2)
print(f"Spearman rank correlation coefficient: {spearman_corr}")


