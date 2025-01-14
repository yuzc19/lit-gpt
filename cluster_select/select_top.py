import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict
import os
import pathlib
import random

import datasets
import numpy as np
import torch
import yaml
import os
import numpy as np
from litdata import optimize
from litdata.streaming import StreamingDataset, TokensLoader


def print_top_score_data():
    from collections import Counter

    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less_full/{i}"
            )
            for i in range(32)
        ]
    )
    print(dataset) # 1638400
    metrics = np.array(dataset["scores"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)

    # Get the indices of the top k scores
    k = 102400
    top_k_indices = list(np.argsort(metrics)[-k:][::-1])
    print(top_k_indices[:10])
    # Get the indices of the lower k scores
    low_k_indices = list(np.argsort(metrics)[:k])
    print(low_k_indices[:10])

    # optimize data of selected indices
    pretrain_dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train_decode/0/{i}"
            )
            for i in range(16)
        ]
    )
    print(pretrain_dataset)
    print(">> Top score data:")
    for idx in top_k_indices[:10]:
        print("top", idx, "score:", metrics[idx])
        print(pretrain_dataset[int(idx)]['text'])
    print(">> Lower score data:")
    for idx in low_k_indices[:10]:
        print("low", idx, "score:", metrics[idx])
        print(pretrain_dataset[int(idx)]['text'])




def select_and_optimize_top():
    from collections import Counter

    # get cluster indices (for analysis)
    cluster_indices = {}
    num_clusters = 1000
    base_dir = "/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000/cluster_indices/"
    for cluster_id in range(num_clusters):
        cluster_file = os.path.join(base_dir, f"cluster_{cluster_id}.npy")
        if os.path.exists(cluster_file):
            cluster_i = np.load(cluster_file)
            if cluster_i.size > 0:
                try:
                    indices = cluster_i[:, 1].astype("int32")
                except:
                    indices = cluster_i.astype("int32")
                cluster_indices[cluster_id] = indices
        else:
            print(f"Cluster file {cluster_file} does not exist.")
    # get metrics
    metric_name = "less_full" #"less_full", "oracle"
    metric_split_path = os.path.join("/data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000", metric_name)
    max_splits = len([name for name in os.listdir(metric_split_path) if os.path.isdir(os.path.join(metric_split_path, name))])
    metrics_dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                os.path.join(metric_split_path, str(i))
            )
            for i in range(max_splits)
        ]
    )
    metrics = np.array(metrics_dataset["scores"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)

    # Get the indices of the top k scores
    k = 102400
    top_k_indices = list(np.argsort(metrics)[-k:][::-1])
    print(top_k_indices[:10])

    # Calculate the average metric score and standard deviation of the top k scores
    top_k_scores = metrics[top_k_indices]
    avg_top_k_score = np.mean(top_k_scores)
    std_top_k_score = np.std(top_k_scores)

    # Create a reverse mapping from indices to cluster IDs
    index_to_cluster = {}
    for cluster_id, indices in cluster_indices.items():
        for idx in indices:
            index_to_cluster[idx] = cluster_id
    # Determine how many clusters these top k scores come from
    cluster_counts = Counter(index_to_cluster[idx] for idx in top_k_indices if idx in index_to_cluster)
    num_clusters = len(cluster_counts)

    print(f"Average metric score of top {k} scores: {avg_top_k_score:.4f}")
    print(f"Standard deviation of top {k} scores: {std_top_k_score:.4f}")
    print(f"Number of clusters containing top {k} scores: {num_clusters}")
    
    # optimize data of selected indices
    if True:
        dataset = StreamingDataset(
            input_dir="/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0",
            item_loader=TokensLoader(block_size=2048 + 1),
        )
        dataset_size = len(dataset)
        print(f">> Dataset size: {dataset_size}")
        optimize(
            fn=lambda index: dataset[int(index)],
            inputs=top_k_indices,
            output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_full_grad_sim_top_decay",
            num_workers=4,
            chunk_bytes="200MB",
        )

def select_and_optimize_top_bert():
    from collections import Counter

    # get cluster indices (for analysis)
    cluster_indices = {}
    num_clusters = 1000
    base_dir = "/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000/cluster_indices/"
    for cluster_id in range(num_clusters):
        cluster_file = os.path.join(base_dir, f"cluster_{cluster_id}.npy")
        if os.path.exists(cluster_file):
            cluster_i = np.load(cluster_file)
            if cluster_i.size > 0:
                try:
                    indices = cluster_i[:, 1].astype("int32")
                except:
                    indices = cluster_i.astype("int32")
                cluster_indices[cluster_id] = indices
        else:
            print(f"Cluster file {cluster_file} does not exist.")
    # get metrics
    metric_name = "10000-data_influence_model-flan-prediction" #"less_full", "oracle"
    metric_split_path = os.path.join("/data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/train/0", metric_name)
    max_splits = len([name for name in os.listdir(metric_split_path) if os.path.isdir(os.path.join(metric_split_path, name))])
    metrics_dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                os.path.join(metric_split_path, str(i))
            )
            for i in range(max_splits)
        ]
    )
    print(metrics_dataset)
    metrics = np.array(metrics_dataset["prediction"]).reshape(-1)
    metrics = metrics[:1638400]
    print(">> Metrics shape:", metrics.shape)

    # Get the indices of the top k scores
    k = 102400
    top_k_indices = list(np.argsort(metrics)[-k:][::-1])
    print(top_k_indices[:10])

    # Calculate the average metric score and standard deviation of the top k scores
    top_k_scores = metrics[top_k_indices]
    avg_top_k_score = np.mean(top_k_scores)
    std_top_k_score = np.std(top_k_scores)

    # Create a reverse mapping from indices to cluster IDs
    index_to_cluster = {}
    for cluster_id, indices in cluster_indices.items():
        for idx in indices:
            index_to_cluster[idx] = cluster_id
    # Determine how many clusters these top k scores come from
    cluster_counts = Counter(index_to_cluster[idx] for idx in top_k_indices if idx in index_to_cluster)
    num_clusters = len(cluster_counts)

    print(f"Average metric score of top {k} scores: {avg_top_k_score:.4f}")
    print(f"Standard deviation of top {k} scores: {std_top_k_score:.4f}")
    print(f"Number of clusters containing top {k} scores: {num_clusters}")
    
    # optimize data of selected indices
    if True:
        dataset = StreamingDataset(
            input_dir="/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0",
            item_loader=TokensLoader(block_size=2048 + 1),
        )
        dataset_size = len(dataset)
        print(f">> Dataset size: {dataset_size}")
        optimize(
            fn=lambda index: dataset[int(index)],
            inputs=top_k_indices,
            output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_bert_top_decay",
            num_workers=4,
            chunk_bytes="200MB",
        )


select_and_optimize_top_bert()


