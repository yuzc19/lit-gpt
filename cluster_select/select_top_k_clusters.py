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

def get_cluster_indices(assign_file: str, output_dir: str) -> Dict[int, np.ndarray]:
    """
    Load the assignment file and return a dictionary where the keys are cluster IDs
    and the values are numpy arrays of indices for each cluster. If the cluster indices
    are already saved, load them from the files.

    Args:
        assign_file (str): Path to the assignment.pkl file.
        output_dir (str): Directory to save/load the cluster indices.

    Returns:
        Dict[int, np.ndarray]: Dictionary with cluster IDs as keys and numpy arrays of indices as values.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the cluster indices are already saved
    cluster_indices = {}
    for cluster_file in output_dir.glob('cluster_*.npy'):
        cluster_id = int(cluster_file.stem.split('_')[1])
        cluster_indices[cluster_id] = np.load(cluster_file)

    if cluster_indices:
        print(f"Loaded cluster indices from {output_dir}")
        return cluster_indices

    # Load the assignment.pkl file
    with open(assign_file, 'rb') as f:
        assignments = pickle.load(f)

    # Ensure assignments is a numpy array
    assignments = np.array(assignments)

    # Create a dictionary to store the indices for each cluster
    cluster_indices = defaultdict(list)

    # Iterate through the assignments and collect indices for each cluster
    for idx, cluster in enumerate(assignments.flatten()):
        cluster_indices[cluster].append(idx)

    # Convert lists to numpy arrays and save them
    cluster_indices = {cluster: np.array(indices) for cluster, indices in cluster_indices.items()}
    for cluster, indices in cluster_indices.items():
        np.save(output_dir / f'cluster_{cluster}.npy', indices)

    print(f"Saved cluster indices to {output_dir}")
    return cluster_indices


def collect_gradient_similarity_reward(num_clusters, cluster_indices):
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less/{i}"
            )
            for i in range(8)
        ]
    )
    print(dataset) # 409600
    metrics = np.array(dataset["scores"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)

    cluster_average_reward = np.zeros(num_clusters)
    for cluster_id in range(num_clusters):
        indices = cluster_indices[cluster_id].astype("int32")
        indices = indices[indices < len(metrics)]
        cluster_average_reward[cluster_id] = np.mean(metrics[indices])
    np.save(pathlib.Path(base_dir, "average_reward.npy"), cluster_average_reward)
    return cluster_average_reward

def sample_from_top_k_clusters(cluster_average_reward, selected_data_size, sample_rate, data_size):
    # sort clusters by average reward
    cluster_ids = np.argsort(-cluster_average_reward)
    
    # for each cluster in order, sample sample_rate% of the data, until desired_data_size is reached
    sampled_indices = []
    k = 0
    for cluster_id in cluster_ids:
        k += 1
        indices = cluster_indices[cluster_id]
        indices = indices[indices < data_size]  # only consider indices less than data_size
        num_samples = int(sample_rate * len(indices))
        sampled_indices.extend(random.sample(list(indices), num_samples))
        if len(sampled_indices) >= selected_data_size:
            break
    return sampled_indices, k

base_dir = '/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000'
assign_file = '/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000/assignment.pkl'
cluster_indices_dir = '/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000/cluster_indices'
cluster_indices = get_cluster_indices(assign_file, cluster_indices_dir)

def select_and_optimize_v1():
    # select top k clusters, sample from each cluster, based on gradient similarity reward
    cluster_indices = get_cluster_indices(assign_file, cluster_indices_dir)

    cluster_average_reward = collect_gradient_similarity_reward(1000, cluster_indices)
    print(cluster_average_reward)

    selected_indices, k = sample_from_top_k_clusters(cluster_average_reward, 102400, 0.7, 409600)
    print(len(selected_indices), k, selected_indices[:10]) # 102516 364 [76311, 189451, 371479, 100164, 73395, 100165, 291825, 303169, 368284, 179273]
    # sort selected_indices
    selected_indices.sort()

    # Load metrics
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less/{i}"
            )
            for i in range(8)
        ]
    )
    print(dataset) # 409600
    metrics = np.array(dataset["scores"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)

    # Calculate the average metric score and standard deviation of the selected indices
    selected_scores = metrics[selected_indices]
    avg_selected_score = np.mean(selected_scores)
    std_selected_score = np.std(selected_scores)

    print(f"Average metric score of selected indices: {avg_selected_score:.4f}")
    print(f"Standard deviation of selected indices: {std_selected_score:.4f}")


    # optimize data of selected indices
    if False:
        dataset = StreamingDataset(
            input_dir="/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0",
            item_loader=TokensLoader(block_size=2048 + 1),
        )
        dataset_size = len(dataset)
        print(f">> Dataset size: {dataset_size}")
        optimize(
            fn=lambda index: dataset[int(index)],
            inputs=selected_indices,
            output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_grad_sim_cluster_decay",
            num_workers=4,
            chunk_bytes="200MB",
        )

def select_and_optimize_v2():
    from collections import Counter

    # select top k indices based on gradient similarity score
    cluster_indices = get_cluster_indices(assign_file, cluster_indices_dir)

    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(
                f"/data/datasets/hf_cache/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/less/{i}"
            )
            for i in range(8)
        ]
    )
    print(dataset) # 409600
    metrics = np.array(dataset["scores"]).reshape(-1)
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
    if False:
        dataset = StreamingDataset(
            input_dir="/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0",
            item_loader=TokensLoader(block_size=2048 + 1),
        )
        dataset_size = len(dataset)
        print(f">> Dataset size: {dataset_size}")
        optimize(
            fn=lambda index: dataset[int(index)],
            inputs=top_k_indices,
            output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_grad_sim_top_decay",
            num_workers=4,
            chunk_bytes="200MB",
        )

select_and_optimize_v1()

