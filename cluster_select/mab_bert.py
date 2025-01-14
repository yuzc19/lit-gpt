import os
import pathlib
import random

import datasets
import numpy as np
import torch
import yaml


def get_cluster_indices(num_clusters=1000):
    cluster_indices = {}
    
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
    
    return cluster_indices
    

def collect_reward(num_clusters, metric_name):
    cluster_indices = get_cluster_indices(num_clusters)
    cluster_average_reward = np.zeros(num_clusters)
    
    for cluster_id, indices in cluster_indices.items():
        valid_indices = indices[indices < len(metrics)]
        if len(valid_indices) > 0:
            cluster_average_reward[cluster_id] = np.mean(metrics[valid_indices])
        else:
            cluster_average_reward[cluster_id] = 0
    print("cluster_average_reward:", cluster_average_reward)
    np.save(pathlib.Path(base_dir, f"average_reward_{metric_name}.npy"), cluster_average_reward)

def mab(
    num_clusters,
    cluster_average_reward,
):
    sum_chose = 0
    cluster_chose_ratio = np.zeros(num_clusters)
    cluster_chose_time = np.zeros(num_clusters)
    cluster_ucb = cluster_average_reward.copy()
    for _ in range(1000): # 1400 -> 0.28 selection ratio
        # 0.02 * 0.05 * 200 = 0.2
        # 1000 -> 0.2 selection ratio
        current_chose_num = 0
        current_chose = []
        for k in np.argsort(-cluster_ucb):
            if cluster_chose_ratio[k] < 1:
                cluster_chose_ratio[k] += 0.02
                cluster_chose_time[k] += 1
                current_chose_num += 1
                current_chose.append(k)
                if current_chose_num == batch:
                    break
        sum_chose += batch
        for k in range(num_clusters):
            ucb = alpha * np.sqrt(
                2 * (np.log(float(sum_chose))) / float(cluster_chose_time[k] + 1)
            )
            cluster_ucb[k] = cluster_average_reward[k] + ucb
    return cluster_chose_ratio


if __name__ == "__main__":
    random.seed(42)
    num_clusters = 1000
    alpha = 0.002 #0.0001
    batch = 10

    # base_dir = "/data/datasets/hf_cache/data/fineweb/sample-350BT/bge_micro_cluster_info/0/sorted_clusters/"
    base_dir = "/data/datasets/hf_cache/data/fineweb/sample-350BT/train_bge_micro_embeddings_index_1000/cluster_indices/"
    #get metrics
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

    collect_reward(num_clusters, metric_name=metric_name)

    cluster_indices = get_cluster_indices(num_clusters)
    print("Cluster indices:", {k: len(v) for k, v in cluster_indices.items()})
    cluster_chose_ratio = mab(num_clusters, np.load(pathlib.Path(base_dir, f"average_reward_{metric_name}.npy")))
    print("Cluster chose ratio:", cluster_chose_ratio)
    np.save(f"out/{metric_name}/cluster_chose_ratio.npy", cluster_chose_ratio) # TODO fix dir

    selected_indices = []
    for cluster_id in range(num_clusters):
        if cluster_id not in cluster_indices:
            continue
        indices = cluster_indices[cluster_id]
        indices = indices[indices < len(metrics)]  # Only consider indices within len(metrics)
        random.shuffle(indices)
        selected_indices += indices[
            : int(cluster_chose_ratio[cluster_id] * len(indices))
        ].tolist()
    print(">> Selected indices shape:", len(selected_indices))
    np.save(f"out/{metric_name}/selected_indices.npy", selected_indices) # TODO fix dir


    #optimize
    if True:
        from litdata.streaming import StreamingDataset, TokensLoader
        from litdata import optimize

        dataset = StreamingDataset(
            input_dir="/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0",
            item_loader=TokensLoader(block_size=2048 + 1),
        )
        dataset_size = len(dataset)
        print(f">> Dataset size: {dataset_size}")
        optimize(
            fn=lambda index: dataset[int(index)],
            inputs=selected_indices,
            output_dir=f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0/select_bert_mab_decay",
            num_workers=4,
            chunk_bytes="200MB",
        )
