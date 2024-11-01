import os

import numpy as np
from litdata import optimize
from litdata.streaming import CombinedStreamingDataset, StreamingDataset, TokensLoader

rng = np.random.default_rng()
target_size = 102400
target_ratios = [0.9, 0.1]

train_dataset_1 = StreamingDataset(
    # input_dir="/data/users/zichunyu/data/fineweb/sample-100BT/pythia-1b/mates/10000-flan/2000000",
    input_dir="/data/users/zichunyu/data/games/train/0",
    item_loader=TokensLoader(block_size=2048 + 1),
    drop_last=True,
)
dataset_size = len(train_dataset_1)
ls = rng.choice(
    dataset_size,
    size=(int(target_size * target_ratios[0]),),
    replace=False,
)
ls = list(map(int, ls))
indices = [(0, i) for i in ls]

train_dataset_2 = StreamingDataset(
    # input_dir="/data/users/zichunyu/data/fineweb-edu/sample-100BT/train",
    input_dir="/data/users/zichunyu/data/fineweb/sample-100BT/train",
    item_loader=TokensLoader(block_size=2048 + 1),
    drop_last=True,
)
dataset_size = len(train_dataset_2)
ls = rng.choice(
    dataset_size,
    size=(int(target_size * target_ratios[1]),),
    replace=False,
)
ls = list(map(int, ls))
indices += [(1, i) for i in ls]

optimize(
    fn=lambda i: train_dataset_1[i[1]] if i[0] == 0 else train_dataset_2[i[1]],
    inputs=indices,
    output_dir=f"/data/users/zichunyu/data/games-fineweb-{target_size}-{target_ratios[0]}-{target_ratios[1]}",
    num_workers=(os.cpu_count() // 8),
    chunk_bytes="200MB",
)
