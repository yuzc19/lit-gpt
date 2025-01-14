
from litdata.streaming import StreamingDataset, TokensLoader
import datasets
import torch

ds = StreamingDataset(
    input_dir=f"/data/datasets/hf_cache/data/fineweb/sample-10BT/val",
    item_loader=TokensLoader(block_size=2048 + 1),
)
# 17501021

# ds = datasets.load_from_disk(f"/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0_arrow") #51863106

print(len(ds))
# ds = my_dataset.select_columns(["text"])
# format = {'type': 'torch', 'format_kwargs' :{'dtype': torch.int32}}
# ds.set_format(**format)
# # ds = ds.with_format("torch")
print(len(ds))
print(ds[0])

