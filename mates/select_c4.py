import selections
import os

method = "tar"
target = "lambada_openai"
ckpt = "080000"
num_examples = 24348947 // 4
# num_examples = 24348947
indices = selections.get_indices(method, num_examples, target)

# select a subset
ds = selections.get_candidate_dataset()
print(f"Dataset size: {len(ds)}")
selected_ds = ds.select(indices)
selected_ds.save_to_disk(
    f"/data/datasets/hf_cache/c4-pythia-1b/iter-{ckpt}-ckpt/0-100/{method}-{target}-{num_examples}-sample",
    num_proc=os.cpu_count() // 2,
)
