from utils import AutoTokenizer
import selections, utils
import datasets
import os

# load dataset and tokenizer
# (WARNING: this will download a 400GB dataset)


def get_candidate_dataset():
    data_files = [f"data/train-{str(i).zfill(5)}-of-00891*" for i in range(891)]
    return datasets.load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=os.cpu_count() // 2,
        data_files=data_files,
        cache_dir="/data/datasets/models/hf_cache",
        # verification_mode="no_checks",
    )["train"]


# ds = selections.get_candidate_dataset()
ds = get_candidate_dataset()
tokenizer = utils.tokenizer_maker()

# display the first example in text form
text = tokenizer.decode(ds[0]["input_ids"])
# print(text)

tokenizer_kwargs = {"cache_dir": "/data/user_data/zichunyu/.cache/doremi"}
pythia_tokenizer = AutoTokenizer.from_pretrained(
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1", **tokenizer_kwargs
)
pythia_tokenizer.model_max_length = 1024
text1 = tokenizer.decode(ds[0]["input_ids"])

print(text == text1)
