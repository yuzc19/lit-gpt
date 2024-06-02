import datasets
import numpy as np
from pathlib import Path
import os
import torch as ch

# DATASET_SIZE = 216948746
DATASET_SIZE = 24348947

jeopardy = "jeopardy"
squad = "squad"
lambada = "lambada"
cs_alg = "cs_algorithms"
lm_combo = "lm_task_mix"
gpt3_combo = "gpt3_mix"

DSIR_DATA_PATH = "dsir"
DSDM_DATA_PATH = "dsdm"
CLASSIFIER_DATA_PATH = "classifier"
SDD_DATA_PATH = "sdd.pt"

TARGETS = [jeopardy, squad, lambada, cs_alg, lm_combo, gpt3_combo, "education"]


def get_candidate_dataset():
    data_files = [f"data/train-{str(i).zfill(5)}-of-00891*" for i in range(100)]
    return datasets.load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=os.cpu_count() // 2,
        data_files=data_files,
        cache_dir="/data/datasets/hf_cache",
        verification_mode="no_checks",
    )["train"]


def process_path(path):
    # get file directory
    data_dir = Path("/data/user_data/zichunyu/out/DsDm/data")
    assert data_dir.exists(), f"{data_dir} does not exist!"
    return data_dir / path


def _cached_ch_load(data_path, key):
    return ch.load(process_path(data_path) / f"{key}.pt")[:DATASET_SIZE]


# get indices for dataset selection of size selection_size
def dsdm_select(target_task, selection_size):
    # cs_algorithms: bigbench_cs_algorithms_0-shot
    # squad: squad2_3-shot
    # jeopardy: jeopardy_fixed_3-shot
    # lambada: lambada
    # combo: average of the last three
    if target_task in [squad, lambada, cs_alg, jeopardy]:
        # dm_params = dms[target_task]
        dm_params = _cached_ch_load(DSDM_DATA_PATH, target_task)
    elif target_task == lm_combo:
        dms_squad = _cached_ch_load(DSDM_DATA_PATH, squad)  # mean 0, std 3.1
        dms_lambada = _cached_ch_load(DSDM_DATA_PATH, lambada)  # mean 0, std 2.3
        # dms_jeopardy = _cached_ch_load(DSDM_DATA_PATH, jeopardy)
        dm_params = (dms_squad + dms_lambada) / 2

    # now get the indices for the top k indices
    sorted_indices = ch.argsort(dm_params)
    # check ordering
    assert dm_params[sorted_indices[0]] <= dm_params[sorted_indices[-1]], (
        dm_params[sorted_indices[0]],
        dm_params[sorted_indices[-1]],
    )
    assert dm_params.max() == dm_params[sorted_indices[-1]]
    indices_to_take = sorted_indices[-selection_size:]
    indices_to_take = indices_to_take.cpu().numpy()
    return indices_to_take


def classifier_select(target_task, selection_size):
    assert target_task in [squad, lambada, cs_alg, jeopardy, lm_combo, gpt3_combo]
    weights = _classifier_weights(target_task)
    indices = np.argsort(weights)[::-1][: int(selection_size)]
    return indices


def _classifier_weights(target_task):
    s = _cached_ch_load(CLASSIFIER_DATA_PATH, target_task)
    return s.numpy().astype(np.float64)


def _get_logratios(target_task):
    logratios = _cached_ch_load(DSIR_DATA_PATH, target_task)
    print(len(logratios))
    return logratios.numpy().astype(np.float64)


def dsir_select(target_task, selection_size):
    logratios = _get_logratios(target_task)
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(logratios))
    logratios += gumbel_noise
    return np.argpartition(-logratios, selection_size)[:selection_size]


def qurating_select(target_task, selection_size):
    temperature = 2.0
    metrics = np.array(ch.load(process_path(target_task + ".pt")))
    metrics = (metrics - metrics.mean()) / metrics.std()
    metrics = metrics / temperature

    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise
    return np.argpartition(-metrics, selection_size)[:selection_size]


def tar_select(target_task, selection_size, ckpt=20000):
    temperature = 2.0

    def normalize(input):
        return (input - input.mean()) / input.std()

    if target_task != "mix":
        metrics = np.array(
            ch.load(
                "/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-1b-baseline/iter-080000-ckpt/0-100/lambada_openai.pt"
            )
        ).reshape(-1)
        print("single!", metrics.mean(), metrics.std())
    else:
        print("mix!")
        metrics = (
            +np.array(
                # ch.load(process_path(f"tar/iter-1{ckpt}-ckpt/lambada_openai.pt"))
                ch.load(
                    "/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-410m-baseline/iter-080000-ckpt/0-100/lambada_openai.pt"
                )
            ).reshape(
                -1
            )  # mean 0.5, std 0.7
            + np.array(
                # ch.load(process_path(f"tar/iter-1{ckpt}-ckpt/squad.pt"))
                ch.load(
                    "/data/datasets/hf_cache/c4-pythia/c4-dsdm-pythia-410m-baseline/iter-080000-ckpt/0-100/squad.pt"
                )
            ).reshape(
                -1
            )  # mean 2.9, std 0.06
        ) / 2
    print(metrics.shape)
    # metrics = (metrics - metrics.mean()) / metrics.std()
    # metrics = metrics / temperature

    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise

    return np.argpartition(metrics, selection_size)[:selection_size]


def random_select(selection_size):
    rng = np.random.default_rng()
    return rng.choice(DATASET_SIZE, size=(selection_size,), replace=False)


def semdedup_select(selection_size):
    rng = np.random.default_rng()
    sdd_20pct = ch.load(process_path(SDD_DATA_PATH))
    sdd_20pct = sdd_20pct.numpy().astype(np.int64)

    # modify
    sdd_20pct = ch.tensor([x for x in sdd_20pct if x < DATASET_SIZE])

    return rng.choice(sdd_20pct, size=(selection_size,), replace=False)


TARGETED_METHODS = {
    "dsdm": dsdm_select,
    "classifier": classifier_select,
    "dsir": dsir_select,
    "qurating": qurating_select,
    "tar": tar_select,
}

UNTARGETED_METHODS = {"random": random_select, "semdedup": semdedup_select}


# get train set indices for `method` of size `selection_size`, (maybe) for `target_task`
# return a numpy array of indices
def get_indices(method, selection_size, target_task=None):
    if method in TARGETED_METHODS.keys():
        # assert (
        #     target_task in TARGETS
        # ), f"{target_task} is not a valid target task! OK targets: {TARGETS}"
        print(
            f">> Selecting {selection_size} indices for",
            method,
            "with target task",
            target_task,
        )
        select_it = TARGETED_METHODS[method]
        ls = select_it(target_task, selection_size)
    elif method in UNTARGETED_METHODS.keys():
        assert target_task is None, f"{method} does not support a target task!"
        print(f">> Selecting {selection_size} indices for", method)
        select_it = UNTARGETED_METHODS[method]
        ls = select_it(selection_size)
    else:
        raise NotImplementedError(f"{method} is not implemented!")

    indices = list(map(int, ls))
    return indices


def dataset_select(method, selection_size, target_task=None):
    indices = get_indices(method, selection_size, target_task)
    return get_candidate_dataset().select(indices=indices)


def _test():
    for target_task in list(TARGETS)[::-1]:
        for method in TARGETED_METHODS.keys():
            if target_task == gpt3_combo and method == "dsdm":
                continue

            indices = get_indices(method, 100, target_task)

    for method in UNTARGETED_METHODS.keys():
        indices = get_indices(method, 100)


if __name__ == "__main__":
    _test()
