from pathlib import Path
from tqdm import tqdm
import random
import torch
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.api.task import Task
from litgpt import Tokenizer


def encode_pair(tokenizer: Tokenizer, context: str, continuation: str):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tokenizer.encode(context + continuation, bos=False, eos=False).tolist()
    context_enc = tokenizer.encode(context, bos=False, eos=False).tolist()
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc


def prepare_sample(
    task: Task,
    example: dict,
    tokenizer: Tokenizer,
    ignore_index: int,
) -> dict:
    context = task.doc_to_text(example)
    target = task.doc_to_target(example)
    context_enc, continuation_enc = encode_pair(tokenizer, context, target)
    return {
        "input_ids": context_enc + continuation_enc,
        "labels": [ignore_index] * len(context_enc) + continuation_enc,
    }


def prepare(
    destination_path: Path = Path("/data/users/zichunyu/data"),
    tokenizer_dir: Path = Path("checkpoints/EleutherAI/pythia-1b"),
    ignore_index: int = -100,
    task_name: str = "lambada_openai",
) -> None:
    random.seed(1234)
    destination_path = destination_path / task_name
    destination_path.mkdir(parents=True, exist_ok=True)

    print("Loading data file...")
    task_manager = TaskManager("INFO")
    task_dict = get_task_dict([task_name], task_manager)
    task = task_dict[task_name]
    if task_name == "lambada_openai":
        train_set = list(task.test_docs())
        val_set = list(task.test_docs())
    elif task_name == "lambada":
        train_set = list(task.validation_docs())
        val_set = list(task.test_docs())
    else:
        train_set = list(task.training_docs())
        val_set = list(task.validation_docs())

    print("Loading tokenizer...")
    tokenizer = Tokenizer(tokenizer_dir)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(val_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            task,
            example=sample,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
        )
        for sample in tqdm(train_set)
    ]
    random.shuffle(train_set)
    train_set = train_set[:1024]
    torch.save(train_set, destination_path / "train-1024.pt")

    print("Processing val split ...")
    val_set = [
        prepare_sample(
            task,
            example=sample,
            tokenizer=tokenizer,
            ignore_index=ignore_index,
        )
        for sample in tqdm(val_set)
    ]
    torch.save(val_set, destination_path / "val.pt")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
