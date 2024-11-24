# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import math
import pprint
import time
import pickle
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, Features, Sequence, Value
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import measure_flops, ThroughputMonitor
from litdata.streaming import StreamingDataset, TokensLoader

from litgpt import Tokenizer
from litgpt.args import EvalArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.data import DataModule, TinyLlama
from litgpt.model import Block, CausalSelfAttention, Config, GPT, LLaMAMLP
from litgpt.utils import (
    capture_hparams,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    CycleIterator,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
    save_hyperparameters,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard
from typing_extensions import Literal


def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    base_dir: Path = Path("litgpt"),
    probe_data_dir: Path = Path(
        "/data/datasets/hf_cache/data/fineweb/sample-350BT/train/0"
    ),
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=2000,
        log_interval=50,
        global_batch_size=16,
        micro_batch_size=16,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
        resume_steps=0,
    ),
    eval: EvalArgs = EvalArgs(interval=20000, max_iters=100),
    optimizer: Union[str, Dict] = {
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 4e-4, "weight_decay": 0.1, "betas": (0.9, 0.95)},
    },
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
    exp_name: Optional[str] = None,
    seed: int = 42,
    rank: int = 0,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Mutually exclusive with
            ``model_config``.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use. Uses all GPUs by default.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    hparams = capture_hparams()
    if model_config is not None and model_name is not None:
        raise ValueError("Only one of `model_name` or `model_config` can be set.")
    elif model_config is None and model_name is None:
        available_models = "\n".join(sorted(name_to_config))
        raise ValueError(
            f"Please specify --model_name <model_name>. Available values:\n{available_models}"
        )
    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = None

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            # state_dict_type="full",
            limit_all_gathers=True,
            sharding_strategy="HYBRID_SHARD",
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=devices, strategy=strategy, precision=precision, loggers=[logger]
    )
    fabric.launch()

    fabric.print(pprint.pformat(hparams))

    main(
        fabric,
        devices,
        seed,
        initial_checkpoint_dir,
        resume,
        config,
        data,
        base_dir,
        out_dir,
        probe_data_dir,
        tokenizer_dir,
        tokenizer,
        train,
        eval,
        optimizer,
        rank,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Path],
    config: Config,
    data: Optional[DataModule],
    base_dir: Path,
    out_dir: Path,
    probe_data_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    rank: int,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # model = torch.compile(model)
    model = fabric.setup(model)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(
        optimizer, model.parameters(), **extra_kwargs
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # different for each rank
    train_dataset = StreamingDataset(
        input_dir=str(probe_data_dir),
        item_loader=TokensLoader(block_size=model.max_seq_length + 1),
    )
    train_dataset = train_dataset[:500000]
    print("Rank", rank, "Size", len(train_dataset))  # 3150, each showing the same

    if data:
        # This will increase the inference time (3s -> 25s)
        data.connect(
            tokenizer=tokenizer,
            batch_size=32,
            max_seq_length=model.max_seq_length,
        )
        with fabric.rank_zero_first():
            data.download_dir = base_dir / "data/tulu"
            data.prepare_data()
        data.setup()
        val_dataloader_1 = data.val_dataloader()

    else:

        def val_collate_fn(batch):
            input_ids = [torch.tensor(s["input_ids"], device="cuda") for s in batch]
            labels = [torch.tensor(s["labels"], device="cuda") for s in batch]

            x = pad_sequence(input_ids, batch_first=True, padding_value=0)
            y = pad_sequence(labels, batch_first=True, padding_value=-100)

            max_seq_length = model.max_seq_length
            if max_seq_length:
                x = x[:, :max_seq_length]
                y = y[:, :max_seq_length]

            return {"input_ids": x, "labels": y}

        val_dataloader_1 = DataLoader(
            # torch.load(base_dir / "data/lambada_openai/train-1024.pt"),
            torch.load(base_dir / "data/sciq/train-32.pt")
            + torch.load(base_dir / "data/piqa/train-32.pt"),
            batch_size=32,
            collate_fn=val_collate_fn,
        )

    val_dataloader_1 = fabric.setup_dataloaders(val_dataloader_1)
    val_dataloaders = [val_dataloader_1]

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    if train.resume_steps > 0:
        resume = out_dir / (f"step-{train.resume_steps:08d}/lit_model.pth")
    state = fabric.load(resume)

    train_time = time.perf_counter()

    dataset_len = len(train_dataset)
    # 40s for 1 steps
    # probe_steps = 1
    start = -1
    cnt = 10000
    oracle = []
    checkpoint_path = f"out/step-{train.resume_steps}-comb-sp/{config.name}/{rank}.pkl"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            start = checkpoint["start"]
            oracle = checkpoint["oracle"]
            fabric.print(f"Resuming from checkpoint start={start}")

    for i in tqdm(range(cnt)):
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # train_indices = np.random.permutation(probe_index)[
        #     : train.global_batch_size * probe_steps
        # ]
        # probe_data = train_dataset[int(train_indices[0])].unsqueeze(0)
        # train_data = torch.stack([train_dataset[int(index)] for index in train_indices])
        train_index, probe_index = np.random.permutation(dataset_len)[:2]
        train_data = train_dataset[train_index].unsqueeze(0)
        probe_data = train_dataset[probe_index].unsqueeze(0)
        if i <= start:
            continue
        scores = fit(
            fabric,
            devices,
            {"model": model, "optimizer": optimizer},
            train_data.cuda(),
            probe_data.cuda(),
            val_dataloaders,
            out_dir,
            tokenizer_dir,
            train,
            eval,
        )
        oracle.append(
            {
                "train_index": train_index,
                "probe_index": probe_index,
                "scores": scores,
            }
        )
        if i == cnt - 1 or (i + 1) % 1000 == 0:
            features = Features(
                {
                    # "train_indices": Sequence(Value("int32")),
                    "train_index": Value("int32"),
                    "probe_index": Value("int32"),
                    "scores": Sequence(Value("float32")),
                }
            )
            processed_ds = Dataset.from_list(oracle, features=features)
            save_dir = f"out/step-{train.resume_steps}-comb-sp/{config.name}/{rank}"
            processed_ds.save_to_disk(save_dir)

            with open(checkpoint_path, "wb") as f:
                pickle.dump({"start": i, "oracle": oracle}, f)

    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_data: torch.Tensor,
    probe_data: torch.Tensor,
    val_dataloaders: List[DataLoader],
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
):
    model = state["model"]
    optimizer = state["optimizer"]

    lr = get_wsd_lr(
        optimizer.defaults["lr"],
        1e6 - 1,
        0,
        1e6,
        train.min_lr,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # cnt = 0
    # for train_batch in train_data.split(train.micro_batch_size):
    #     input_ids = train_batch[:, 0 : model.max_seq_length].contiguous().long()
    #     targets = train_batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()

    #     logits = model(input_ids)
    #     loss = chunked_cross_entropy(logits, targets)
    #     fabric.backward(loss / train.gradient_accumulation_iters(1))
    #     fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
    #     cnt += 1

    #     if cnt % train.gradient_accumulation_iters(1) == 0:
    #         optimizer.step()
    #         optimizer.zero_grad()
    # eval_after_train = evaluate(fabric, model, val_dataloaders)[0]

    scores = []
    for now_data in [train_data, probe_data]:
        input_ids = now_data[:, 0 : model.max_seq_length].contiguous().long()
        targets = now_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()

        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        scores.append(evaluate(fabric, model, val_dataloaders)[0])

    # eval_after_probe = evaluate(fabric, model, val_dataloaders)[0]
    return scores


@torch.no_grad()
def evaluate(fabric, model, val_dataloaders):
    model.eval()
    losses = []
    for val_dataloader in val_dataloaders[:1]:
        loss = torch.tensor(0.0, device=fabric.device)
        cnt = 0
        for batch in val_dataloader:
            input_ids, labels = batch["input_ids"], batch["labels"]
            logits = model(input_ids)
            loss += chunked_cross_entropy(logits[:, :-1, :], labels[:, 1:])
            cnt += 1
        loss = loss / cnt
        losses.append(loss.item())
    model.train()
    return losses


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# learning rate decay scheduler (wsd with warmup)
def get_wsd_lr(
    learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it < max_iters:
        return learning_rate
    return learning_rate * math.pow(0.5, (it - max_iters) / 200)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(
                init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd)
            )

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(
                init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer)
            )

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(
    train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume
) -> None:
    issues = []
    unsupported = [(train, ["max_steps", "epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f"{__file__} requires the {name!r} argument. This is set in {args}"
                )
    if initial_checkpoint_dir and resume:
        issues.append(
            "Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one."
        )
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
