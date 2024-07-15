# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import logging
import math
import pprint
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import measure_flops, ThroughputMonitor

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
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal


def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    in_dir: Path = Path("in/pretrain"),
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=2000,
        log_interval=50,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=1.2e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
        resume_steps=0,
    ),
    eval: EvalArgs = EvalArgs(interval=20000, max_iters=100),
    optimizer: Union[str, Dict] = {
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 1.2e-4, "weight_decay": 0.1, "betas": (0.9, 0.95)},
    },
    devices: Union[int, str] = "auto",
    tokenizer_dir: Optional[Path] = None,
    logger_name: Literal["wandb", "tensorboard", "csv"] = "tensorboard",
    exp_name: Optional[str] = None,
    seed: int = 42,
):
    hparams = capture_hparams()
    data = TinyLlama() if data is None else data
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

    logger = choose_logger(
        logger_name,
        out_dir,
        name=exp_name,
        resume=resume,
        log_interval=train.log_interval,
    )

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            # Enable it if possible (very useful to save memory)
            activation_checkpointing_policy={Block},
            # Default: Save a single, consolidated checkpoint file
            state_dict_type="full",
            # Enable this if you are close to the max. GPU memory usage
            limit_all_gathers=True,
            # Full-shard within a machine, replicate across machines
            sharding_strategy="HYBRID_SHARD",
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=devices, strategy=strategy, precision=precision, loggers=[logger]
    )
    fabric.launch()

    main(
        fabric,
        seed,
        initial_checkpoint_dir,
        resume,
        config,
        in_dir,
        out_dir,
        tokenizer_dir,
        train,
    )


def main(
    fabric: L.Fabric,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Path],
    config: Config,
    in_dir: Path,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
) -> None:
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # stderr
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

    model = fabric.setup(model)

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {"model": model}

    if train.resume_steps > 0:
        resume = in_dir / f"step-{train.resume_steps:08d}/lit_model.pth"
    else:
        if resume is True:
            resume = max(
                in_dir.rglob("step-*/*.pth"),
                key=(lambda p: int(p.parent.name.split("-")[1])),
            )
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    # Save final checkpoint
    save_checkpoint(
        fabric,
        state,
        tokenizer_dir,
        Path(f"{str(out_dir)}/step-{train.resume_steps:08d}/lit_model.pth"),
    )


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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
