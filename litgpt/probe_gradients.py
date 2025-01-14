# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import pprint
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from datasets import Dataset, Features, Sequence, Value
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import measure_flops, ThroughputMonitor
from litdata.streaming import CombinedStreamingDataset, StreamingDataset, TokensLoader

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
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from torch.nn.functional import normalize
import psutil



def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except Exception as e:
        print(f"Failed to use CudaProjector: {e}")
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def setup(
    model_name: Optional[str] = None,
    model_config: Optional[Config] = None,
    base_dir: Path = Path("litgpt"),
    train_data_dir: Path = Path("/data/datasets/hf_cache/data/fineweb/sample-10BT/val"),
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = "bf16-true",
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=2000,
        log_interval=50,
        global_batch_size=512,
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
        "init_args": {"lr": 0.0001, "weight_decay": 0.1, "betas": (0.9, 0.95)},
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
        train_data_dir,
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
    train_data_dir: Path,
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

    model = fabric.setup(model)

    # extra_kwargs = {"fused": fabric.device.type == "cuda"}
    # optimizer = instantiate_torch_optimizer(
    #     optimizer, model.parameters(), **extra_kwargs
    # )
    # optimizer = fabric.setup_optimizers(optimizer)

    # different for each rank
    train_dataset = StreamingDataset(
        input_dir=str(train_data_dir),
        item_loader=TokensLoader(block_size=model.max_seq_length + 1),
        drop_last=True,
    )
    shard_size = 51200 # Decay 102400/8 * 4 = 51200 #TODO
    train_dataset = train_dataset[
        rank
        * shard_size : (
            (rank + 1) * shard_size if rank + 1 < 128 else len(train_dataset)
        )
    ]
    print("Rank", rank, "Size", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        pin_memory=True,
    )
    train_dataloader = fabric.setup_dataloaders(
        train_dataloader
    )

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    if train.resume_steps > 0:
        resume = out_dir / (f"step-{train.resume_steps:08d}/lit_model.pth")
    fabric.print(f"Resuming training from {resume}")
    state = fabric.load(resume)
    model.load_state_dict(state["model"]) # TODO del
    # optimizer.load_state_dict(state["optimizer"]) # don't need optimizer state
    fabric.print(f"Loaded model from {resume}")

    # checkpointing for preempt
    import os, pickle
    checkpoint_path = f"{out_dir}/checkpoint_{rank}.pkl"
    write_dataset = []
    cnt = 0
    start_index = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            cnt = checkpoint['cnt']
            start_index = checkpoint['start_index']
            write_dataset = checkpoint['write_dataset']
            fabric.print(f"Resuming from checkpoint at count {cnt}, start index {start_index}")

    # Initialize the projector
    projector = get_trak_projector(fabric.device)
    number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number_of_params", number_of_params)

    # Set projection dimensions and parameters
    proj_dim = 8192
    proj = projector(
        grad_dim=number_of_params,
        proj_dim=proj_dim,
        seed=0,
        proj_type=ProjectionType.rademacher,
        device=fabric.device,
        block_size=128,
        max_batch_size=16,
        dtype=torch.bfloat16
    )
    project_interval = 10  # Interval for projecting gradients

    # Initialize training variables
    train_time = time.perf_counter()
    train_iterator = iter(train_dataloader)
    current_full_grads = torch.zeros((project_interval, number_of_params), dtype=torch.float16, device=fabric.device)
    grad_idx = 0

    # Training loop
    for _ in range(start_index):
        next(train_iterator)
    for train_data in tqdm(train_iterator, initial=start_index, total=len(train_dataloader)):
        iter_start_time = time.perf_counter()

        cnt += 1
        grad_start_time = time.perf_counter()

        # Compute gradients for the current batch
        full_grad = fit(
            fabric,
            {"model": model},
            train_data,
        )
        current_full_grads[grad_idx] = full_grad
        grad_idx += 1

        grad_end_time = time.perf_counter()
        # fabric.print(f"Gradient computation time: {grad_end_time - grad_start_time:.4f} seconds")

        proj_start_time = time.perf_counter()

        # Project gradients at specified intervals
        if cnt % project_interval == 0 or cnt == len(train_dataset):
            current_projected_grads = proj.project(current_full_grads, model_id=0) #torch.Size([10, 8192])
            current_projected_grads = normalize(current_projected_grads, dim=1)

            # Append projected gradients to the dataset
            for i in range(grad_idx):
                write_dataset.append(
                    {
                        "__embedding": current_projected_grads[i].cpu().numpy(),
                    }
                )
            del current_projected_grads
            grad_idx = 0  # Reset the index for the next batch
            current_full_grads.zero_()  # Reset the tensor for the next batch
            torch.cuda.empty_cache()

        proj_end_time = time.perf_counter()
        # fabric.print(f"Projection time: {proj_end_time - proj_start_time:.4f} seconds")

        # Save the dataset at regular intervals
        if cnt % 2000 == 0 or cnt == len(train_dataset):
            processed_ds = Dataset.from_list(write_dataset)
            print("processed_ds", processed_ds)
            # print("sample", processed_ds[0]["__embedding"][:10])
            processed_ds.save_to_disk(f"{out_dir}/{rank}")
            print_memory_usage()

            # Save checkpoint for preempt
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'cnt': cnt, 'start_index': cnt, 'write_dataset': write_dataset}, f)

        iter_end_time = time.perf_counter()
        # fabric.print(f"Iteration time: {iter_end_time - iter_start_time:.4f} seconds")

    # Print total training time and memory usage
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def fit(
    fabric: L.Fabric,
    state: dict,
    train_data: torch.Tensor,
) -> None:
    """
    Compute the gradients for the given training data.

    Args:
        fabric (L.Fabric): The Lightning Fabric object.
        state (dict): The state dictionary containing the model.
        train_data (torch.Tensor): The training data. torch.Size([1, 2049])

    Returns:
        torch.Tensor: The computed gradients in torch.float16
    """
    
    model = state["model"]
    model.zero_grad()  # Reset gradients

    # Prepare input and labels
    input_ids = train_data[:, :-1].contiguous().long()
    shift_labels = train_data[:, 1:].contiguous().long()

    # Forward pass
    logits = model(input_ids).to(torch.float16)
    shift_logits = logits.contiguous().float()

    # Compute per-token loss
    pertoken_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-1,
        reduction="none",
    )
    pertoken_loss = pertoken_loss.view(shift_labels.size(0), shift_labels.size(1))

    # Compute per-sequence loss
    perseq_loss = torch.mean(pertoken_loss, dim=1)
    sample_loss = perseq_loss[0]  #  batch size is 1

    # Backward pass to compute gradients
    fabric.backward(sample_loss, retain_graph=True)

    # Concatenate gradients into a single tensor
    sample_grad = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    ).to(torch.float16)  # Convert gradients to float16

    model.zero_grad()  # Reset gradients

    return sample_grad

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RSS: {mem_info.rss / 1e9:.2f} GB, VMS: {mem_info.vms / 1e9:.2f} GB")

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
    # torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
