# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule


@dataclass
class C4(DataModule):
    """The C4 data module for pretraining."""

    data_path: Union[str, Path] = Path("/data/users/zichunyu/data/c4")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = (
            max_seq_length + 1
        )  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        if str(self.data_path).startswith("s3://"):
            print(
                f"The C4 data path points to an S3 location: {self.data_path}. Skipping preprocessing."
            )
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(
                f"Found C4 train and val dir: {self.data_path}. Skipping preprocessing."
            )
            return

        dataset = load_dataset(
            "allenai/c4",
            num_proc=os.cpu_count() // 2,
            name="en",
            cache_dir="/data/users/zichunyu/data/hf_cache",
        )
        print("Total examples:", len(dataset))

        # Split the data in training and validation
        split_dataset = dataset
        split_dataset["train"] = (
            split_dataset["train"]
            .shuffle(seed=1314)
            .select(range(int(len(split_dataset["train"]) * 0.6)))
        )
        split_dataset["val"] = split_dataset.pop("validation")

        def tokenize(data: Dataset, index: int):
            yield self.tokenizer.encode(data[index]["text"], eos=True)

        optimize(
            fn=partial(tokenize, split_dataset["train"]),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=(os.cpu_count() // 2),
            chunk_bytes="200MB",
        )
        optimize(
            fn=partial(tokenize, split_dataset["val"]),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=(os.cpu_count() // 2),
            chunk_bytes="200MB",
        )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import (
            StreamingDataLoader,
            StreamingDataset,
            TokensLoader,
        )

        train_dataset = StreamingDataset(
            input_dir=self.data_path_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return val_dataloader
