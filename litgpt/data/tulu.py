# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Implementation derived from https://github.com/tloen/alpaca-lora"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch

from litgpt import PromptStyle
from litgpt.data import DataModule, get_sft_collate_fn, SFTDataset
from litgpt.tokenizer import Tokenizer
from torch.utils.data import DataLoader


@dataclass
class Tulu(DataModule):
    """Tulu data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    include_multiturn_conversations: bool = False
    """Whether to include multi-turn conversations in the dataset."""
    download_dir: Path = Path("/data/users/zichunyu/data/tulu")
    """The directory in which the downloaded dataset gets saved."""
    repo_id: str = "allenai/tulu-v2-sft-mixture"
    """The repo from where the data is downloaded"""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        from datasets import load_dataset

        load_dataset(self.repo_id, split=["train"], cache_dir=self.download_dir)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, split=["train"])
        dataset = dataset[0].train_test_split(
            test_size=0.01, seed=self.seed, shuffle=True
        )
        train_data = format_dataset(
            dataset["train"], self.include_multiturn_conversations
        )
        # 998, all the same
        test_data = format_dataset(
            dataset["test"], self.include_multiturn_conversations
        )

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length, ignore_index=self.ignore_index
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length, ignore_index=self.ignore_index
            ),
        )


def format_dataset(
    dataset: List[dict], include_multi_turn_conversations: bool
) -> List[dict]:
    formatted = []

    for entry in dataset:
        if entry["dataset"] != "flan_v2" and entry["dataset"] != "cot":
            continue
        convo = entry["messages"]
        if include_multi_turn_conversations:
            for i in range(0, len(convo) - 1, 2):
                formatted.append(
                    {
                        "instruction": convo[i]["content"],
                        "input": "",
                        "output": convo[i + 1]["content"],
                    }
                )
        else:
            formatted.append(
                {
                    "instruction": convo[0]["content"],
                    "input": "",
                    "output": convo[1]["content"],
                }
            )
    print(len(formatted))

    return formatted
