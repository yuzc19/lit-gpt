import argparse
import os
from typing import Literal

import datasets
import numpy as np
import torch
from datasets import Dataset
from litdata.streaming import StreamingDataset, TokensLoader
from transformers import AutoModel, AutoTokenizer

model_name = "TaylorAI/bge-micro"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().cuda()


def split_and_tokenize_single(
    text: str,
    pad: bool = False,
    split_strategy: Literal["truncate", "greedy", "even"] = "even",
    max_length: int = 512,
) -> dict[str, list[list[int]]]:
    """
    Split and tokenize a single text to prepare it for the embedding model.
    Padding is only necessary if running more than 1 sequence thru the model at once.
    Splitting happens when the model exceeds the max_length (usually 512).
    You can either truncate the text, or split into chunks. Chunking can be "greedy"
    (as many 512 chunks as possible), or "even" (split into even-ish chunks with np.array_split).
    """

    # first make into tokens
    tokenized = tokenizer(text)  # (seq_len, )

    # if don't have to pad and don't have to split into chunks, we're done
    if not pad and len(tokenized["input_ids"]) <= max_length:
        return {k: [tokenized[k]] for k in tokenized}

    # handle splitting
    if split_strategy == "truncate":
        for k in tokenized:
            tokenized[k] = [tokenized[k][:max_length]]

    elif split_strategy == "greedy":
        for k in tokenized:
            tokenized[k] = [
                tokenized[k][idx : idx + max_length]
                for idx in range(0, len(tokenized[k]), max_length)
            ]

    elif split_strategy == "even":
        for k in tokenized:
            tokenized[k] = [
                arr.tolist()
                for arr in np.array_split(
                    tokenized[k],
                    int(np.ceil(len(tokenized[k]) / max_length)),
                )
            ]

    else:
        raise ValueError(
            f"split_strategy must be 'truncate', 'greedy', or 'even', not {split_strategy}"
        )

    # pad if applicable
    if pad:
        # first make sure list is nested
        if not isinstance(tokenized["input_ids"][0], list):
            for k in tokenized:
                tokenized[k] = [tokenized[k]]

        # get pad token
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        pad_len = max_length
        for k in tokenized:
            tokenized[k] = [
                np.pad(
                    tokenized[k][i],
                    (0, pad_len - len(tokenized[k][i])),
                    constant_values=pad_token_id,
                ).tolist()
                for i in range(len(tokenized[k]))
            ]

    return tokenized


def split_and_tokenize_batch(
    texts: str,
    pad: bool = False,
    split_strategy: Literal["truncate", "greedy", "even"] = "even",
    max_length: int = 512,
) -> dict:
    """
    Tokenize the text and pad if applicable.

    :param text: The input text to be tokenized.
    :type text: str
    :return: Returns a tuple. dictionary containing tokenized and padded 'input_ids',
    'attention_mask' and 'token_type_ids', along with a list of offsets.
    :rtype: Tuple[Dict[str, numpy.ndarray], list[int]]

    Example:

    .. code-block:: python

        tokenized_text = model.split_and_tokenize('sample text')
    """
    result = {}
    offsets = [0]

    # first tokenize without padding
    for text in texts:
        tokenized = split_and_tokenize_single(
            text,
            pad=True,
            split_strategy=split_strategy,
        )
        for k in tokenized:
            if k not in result:
                result[k] = tokenized[k]
            else:
                result[k].extend(tokenized[k])

        offsets.append(len(result["input_ids"]))

    # then, if padding, use longest length in batch
    if pad:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        pad_len = max_length
        for k in result:
            result[k] = [
                np.pad(
                    result[k][i],
                    (0, pad_len - len(result[k][i])),
                    constant_values=pad_token_id,
                ).tolist()
                for i in range(len(result[k]))
            ]

    return {
        "tokens": result,
        "offsets": offsets,
    }


@torch.no_grad()
def embed_batch(
    texts: list[str],
    normalize: bool = False,
    split_strategy: Literal["truncate", "greedy", "even"] = "even",
    batch_size: int = 128,
):
    tokenized = split_and_tokenize_batch(
        texts,
        pad=True,
        split_strategy=split_strategy,
    )
    inputs = tokenized["tokens"]
    offsets = tokenized["offsets"]
    outputs = None
    for i in range(0, len(inputs["input_ids"]), batch_size):
        # import time
        # start = time.time()
        batch_out = (
            model(
                input_ids=torch.tensor(
                    inputs["input_ids"][i : i + batch_size],
                    device="cuda",
                ),
                attention_mask=torch.tensor(
                    inputs["attention_mask"][i : i + batch_size],
                    device="cuda",
                ),
            )
            .pooler_output.float()
            .cpu()
            .numpy()
        )
        if outputs is None:
            outputs = batch_out
        else:
            outputs = np.concatenate([outputs, batch_out], axis=0)
        # end = time.time()
        # print(f"Batch {i} took {end - start} seconds")

    # use offsets to average each text's embeddings
    embs = []
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        chunk = outputs[start:end]
        averaged = chunk.mean(axis=0)
        if normalize:
            averaged = averaged / np.linalg.norm(averaged)
        embs.append(averaged)

    return np.array(embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/users/zichunyu")
    parser.add_argument("--model_name", type=str, default="pythia-1b")
    parser.add_argument("--ckpt", type=int, default=10000)
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    # output_dir = f"{args.base_dir}/out/{args.model_name}/fineweb/sample-350BT/train/0/{args.ckpt}-data_influence_model-prediction"

    dataset = datasets.load_dataset(
        "json",
        data_files="/data/users/zichunyu/data/fineweb/sample-350BT/val.jsonl",
    )

    dataset = dataset.map(
        lambda x: {"__embedding": embed_batch(x["text"])},
        batched=True,
        batch_size=args.device_batch_size,
    )

    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
