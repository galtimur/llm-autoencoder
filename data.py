from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_texts(
    tokenized_texts: List[List[int]], seq_length: int
) -> List[torch.Tensor]:
    # tokenized_texts = [torch.tensor(text) for text in tokenized_texts]

    text_segments = []
    text_ids_merged = []

    for text_ids in tokenized_texts:
        text_ids_merged.extend(text_ids)
        if len(text_ids_merged) < seq_length:
            continue

        segments = [
            text_ids_merged[i : i + seq_length]
            for i in range(0, len(text_ids_merged), seq_length)
        ]
        if len(segments[-1]) < seq_length:
            segments = segments[:-1]
        text_segments.extend(segments)

        text_ids_merged = []

    return text_segments


class AuCoBatcher:
    def __init__(
        self,
        seq_length: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        text_key: str = "text",
    ):
        self.batch_size = batch_size
        self.buffer = []
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_key = text_key

    def __call__(self, batch: List[Dict]) -> torch.Tensor:
        # Just to make memory safe
        if len(self.buffer) > 10000:
            self.buffer = []

        batch = [item[self.text_key] for item in batch]
        tokenized_texts = self.tokenizer(batch, truncation=False, padding=False)[
            "input_ids"
        ]

        segments = split_texts(tokenized_texts, self.seq_length)
        self.buffer.extend(segments)
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer[: self.batch_size]
        self.buffer = self.buffer[self.batch_size :]

        return torch.tensor(batch, device=device)


def get_dataloader(split: str, args: Dict, tokenizer: AutoTokenizer) -> DataLoader:
    training_args = args["train"]
    data_args = args["data"]

    # TODO make it better. Possibly move to a class
    if split == "train":
        dataset_name = data_args.train_dataset_name
        data_subset = data_args.train_dataset_subset
    elif split == "val":
        dataset_name = data_args.val_dataset_name
        data_subset = data_args.val_dataset_subset

    seg_len = training_args.segment_length
    batch_size = training_args.batch_size
    outer_batch_size = training_args.outer_batch_size

    custom_batcher = AuCoBatcher(
        seq_length=seg_len,
        batch_size=batch_size,
        tokenizer=tokenizer,
        text_key=data_args.text_key,
    )

    dataset = load_dataset(dataset_name, name=data_subset)["train"]
    dataset = dataset.shuffle(data_args.rnd_seed)

    dataloader = DataLoader(
        dataset, batch_size=outer_batch_size, collate_fn=custom_batcher
    )

    return dataloader


def get_data(args: Dict) -> Tuple[DataLoader, DataLoader]:
    model_args = args["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    train_dl = get_dataloader("train", args, tokenizer)
    val_dl = get_dataloader("val", args, tokenizer)

    return train_dl, val_dl
