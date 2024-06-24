from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset, RandomSampler
from transformers import AutoTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_texts(tokenized_batch: List[List[int]], seg_length: int) -> List[List[int]]:
    text_segments = []
    text_ids_merged = []

    """
    Splitting each text into segment of length seg_length.
    Then each segment would be splitted into two and used for either autoencoder or autocompressor
    Returns list of segments.
    """

    for text_ids in tokenized_batch:
        text_ids_merged.extend(text_ids)
        if len(text_ids_merged) < seg_length:
            continue

        segments = [
            text_ids_merged[i : i + seg_length]
            for i in range(0, len(text_ids_merged), seg_length)
        ]
        if len(segments[-1]) < seg_length:
            segments = segments[:-1]
        text_segments.extend(segments)

        text_ids_merged = []

    return text_segments


class AuCoBatcher:
    def __init__(
        self,
        seg_length: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        text_key: str = "text",
        task_type: str = "autoencoder",
    ):
        self.batch_size = batch_size
        self.buffer = []
        self.tokenizer = tokenizer
        self.seg_length = seg_length
        self.text_key = text_key
        self.task_type = task_type

    def __call__(self, batch: List[Dict]) -> torch.Tensor:
        # Just to make memory-safe
        if len(self.buffer) > 10000:
            print(
                f"The buffer has been cleaned. It was too large: {len(self.buffer)} segments"
            )
            self.buffer = []

        batch = [item[self.text_key] for item in batch]
        tokenized_batch = self.tokenizer(batch, truncation=False, padding=False)[
            "input_ids"
        ]

        # Splitting each text into segments.
        if self.task_type == "autoencoder":
            segments = split_texts(tokenized_batch, self.seg_length)
        elif self.task_type in ["autocompressor", "base", "base_no_context"]:
            # Each segment would be splitted into two for autocompressor
            segments = split_texts(tokenized_batch, 2 * self.seg_length)
            segments = [
                [segment[: self.seg_length], segment[self.seg_length :]]
                for segment in segments
            ]
        else:
            raise NotImplementedError(f"Task type {self.task_type} is not implemented")

        self.buffer.extend(segments)
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer[: self.batch_size]
        self.buffer = self.buffer[self.batch_size :]

        return torch.tensor(batch, device=device)


def get_dataloader(split: str, args: Dict, tokenizer: AutoTokenizer) -> DataLoader:
    training_args = args["train"]
    data_args = args["data"]

    seg_len = training_args.segment_length
    batch_size_mini = training_args.batch_size_mini
    batch_size_outer = training_args.batch_size_outer

    custom_batcher = AuCoBatcher(
        seg_length=seg_len,
        batch_size=batch_size_mini,
        tokenizer=tokenizer,
        text_key=data_args.text_key,
        task_type=args["model"].task_type,
    )

    # TODO make it better. Possibly move to a class
    if split == "train" or data_args.val_dataset_name == "train":
        dataset_name = data_args.train_dataset_name
        data_subname = data_args.train_dataset_subname
        data_subdir = data_args.train_dataset_subdir
    elif split == "val":
        dataset_name = data_args.val_dataset_name
        data_subname = data_args.val_dataset_subname
        data_subdir = data_args.val_dataset_subdir

    dataset = load_dataset(dataset_name, name=data_subname, data_dir=data_subdir)[
        "train"
    ]
    if data_args.val_dataset_name == "train":
        # This is implemented to the Stack dataset.
        # I split the dataset not randomly, to avoid files from similar repositories.
        val_samples = training_args.max_eval_samples
        if split == "train":
            sample_range = np.arange(len(dataset)-val_samples)
        if split == "val":
            sample_range = np.arange(len(dataset) - val_samples, len(dataset))
        dataset = Subset(dataset, sample_range)
    # dataset = dataset.shuffle(data_args.rnd_seed)
    generator = torch.Generator()
    generator.manual_seed(data_args.rnd_seed)

    dataloader = DataLoader(
        dataset, batch_size=batch_size_outer, sampler=RandomSampler(dataset, generator=generator), collate_fn=custom_batcher
    )

    return dataloader


def get_data(args: Dict) -> Tuple[DataLoader, DataLoader]:
    model_args = args["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    train_dl = get_dataloader("train", args, tokenizer)

    if args["data"].validate_ce:
        val_dl = get_dataloader("val", args, tokenizer)
    else:
        val_dl = None

    return train_dl, val_dl
