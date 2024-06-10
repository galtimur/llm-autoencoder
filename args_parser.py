import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers
import yaml

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    lora: bool = field(default=False, metadata={"help": "Whether to use LORA"})
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=32, metadata={"help": "lora alpha"})
    lora_bias: str = field(default="none", metadata={"help": "lora bias"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    # train: bool = field(
    #     default=True,
    #     metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    # )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    train_dataset_name: str = field(default=None)
    train_dataset_subset: str = field(default=None)
    val_dataset_name: str = field(default=None)
    val_dataset_subset: str = field(default=None)
    rnd_seed: int = field(default=42, metadata={"help": "Rnd seed for data shuffling"})
    text_key: str = field(
        default="text", metadata={"help": "The key in dataset for the text item"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)

    optim: str = field(default="adamw_torch")
    # model_max_length: int = field(
    #     default=28000,
    #     metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    # )
    # TODO may be do it in terms of samples or segments
    max_eval_steps: int = field(
        default=100,
        metadata={
            "help": "How many samples to use for validation."
        },
    )
    batch_size_mini: int = field(
        default=1,
        metadata={
            "help": "Number of segments passed to a model (each item is splitted into segments)."
        },
    )
    batch_size_global: int = field(
        default=1,
        metadata={"help": "Global batch size, including batch accumulation."},
    )
    batch_size_outer: int = field(
        default=1,
        metadata={
            "help": "Classic batch size, the input to the model does not depend on this number."
        },
    )
    segment_length: int = field(
        default=128,
        metadata={"help": "Segment length to compress."},
    )
    compression_rate: int = field(
        default=4,
        metadata={"help": "Compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={
            "help": "Add a special token for the prompt of language modeling; default: False"
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    wandb_project_name: str = field(default="autoencoder")


def parse_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config_raw = yaml.safe_load(f)

    config = dict()
    for outer_key, outer_value in config_raw.items():
        for inner_key, inner_value in outer_value.items():
            config[inner_key] = inner_value

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_dict(config)

    training_args.learning_rate = float(training_args.learning_rate)

    return {"model": model_args, "train": training_args, "data": data_args}
