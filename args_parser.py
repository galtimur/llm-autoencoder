import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import transformers
import yaml

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    task_type: str = field(
        default="autoencoder",
        metadata={"help": "options: 'autoencoder', 'autocompressor'"},
    )
    flash_attn: bool = field(default=True)
    pretrained_encoder: bool = field(
        default=True, metadata={"help": "Start from pretrained encoder"}
    )
    pretrained_decoder: bool = field(
        default=True, metadata={"help": "Start from pretrained decoder"}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Do not train encoder"}
    )
    freeze_decoder: bool = field(
        default=True, metadata={"help": "Do not train decoder"}
    )
    share_enc_dec: bool = field(
        default=False, metadata={"help": "Use same instance for decoder and encoder"}
    )
    init_same_weights: bool = field(
        default=True,
        metadata={"help": "Init decoder with same weights as encoder if possible"},
    )
    use_linear_layer: bool = field(
        default=True,
        metadata={
            "help": "Apply linear layer to memory embeds between ancoder and decoder"
        },
    )
    lora_encoder: bool = field(
        default=False, metadata={"help": "Whether to use LORA on encoder"}
    )
    lora_decoder: bool = field(
        default=False, metadata={"help": "Whether to use LORA on decoder"}
    )
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_alpha: int = field(default=32, metadata={"help": "lora alpha"})
    lora_bias: str = field(default="none", metadata={"help": "lora bias"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    lora_target_modules: List[str] = field(default=list)
    alter_model: bool = field(
        default=False,
        metadata={
            "help": "Change model hyperparameters. Works only model is not pretrained"
        },
    )
    hidden_size: int = field(default=None)
    num_layers: int = field(default=None)


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
    train_dataset_subname: str = field(default=None)
    train_dataset_subdir: str = field(default=None)
    val_dataset_name: str = field(default=None)
    val_dataset_subname: str = field(default=None)
    val_dataset_subdir: str = field(default=None)
    rnd_seed: int = field(default=42, metadata={"help": "Rnd seed for data shuffling"})
    text_key: str = field(
        default="text", metadata={"help": "The key in dataset for the text item"}
    )
    validate_ce: bool = field(
        default=False,
        metadata={"help": "Whether to calculate CE loss on validation"},
    )
    validate_em: bool = field(
        default=False,
        metadata={"help": "Whether to calculate exact match metric loss on validation"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)

    optim: str = field(default="adamw_torch")
    # TODO may be do it in terms of samples or segments
    max_eval_steps: int = field(
        default=100,
        metadata={"help": "How many samples to use for validation."},
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


def process_args(model_args, data_args, training_args) -> Tuple:
    if model_args.task_type not in ["autoencoder", "autocompressor"]:
        raise ValueError(
            f"Wrong model type {model_args.model_type}. Allowed options: ['autoencoder', 'autocompressor']"
        )
    training_args.learning_rate = float(training_args.learning_rate)
    if model_args.freeze_decoder and model_args.freeze_encoder:
        print("!!!! NOTE that you freezed both encoder and decoder")
    if model_args.share_enc_dec:
        eq_freeze = model_args.freeze_decoder == model_args.freeze_encoder
        eq_pretrained = model_args.pretrained_decoder == model_args.pretrained_encoder
        if not (eq_freeze and eq_pretrained):
            raise ValueError(
                "If you share decoder and encoder, the freezing and pretraining flags should be equal"
            )
    if model_args.freeze_decoder and not model_args.pretrained_decoder:
        print("!!!! NOTE you freezed not pretrained decoder")
    if model_args.freeze_encoder and not model_args.pretrained_encoder:
        print("!!!! NOTE you freezed not pretrained encoder")

    return model_args, data_args, training_args


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
    model_args, data_args, training_args = process_args(
        model_args, data_args, training_args
    )

    return {
        "model": model_args,
        "train": training_args,
        "data": data_args,
        "train_short": config_raw["TrainingArguments"],
    }
