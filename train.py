import os
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

from args_parser import parse_config
from autoencoder import AutoencoderLP


def calc_grad_norm(module_list: List) -> float:
    total_norm = 0
    for module in module_list:
        for name, p in module.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)

    return total_norm


def save_model(model, checkpoint_folder: str | Path, current_state: dict):
    checkpoint_folder = Path(checkpoint_folder)
    checkpoint_path = checkpoint_folder / "checkpoint.pt"
    config_path = checkpoint_folder / "config.json"
    torch.save(
        {
            "tokens": current_state["tokens"],
            "model_state_dict": model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            "loss": current_state["loss"],
        },
        checkpoint_path,
    )
    model.config.save_pretrained(config_path)


def load_model(checkpoint_folder: str | Path) -> Dict:
    print("------- Loading the model from the checkpoint -------")

    checkpoint_folder = Path(checkpoint_folder)
    config_path = checkpoint_folder / "config.yaml"
    checkpoint_path = checkpoint_folder / "checkpoint.pt"

    args = parse_config(config_path)
    autoencoder = AutoencoderLP(args)

    checkpoint = torch.load(checkpoint_path)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])

    if "tokens" in checkpoint:
        tokens = checkpoint["tokens"]
        print(f"Number of tokens trained: {tokens}")
    if "tokens" in checkpoint:
        loss = checkpoint["loss"]
        print(f"Last loss: {loss}")

    return {"model": autoencoder, "args": args}


def change_model_mode(module_list: List, mode: str) -> None:
    if mode not in ["eval", "train"]:
        ValueError('mode should be "eval" or "train"')
    if mode == "eval":
        [module.eval() for module in module_list]
    if mode == "train":
        [module.train() for module in module_list]


def evaluate_ce(model, dataloader: DataLoader, max_eval_steps: int) -> float:
    print("------ Validating ------")
    total_loss = 0
    step = 0

    progress = tqdm(total=max_eval_steps, leave=True)
    for item in dataloader:
        # dataloader can return None sometimes, since it accumulates buffer of segments
        if item is None:
            continue
        with torch.no_grad():
            outputs = model(item)
        loss = outputs["loss"].item()
        total_loss += loss
        progress.update()
        progress.refresh()
        step += 1
        if step >= max_eval_steps:
            break

    val_loss = total_loss / step
    return val_loss


class Trainer:
    def __init__(
        self,
        model,
        train_dl: DataLoader,
        val_dl: DataLoader,
        args: Dict,
        config_path: str | Path,
    ):
        self.args = args
        self.train_args = args["train"]
        self.batch_size_global = self.train_args.batch_size_global
        self.accumulation_steps = (
            self.batch_size_global // self.train_args.batch_size_mini
        )
        self.val_ce = self.args["data"].validate_ce
        self.val_em = self.args["data"].validate_em
        self.num_epochs = self.train_args.num_train_epochs
        self.eval_steps = self.train_args.eval_steps
        self.max_eval_steps = self.train_args.max_eval_steps
        self.save_steps = self.train_args.save_steps
        self.output_dir = self.train_args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = model
        # trainable_modules are modules to be trained.
        # It could be encoder and/or decoder, linear layer (has not been implemented)
        trainable_parameters = []
        for module in self.model.trainable_modules:
            trainable_parameters.extend(module.parameters())
        self.optimizer = AdamW(trainable_parameters, lr=self.train_args.learning_rate)
        self.set_train()

        self.progress_train = tqdm(train_dl, total=len(train_dl))
        self.val_dl = val_dl

        # Initialize wandb
        self.wandb_init()
        shutil.copy(config_path, os.path.join(self.output_dir, "config.yaml"))

    def wandb_init(self):
        model_name = self.args["model"].model_name_or_path.split("/")[-1]
        wandb_run_name = f"{model_name}"
        wandb_run_name += f"_cr_{self.train_args.compression_rate}"
        wandb_run_name += f"_seg_{self.train_args.segment_length}"
        wandb_run_name += f"_batch_{self.batch_size_global}"
        wandb.init(
            project=self.train_args.wandb_project_name,
            config=self.args,
            name=wandb_run_name,
        )
        wandb.define_metric("Tokens")
        wandb.define_metric("loss vs tokens", step_metric="Tokens")
        wandb.define_metric("val/loss vs tokens", step_metric="Tokens")
        wandb.run.log_code(".")

    def set_train(self) -> None:
        change_model_mode(self.model.trainable_modules, "train")

    def set_eval(self) -> None:
        change_model_mode(self.model.trainable_modules, "eval")

    def train(self) -> None:
        tokens_consumed = 0
        for epoch in range(self.num_epochs):
            train_loss = 0
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            step = 1

            for item in self.progress_train:
                # dataloader can return None sometimes, since it accumulates buffer of segments
                if item is None:
                    continue
                tokens_consumed += item.numel()
                outputs = self.model(item)
                loss = outputs["loss"]
                loss /= self.accumulation_steps
                train_loss += loss.item()
                loss.backward()

                # grad step
                if step % self.accumulation_steps == 0:
                    self.optimizer.step()
                    grad_norm = calc_grad_norm(self.model.trainable_modules)
                    self.progress_train.set_postfix({"loss": train_loss}, refresh=True)
                    log_dict = {
                        "loss": train_loss,
                        "loss vs tokens": train_loss,
                        "grad_norm": grad_norm,
                        "Tokens": tokens_consumed,
                    }
                    wandb.log(log_dict)

                    self.optimizer.zero_grad()
                    train_loss = 0

                # validation
                if step % self.eval_steps == 0:
                    log_dict = {"Tokens": tokens_consumed}
                    if self.val_ce:
                        loss = self.validate_ce()
                        log_dict.update({"val/loss": loss, "val/loss vs tokens": loss})
                    if self.val_em:
                        em = self.validate_em()
                        log_dict.update(
                            {"val/exact match": em, "val/exact match vs tokens": em}
                        )
                    wandb.log(log_dict)

                if step % self.save_steps == 0:
                    print(f"Saving checkpoint to {self.output_dir}")
                    if isinstance(loss, torch.Tensor):
                        loss_float = loss.item()
                    else:
                        loss_float = loss
                    current_state = {"tokens": tokens_consumed, "loss": loss_float}
                    save_model(self.model, self.output_dir, current_state)
                step += 1

    def validate_ce(self):
        self.set_eval()
        val_loss = evaluate_ce(self.model, self.val_dl, self.max_eval_steps)
        self.set_train()

        return val_loss

    def validate_em(self):
        pass
