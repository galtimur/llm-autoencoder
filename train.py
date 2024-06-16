import shutil
from typing import Dict, List
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

import wandb


def calc_grad_norm(module_list: List) -> float:
    total_norm = 0
    for module in module_list:
        for name, p in module.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)

    return total_norm

def save_model(model, checkpoint_path: str | Path, current_state: dict):
    tokens = current_state["tokens"]
    loss = current_state["loss"]
    checkpoint_path = os.path.join(checkpoint_path, "checkpoint.pt")
    torch.save({
        'tokens': tokens,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


class Trainer:
    def __init__(self, model, train_dl: DataLoader, val_dl: DataLoader, args: Dict):
        self.args = args
        self.train_args = args["train"]
        self.batch_size_global = self.train_args.batch_size_global
        self.accumulation_steps = (
            self.batch_size_global // self.train_args.batch_size_mini
        )
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
        shutil.copy("configs/config.yaml", os.path.join(self.output_dir, "config.yaml"))

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

    @staticmethod
    def change_model_mode(module_list: List, mode: str) -> None:
        if mode not in ["eval", "train"]:
            ValueError('mode should be "eval" or "train"')
        if mode == "eval":
            [module.eval() for module in module_list]
        if mode == "train":
            [module.train() for module in module_list]

    def set_train(self) -> None:
        self.change_model_mode(self.model.trainable_modules, "train")

    def set_eval(self) -> None:
        self.change_model_mode(self.model.trainable_modules, "eval")

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
                    loss = self.validate()
                    log_dict = {
                        "val/loss": loss,
                        "val/loss vs tokens": loss,
                        "Tokens": tokens_consumed,
                    }
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

    def validate(self):
        print("------ Validating ------")
        self.set_eval()
        total_loss = 0
        step = 0

        progress = tqdm(total=self.max_eval_steps, leave=True)
        for item in self.val_dl:
            # dataloader can return None sometimes, since it accumulates buffer of segments
            if item is None:
                continue
            with torch.no_grad():
                outputs = self.model(item)
            loss = outputs["loss"].item()
            total_loss += loss
            progress.update()
            progress.refresh()
            step += 1
            if step >= self.max_eval_steps:
                break

        val_loss = total_loss / step
        self.set_train()
        return val_loss
