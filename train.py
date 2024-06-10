from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW

import wandb


def calc_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)

    return total_norm


class Trainer:
    def __init__(self, model, train_dl: DataLoader, val_dl: DataLoader, args: Dict):
        train_args = args["train"]
        batch_size_global = train_args.batch_size_global
        self.accumulation_steps = batch_size_global // train_args.batch_size_mini
        self.num_epochs = train_args.num_train_epochs
        self.eval_steps = train_args.eval_steps
        self.max_eval_steps = train_args.max_eval_steps

        self.model = model
        self.encoder = self.model.encoder
        self.optimizer = AdamW(self.encoder.parameters(), lr=train_args.learning_rate)
        self.encoder.train()

        self.progress_train = tqdm(train_dl, total=len(train_dl))
        self.val_dl = val_dl

        model_name = args["model"].model_name_or_path.split("/")[-1]
        wandb_run_name = f"{model_name}"
        wandb_run_name += f"cr_{train_args.compression_rate}"
        wandb_run_name += f"seg_{train_args.segment_length}"
        wandb_run_name += f"batch_{batch_size_global}"
        wandb.init(
            project=train_args.wandb_project_name, config=args, name=wandb_run_name
        )
        wandb.run.log_code(".")

    def train(self) -> None:
        train_loss = 0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            step = 1

            for item in self.progress_train:
                if item is None:
                    continue

                outputs = self.model(item)
                loss = outputs["loss"]
                loss /= self.accumulation_steps
                train_loss += loss.item()
                loss.backward()

                if step % self.accumulation_steps == 0:
                    self.optimizer.step()
                    grad_norm = calc_grad_norm(self.encoder)

                    self.progress_train.set_postfix({"loss": train_loss}, refresh=True)
                    log_dict = {"loss": train_loss, "grad_norm": grad_norm}
                    wandb.log(log_dict)

                    self.optimizer.zero_grad()
                    train_loss = 0

                if step % self.eval_steps == 0:
                    loss = self.validate()
                    log_dict = {"val/loss": loss}
                    wandb.log(log_dict)

                step += 1

    def validate(self):
        print("------ Validating ------")
        self.encoder.eval()
        total_loss = 0
        step = 0

        progress = tqdm(total=self.max_eval_steps, leave=True)
        for item in self.val_dl:
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
        self.encoder.train()
        return val_loss
