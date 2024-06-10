import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from typing import Dict
import wandb

def calc_grad_norm(model):

    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm

class Trainer:
    def __init__(self, model, train_dl: DataLoader, val_dl: DataLoader, args: Dict):

        train_args = args["train"]
        batch_size_global = train_args.batch_size_global
        self.accumulation_steps = batch_size_global // train_args.batch_size_mini

        self.model = model
        self.num_epochs = train_args.num_train_epochs
        self.encoder = self.model.encoder
        self.optimizer = AdamW(self.encoder.parameters(), lr=train_args.learning_rate)
        self.encoder.train()
        self.progress_train = tqdm(train_dl, total=len(train_dl))
        self.progress_val = tqdm(val_dl, total=len(val_dl))
        self.eval_steps = train_args.eval_steps
        self.max_eval_steps = train_args.max_eval_steps

        model_name = args["model"].model_name_or_path.split("/")[-1]
        wandb_run_name = f"{model_name}"
        wandb_run_name += f"cr_{train_args.compression_rate}"
        wandb_run_name += f"seg_{train_args.segment_length}"
        wandb_run_name += f"batch_{batch_size_global}"
        # TODO make one more key in args - training args, that is setup in config, not default
        wandb.init(
            project=train_args.wandb_project_name, config=args, name=wandb_run_name
        )

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            for step, item in enumerate(self.progress_train, start=1):
                if item is None:
                    continue

                outputs = self.model(item)
                loss = outputs["loss"]
                loss /= self.accumulation_steps
                loss.backward()

                if step % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.progress_train.set_postfix({"loss": loss.item()}, refresh=True)
                    grad_norm = calc_grad_norm(self.encoder)
                    self.optimizer.zero_grad()
                    log_dict = {"loss": loss.item(), "grad_norm": grad_norm}
                    wandb.log(log_dict)

                if step % self.eval_steps == 0:
                    loss = self.validate()
                    log_dict = {"val/loss": loss}
                    wandb.log(log_dict)

    def validate(self):

        print(f"------ Validating ------")
        self.encoder.eval()
        total_loss = 0


        for step, item in enumerate(self.progress_val, start=1):
            if item is None:
                continue
            with torch.no_grad():
                outputs = self.model(item)
            loss = outputs["loss"].item()
            total_loss += loss

            if step > self.max_eval_steps:
                break

        val_loss = total_loss / step
        self.encoder.train()
        return val_loss
