from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from typing import Dict
import wandb

def calc_grad_norm(model):

    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm.item()

class Trainer:
    def __init__(self, model, dataloader: DataLoader, args: Dict):
        batch_size_global = args["train"].batch_size_global
        self.accumulation_steps = batch_size_global // args["train"].batch_size_mini

        self.model = model
        self.dataloader = dataloader
        self.epochs = args["train"].epochs
        self.train_model = self.model.encoder
        self.optimizer = AdamW(self.train_model.parameters())
        self.train_model.train()
        self.progress = tqdm(self.dataloader, total=len(self.dataloader))

        model_name = args["model"].model_name_or_path.split("/")[-1]
        wandb_run_name = f"{model_name}"
        wandb_run_name += f"cr_{args['train'].compression_rate}"
        wandb_run_name += f"seg_{args['train'].segment_length}"
        wandb_run_name += f"batch_{batch_size_global}"
        # TODO make one more key in args - training args, that is setup in config, not default
        wandb.init(
            project=args["train"].wandb_project_name, config=args, name=wandb_run_name
        )

    def train(self) -> None:
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            for step, item in enumerate(self.progress):
                # Note that DataLoader is specific - it can return None, when accumulating the batch
                if item is None:
                    continue

                outputs = self.model(item)
                loss = outputs["loss"]
                loss /= self.accumulation_steps
                loss.backward()

                if (step + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.progress.set_postfix({"loss": loss.item()}, refresh=True)
                    grad_norm = calc_grad_norm(self.train_model)
                    log_dict = {"loss": loss.item(), "grad_norm": grad_norm}
                    wandb.log(log_dict)
