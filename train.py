from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from typing import Dict


class Trainer:
    def __init__(self, model, dataloader: DataLoader, args: Dict):
        batch_size_global = args["train"].batch_size_global
        self.accumulation_steps = batch_size_global // args["train"].batch_size_mini

        self.model = model
        self.dataloader = dataloader
        self.epochs = args["train"].epochs
        self.optimizer = AdamW(self.model.encoder.parameters())
        self.model.encoder.train()
        self.progress = tqdm(self.dataloader, total=len(self.dataloader))

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
