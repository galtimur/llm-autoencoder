from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, dataloader: DataLoader) -> None:
    for item in tqdm(dataloader):
        if item is None:
            continue
        out = model(item)
        loss = out["loss"]
