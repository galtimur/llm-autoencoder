from torch.utils.data import DataLoader


def train(model, dataloader: DataLoader):
    for item in dataloader:
        if item is None:
            continue
        out = model(item)
        loss = out["loss"]
