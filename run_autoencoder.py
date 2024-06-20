from args_parser import parse_config
from autoencoder import AutoencoderLP
from data import get_data
from train import Trainer

# TODO add linear layer for summary tokens
# TODO implement save/load
# TODO do initial calculation of the model's perplexity to set the bar.

if __name__ == "__main__":
    config_path = "configs/config_code.yaml"

    args = parse_config(config_path)

    train_dl, val_dl = get_data(args)
    autoencoder = AutoencoderLP(args)
    trainer = Trainer(autoencoder, train_dl, val_dl, args, config_path)

    trainer.train()
    print(1)
