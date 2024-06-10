from args_parser import parse_config
from autoencoder import AutoencoderLP
from data import get_data
from train import Trainer

# TODO add Validation

if __name__ == "__main__":
    config_path = "configs/config.yaml"

    args = parse_config(config_path)

    train_dl, val_dl = get_data(args)
    autoencoder = AutoencoderLP(args)
    trainer = Trainer(autoencoder, train_dl, args)

    trainer.train()
    print(1)
