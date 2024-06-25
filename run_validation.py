import json
from pathlib import Path

from args_parser import parse_config
from data import get_data
from train import evaluate_ce, load_model_from_checkpoint

if __name__ == "__main__":
    num_steps = 3000
    config_path = "configs/config_code.yaml"
    ckpt_names = ["auco-4x", "auco-10x", "base", "base-no-cont"]
    ckpt_main_folder = Path("/mnt/data2/galimzyanov/llm-autoencoder/output/")
    results_file = "val_res.jsonl"
    args_base = parse_config(config_path)

    for ckpt_name in ckpt_names:
        print(f"----------- Validating {ckpt_name} -----------")
        ckpt_folder = ckpt_main_folder / ckpt_name
        model_checkpoint = load_model_from_checkpoint(ckpt_folder)
        model = model_checkpoint["model"]
        args = model_checkpoint["args"]
        model.eval()

        args_base["model"].task_type = args["model"].task_type
        train_dl, val_dl = get_data(args_base)

        loss = evaluate_ce(model, val_dl, max_eval_steps=num_steps)
        res_dict = {
            "name": ckpt_name,
            "loss": loss,
            "num_items": num_steps * args_base["train"].batch_size_mini,
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(res_dict) + "\n")

        print(loss)
