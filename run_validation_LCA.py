from pathlib import Path
import json

from args_parser import parse_config
from train import load_model_from_checkpoint
from eval_lca_code_completion import FLCC_evaluator
from autoencoder import AutoencoderLP

if __name__ == "__main__":

    # ckpt_names = ["base", "base-no-cont", "auco-4x", "auco-10x"]
    ckpt_names = ["base", "base_no_context", "auco-4x-hybrid_DS-1.3-6.7", "auco-10x-hybrid_DS-1.3-6.7"]#"auco-10x-hybrid_DS-1.3-6.7", "base", "base_no_context",
    ckpt_main_folder = Path("/mnt/data2/galimzyanov/llm-autoencoder/output/")
    results_file = "val_CC_res.jsonl"
    config_path = "configs/config_code_base.yaml"
    args_base = parse_config(config_path)

    for ckpt_name in ckpt_names:

        print(f"----------- Validating {ckpt_name} -----------")
        if "base" not in ckpt_name:
            ckpt_folder = ckpt_main_folder / ckpt_name
            model_checkpoint = load_model_from_checkpoint(ckpt_folder)
            model = model_checkpoint["model"]
            tokenizer = model_checkpoint["tokenizer"]
            args = model_checkpoint["args"]
        else:
            args_base["model"].task_type = ckpt_name
            model = AutoencoderLP(args_base)
            tokenizer = model.tokenizer

        model.eval()

        cc_evaluator = FLCC_evaluator(model, tokenizer)

        res = cc_evaluator.eval_on_lcc(summ_len = 128, context_len=32, limit = 1000, model_name=ckpt_name)
        with open(results_file, "a") as f:
            f.write(json.dumps({ckpt_name: res}) + "\n")

        print(res["exact_match_rate"])
