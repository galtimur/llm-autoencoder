import json
import re
import time
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.generation import StoppingCriteria, StoppingCriteriaList


class LcaPythonCompletionDataset(Dataset):
    def __init__(self) -> None:
        self.dataset_name = "jenyag/repo-codegen-py-py-context-path-distance"
        dataset = load_dataset(self.dataset_name)["test"]
        self.samples = []
        for sample in dataset:
            for context, ground_truth in zip(sample["file_context"], sample["gt"]):
                context = sample["project_context"] + context["content"]
                if len(context) == 0:
                    continue
                if context[-1] != "\n":
                    context = context + "\n"
                self.samples.append({"context": context, "gt": ground_truth})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, str]:
        return self.samples[idx]


class StopOnNewLine(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = set()
        for k, tok_id in tokenizer.get_vocab().items():
            s = tokenizer.convert_tokens_to_string([k])
            if "\n" in s:
                self.stop_ids.add(tok_id)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert input_ids.shape[0] == 1  # only batch_size 1 is supported
        if input_ids[0, -1].item() in self.stop_ids:
            return True
        else:
            return False


def eval_on_lcc(
    modules: dict,
    ds_test: str | None,
    context_size: int,
    limit: int | None = None,
    log_negatives: bool = False,
) -> dict:
    model, tokenizer = (modules["model"], modules["tokenizer"])
    device = model.device

    stopping_criteria = StoppingCriteriaList([StopOnNewLine(tokenizer)])
    if ds_test is None:
        ds_test = LcaPythonCompletionDataset()
    max_comp = 128  # max length of the line
    max_len_ctx = (
        context_size - max_comp
    )  # input context should be less that model context size minus max line length
    assert max_len_ctx > 0, "max_len_ctx should be positive!"

    grtrs = []
    preds = []

    num_samples = len(ds_test) if limit is None else limit
    ds_test = ds_test[:num_samples]
    if log_negatives:
        with open(f"out/false_preds_{model_name}.txt", "a") as f:
            f.write("----- New eval -----\n")

    start_time = time.time()
    for sample in tqdm(ds_test):
        input_ids = tokenizer.encode(sample["context"], return_tensors="pt")
        input_ids = input_ids[:, -max_len_ctx:].to(device)

        model_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_comp,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": False,
        }

        with torch.no_grad():
            out = model.generate(**model_kwargs)

        out_tokens = out[0, len(input_ids[0]) - 1 :]
        pred = tokenizer.decode(out_tokens).strip("\n")
        preds.append(pred)
        grtrs.append(sample["gt"])
        if pred != sample["gt"]:
            with open(f"out/false_preds_{model_name}.txt", "a") as f:
                f.write(f"{pred} --> {sample['gt']}\n")

    time_used_cc = time.time() - start_time
    exact_match = sum(gt == pr for gt, pr in zip(grtrs, preds)) / len(preds)

    results = {
        "exact_match_rate": exact_match,
        "Code completion items/s": num_samples / time_used_cc,
        "number of CC items": num_samples,
    }

    return results
