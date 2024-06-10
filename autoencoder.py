import copy
from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def freeze_model(model) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False


def print_trainable_parameters(model, print_all_trainable: bool = False) -> None:
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    train_pars_str = f"trainable params: {trainable_parameters}"
    all_param_str = f"all params: {all_param}"
    trainable_ratio_str = f"trainable %: {100 * trainable_parameters / all_param}"
    print(f"{train_pars_str} || {all_param_str} || {trainable_ratio_str}")
    if print_all_trainable:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)


class AutoencoderLP(torch.nn.Module):
    def __init__(self, args: Dict):
        super().__init__()

        self.model_args = args["model"]
        self.training_args = args["train"]
        self.model_name = self.model_args.model_name_or_path
        dtype = torch.float16 if self.training_args.bf16 is False else torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, attn_implementation="flash_attention_2"
        )
        self.encoder = self.encoder.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.compression_rate = self.training_args.compression_rate
        self.segment_length = self.training_args.segment_length
        self.num_summary = self.segment_length // self.compression_rate

        self.vocab_size = self.encoder.config.vocab_size
        self.pad_id = self.vocab_size
        self.ae_id = self.vocab_size + 1
        self.vocab_size += 2  # + [PAD] token + [AE] token
        self.vocab_size += self.num_summary

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.encoder.config.pad_token_id = self.pad_id
        self.encoder.config.bos_token_id = self.bos_id
        self.encoder.config.eos_token_id = self.eos_id
        self.summ_tokens = torch.arange(
            self.vocab_size - self.num_summary,
            self.vocab_size,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        self.ae_tok = torch.tensor([[self.ae_id]], device=device)

        self.encoder.resize_token_embeddings(self.vocab_size)
        self.embedder = self.encoder.model.embed_tokens
        self.decoder = copy.deepcopy(self.encoder)

        self.dim = self.encoder.config.hidden_size

        # TODO LORA has not been tested
        if self.model_args.lora:
            lora_config = LoraConfig(
                r=self.model_args.lora_r,
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                bias=self.model_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.initialize()

    def initialize(self) -> None:
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def forward(self, input_ids: torch.LongTensor = None) -> Dict:
        batch_size = input_ids.size(0)

        ae_embed = self.embedder(self.ae_tok).repeat(batch_size, 1, 1)
        summ_tokens = self.summ_tokens.repeat(batch_size, 1)
        segment_input_ids = torch.cat([input_ids, summ_tokens], dim=1)
        input_embeds = self.embedder(segment_input_ids)
        segment_input_embeds = input_embeds[:, :self.segment_length]

        output = self.encoder(inputs_embeds=input_embeds, output_hidden_states=True)
        summary_embeds = output.hidden_states[-1][:, -self.num_summary :]

        dec_input_embeds = torch.cat(
            [summary_embeds, ae_embed, segment_input_embeds], dim=1
        )
        decoder_outputs = self.decoder(inputs_embeds=dec_input_embeds)
        # decoder_outputs = self.decoder(inputs_embeds=segment_input_embeds)
        logits = decoder_outputs.logits

        logits = logits[:, -self.segment_length: -1, :].reshape(-1, logits.size(-1))
        target_ids = input_ids[:, 1:].reshape(-1)
        loss = self.loss_fn(logits, target_ids)

        return {"loss": loss, "logits": logits}
