import copy
from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def print_num_parameters(model, model_name: str = "") -> None:
    total_params = sum(p.numel() for p in model.parameters())
    add_model_name = ""
    if len(model_name) > 0:
        add_model_name = f" in {model_name}"
    print(f"Number of parameters{add_model_name}: {total_params / 1e6:.0f}M")


def get_model(
    model_name: str,
    device: torch.device | str,
    dtype: torch.dtype,
    is_pretrained: bool,
    alter_model: bool = False,
    model_pars: Dict = None,
) -> AutoModelForCausalLM:
    if is_pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="flash_attention_2"
        )
    else:
        config_model = AutoConfig.from_pretrained(model_name)
        if alter_model:
            config_model.hidden_size = model_pars["hidden_size"]
            config_model.intermediate_size = 4 * model_pars["hidden_size"]
            config_model.num_hidden_layers = model_pars["num_layers"]
        model = AutoModelForCausalLM.from_config(
            config_model, torch_dtype=dtype, attn_implementation="flash_attention_2"
        )
    print_num_parameters(model)
    return model.to(device)


def get_embedder(model: AutoModelForCausalLM) -> nn.Embedding:
    # Note that works for the model with ROPE, not classic positional encoding
    if hasattr(model, "model"):
        embedder = model.model.embed_tokens
    elif hasattr(model, "base_model"):
        embedder = model.base_model.embed_tokens
    else:
        raise ValueError('model has no "model" or "base_model" attribute')

    return embedder


def apply_lora(model, model_args):
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        bias=model_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


class AutoencoderLP(torch.nn.Module):
    """
    AutoencoderLP is an autoencoder language model that leverages existing LLMs,
    in which encoder encodes the text segments into the summary embedding vectors,
    decoder decodes them into the original sequence.

    The basic architecture is described in (fig. 3):
    https://arxiv.org/pdf/2307.06945
    """

    def __init__(self, args: Dict):
        super().__init__()

        # TODO track for devices (not to pass from self.device (??)) in case we run multi-gpu.
        self.model_args = args["model"]
        self.training_args = args["train"]

        self.compression_rate = self.training_args.compression_rate
        self.segment_length = self.training_args.segment_length
        self.num_summary = self.segment_length // self.compression_rate

        # Getting the base models: encoder and decoder
        self.trainable_modules = []
        self.model_name = self.model_args.model_name_or_path
        self.dtype = torch.bfloat16 if self.training_args.bf16 else torch.float16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading encoder")
        self.model_pars = {
            "hidden_size": self.model_args.hidden_size,
            "num_layers": self.model_args.num_layers,
        }
        self.encoder = get_model(
            self.model_name,
            self.device,
            self.dtype,
            self.model_args.pretrained_encoder,
            self.model_args.alter_model,
            self.model_pars,
        )
        self.dim = self.encoder.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        vocab_size = self.add_tokens(self.encoder.config.vocab_size)  # adds new tokens
        self.expand_vocab(self.encoder, vocab_size)
        if self.model_args.lora_encoder:
            self.encoder = apply_lora(self.encoder, self.model_args)
        self.embed_summary = nn.Embedding(
            self.num_summary + 1, self.dim, device=self.device, dtype=self.dtype
        )  # + [AE] token
        self.trainable_modules.append(self.embed_summary)
        print("Loading decoder")
        self.decoder = self.get_decoder(
            self.model_args.share_enc_dec,
            self.model_args.init_same_weights,
            self.model_args.pretrained_decoder,
        )
        self.expand_vocab(self.decoder, vocab_size)
        if self.model_args.use_linear_layer:
            self.linear = nn.Linear(
                self.dim, self.dim, device=self.device, dtype=self.dtype
            )
            self.trainable_modules.append(self.linear)
        else:
            self.linear = nn.Identity()

        if not self.model_args.freeze_decoder:
            if self.model_args.lora_decoder:
                self.decoder = apply_lora(self.decoder, self.model_args)
            self.trainable_modules.append(self.decoder)
        if not self.model_args.freeze_encoder:
            self.trainable_modules.append(self.encoder)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.initialize()
        pass

    def add_tokens(self, vocab_size: int) -> int:
        vocab_size_new = vocab_size

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.summ_tokens = torch.arange(
            0,
            self.num_summary,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        self.ae_id = self.num_summary
        # self.mask_id = self.num_summary + 1
        self.ae_tok = torch.tensor([[self.ae_id]], device=self.device)
        # self.mask_tok = torch.tensor([[self.mask_id]], device=self.device)

        return vocab_size_new

    def expand_vocab(self, model, vocab_size: int):
        if hasattr(self, "pad_id"):
            model.config.pad_token_id = self.pad_id
        model.config.bos_token_id = self.bos_id
        model.config.eos_token_id = self.eos_id
        model.resize_token_embeddings(vocab_size)

    def get_decoder(
        self, share_enc_dec: bool, init_same_weights: bool, pretrained_decoder: bool
    ):
        if share_enc_dec:
            return self.encoder
        elif not init_same_weights:
            return get_model(
                self.model_name,
                self.device,
                self.dtype,
                pretrained_decoder,
                self.model_args.alter_model,
                self.model_pars,
            )
        else:
            return copy.deepcopy(self.encoder)

    def initialize(self) -> None:
        print("Freezing the decoder...")
        if self.model_args.freeze_decoder:
            freeze_model(self.decoder)
            self.decoder.eval()
        if self.model_args.freeze_encoder:
            freeze_model(self.encoder)
            self.encoder.eval()
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

        ae_embed = self.embed_summary(self.ae_tok).repeat(batch_size, 1, 1)
        summ_tokens = self.summ_tokens.repeat(batch_size, 1)

        # 1. Embed summary and input tokens. Concat them.
        summ_input_embeds = self.embed_summary(summ_tokens)
        if self.model_args.lora_encoder:
            segment_input_embeds = self.encoder.get_base_model().model.embed_tokens(
                input_ids
            )
        else:
            segment_input_embeds = self.encoder.model.embed_tokens(input_ids)
        input_embeds = torch.cat([segment_input_embeds, summ_input_embeds], dim=1)

        # 2. Encode sequence, get summary_embeddings. Apply linear layer
        output = self.encoder(inputs_embeds=input_embeds, output_hidden_states=True)
        summary_embeds = output.hidden_states[-1][:, -self.num_summary :]
        summary_embeds = self.linear(summary_embeds)

        # 3. Decoder input consists of summary_embeddings, autoencoder token embed and original sequence.
        dec_input_embeds = torch.cat(
            [summary_embeds, ae_embed, segment_input_embeds], dim=1
        )
        decoder_outputs = self.decoder(inputs_embeds=dec_input_embeds)
        logits = decoder_outputs.logits

        # 4. Calculate loss on the original sequence.
        logits = logits[:, -self.segment_length : -1, :].reshape(-1, logits.size(-1))
        target_ids = input_ids[:, 1:].reshape(-1)
        loss = self.loss_fn(logits, target_ids)
        # loss_mask = torch.randint_like(target_ids, 0, 2).to(logits.dtype)

        return {"loss": loss, "logits": logits}
