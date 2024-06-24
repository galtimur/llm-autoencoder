import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
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
    train_pars_str = f"trainable params: {(trainable_parameters/1e6):.2f}M"
    all_param_str = f"all params: {(all_param/1e6):.0f}M"
    trainable_ratio_str = f"trainable %: {(100 * trainable_parameters / all_param):.3f}"
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


def print_trainable_names(model) -> None:
    print("Trainble parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


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

        self.task_type = self.model_args.task_type
        print(f"------- Task type is {self.task_type} -------")
        self.model_name = self.model_args.model_name_or_path
        self.dtype = torch.bfloat16 if self.training_args.bf16 else torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.compression_rate = self.training_args.compression_rate
        self.segment_length = self.training_args.segment_length
        self.num_summary = self.segment_length // self.compression_rate

        self.trainable_modules = []
        self.model_pars = {
            "hidden_size": self.model_args.hidden_size,
            "num_layers": self.model_args.num_layers,
        }

        self.encoder = self.setup_encoder()
        self.add_tokens(self.model_name)  # adds new tokens
        self.setup_new_embeddings_and_linear()
        self.decoder = self.setup_decoder()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.initialize()
        pass

    def add_tokens(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.summ_tokens = torch.arange(
            0,
            self.num_summary,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        # Token after which the autoencoder starts
        self.ae_id = self.num_summary
        # Token after which the continuation starts
        self.summ_id = 0  # self.num_summary + 1
        # self.mask_id = self.num_summary + 2
        self.ae_tok = torch.tensor([[self.ae_id]], device=self.device)
        self.summ_tok = torch.tensor([[self.summ_id]], device=self.device)
        # self.mask_tok = torch.tensor([[self.mask_id]], device=self.device)

    def expand_vocab(self, model, vocab_size: int):
        if hasattr(self, "pad_id"):
            model.config.pad_token_id = self.pad_id
        model.config.bos_token_id = self.bos_id
        model.config.eos_token_id = self.eos_id
        model.resize_token_embeddings(vocab_size)

    def setup_decoder(self):
        print("Loading decoder")
        if self.model_args.share_enc_dec:
            return self.encoder
        elif not self.model_args.init_same_weights:
            decoder = get_model(
                self.model_name,
                self.device,
                self.dtype,
                self.model_args.pretrained_decoder,
                self.model_args.alter_model,
                self.model_pars,
            )
        else:
            decoder = copy.deepcopy(self.encoder)

        if not self.model_args.freeze_decoder:
            if self.model_args.lora_decoder:
                decoder = apply_lora(decoder, self.model_args)
            self.trainable_modules.append(decoder)

        return decoder

    def setup_encoder(self):
        print("Loading encoder")
        encoder = get_model(
            self.model_name,
            self.device,
            self.dtype,
            self.model_args.pretrained_encoder,
            self.model_args.alter_model,
            self.model_pars,
        )
        if self.model_args.lora_encoder:
            encoder = apply_lora(encoder, self.model_args)
        if not self.model_args.freeze_encoder:
            self.trainable_modules.append(encoder)

        return encoder

    def setup_new_embeddings_and_linear(self):
        dim = self.encoder.config.hidden_size
        self.embed_summary = nn.Embedding(
            self.num_summary + 1, dim, device=self.device, dtype=self.dtype
        )  # + [AE] + [SUMM] tokens
        self.embed_compress = nn.Embedding(
            1, dim, device=self.device, dtype=self.dtype
        )  # + [AE] + [SUMM] tokens
        if not self.model_args.freeze_summary:
            self.trainable_modules.append(self.embed_summary)
        if self.task_type == "autocompressor":
            self.trainable_modules.append(self.embed_compress)

        if self.model_args.use_linear_layer:
            self.linear = nn.Linear(dim, dim, device=self.device, dtype=self.dtype)
            if not self.model_args.freeze_linear:
                self.trainable_modules.append(self.linear)
        else:
            self.linear = nn.Identity()

    def initialize(self) -> None:
        print("Freezing the decoder...")
        if self.model_args.freeze_decoder:
            freeze_model(self.decoder)
            self.decoder.eval()
        if self.model_args.freeze_encoder:
            freeze_model(self.encoder)
            self.encoder.eval()
        if self.model_args.freeze_summary:
            freeze_model(self.embed_summary)
            self.embed_summary.eval()
        if self.model_args.freeze_linear:
            freeze_model(self.linear)
            self.linear.eval()
        print_trainable_parameters(self)
        # if (
        #     self.training_args.restore_from is not None
        #     and self.training_args.restore_from != ""
        # ):
        #     print(
        #         f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
        #     )
        #     state_dict = load_file(self.training_args.restore_from)
        #     self.load_state_dict(state_dict)
        #     print(f"Finished loading from {self.training_args.restore_from}")

    def embed_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if self.model_args.lora_encoder:
            input_embeds = self.encoder.get_base_model().model.embed_tokens(input_ids)
        else:
            input_embeds = self.encoder.model.embed_tokens(input_ids)

        return input_embeds

    def get_inputs(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_type == "autoencoder":
            prefix_ids, suffix_ids = input_ids, input_ids
        elif self.task_type in ["autocompressor", "base", "base_no_context"]:
            prefix_ids, suffix_ids = input_ids[:, 0, :], input_ids[:, 1, :]
        return prefix_ids, suffix_ids

    def get_embeds(self, prefix_ids, suffix_ids, summ_tokens, batch_size):
        summ_input_embeds = self.embed_summary(summ_tokens)
        prefix_input_embeds = self.embed_tokens(prefix_ids)

        if self.task_type == "autoencoder":
            suffix_input_embeds = prefix_input_embeds
            sep_embed = self.embed_summary(self.ae_tok).repeat(batch_size, 1, 1)
        elif self.task_type in ["autocompressor", "base", "base_no_context"]:
            suffix_input_embeds = self.embed_tokens(suffix_ids)
            sep_embed = self.embed_compress(self.summ_tok).repeat(batch_size, 1, 1)

        return prefix_input_embeds, suffix_input_embeds, summ_input_embeds, sep_embed

    def get_logits_and_targets(
        self, logits: torch.Tensor, suffix_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_type != "base_no_context":
            logits = logits[:, -self.segment_length - 1 : -1, :].reshape(
                -1, logits.size(-1)
            )
            target_ids = suffix_ids.reshape(-1)
        elif self.task_type == "base_no_context":
            logits = logits[:, -self.segment_length : -1, :].reshape(
                -1, logits.size(-1)
            )
            target_ids = suffix_ids[:, 1:].reshape(-1)

        return logits, target_ids

    def forward(self, input_ids: torch.LongTensor) -> Dict:
        batch_size = input_ids.size(0)
        prefix_ids, suffix_ids = self.get_inputs(input_ids)

        summ_tokens = self.summ_tokens.repeat(batch_size, 1)

        # 1. Embed summary and input tokens. Concat them.
        (
            prefix_input_embeds,
            suffix_input_embeds,
            summ_input_embeds,
            sep_embed,
        ) = self.get_embeds(prefix_ids, suffix_ids, summ_tokens, batch_size)

        if self.task_type in ["autoencoder", "autocompressor"]:
            # 2. Encode sequence, get summary_embeddings. Apply linear layer
            input_embeds = torch.cat([prefix_input_embeds, summ_input_embeds], dim=1)
            output = self.encoder(inputs_embeds=input_embeds, output_hidden_states=True)
            summary_embeds = output.hidden_states[-1][:, -self.num_summary :]
            summary_embeds = self.linear(summary_embeds)

            # 3. Decoder input consists of summary_embeddings, autoencoder token embed and original sequence.
            dec_input_embeds = torch.cat(
                [summary_embeds, sep_embed, suffix_input_embeds], dim=1
            )
        elif self.task_type == "base":
            dec_input_embeds = torch.cat(
                [prefix_input_embeds, suffix_input_embeds], dim=1
            )
        elif self.task_type == "base_no_context":
            dec_input_embeds = suffix_input_embeds

        decoder_outputs = self.decoder(inputs_embeds=dec_input_embeds)

        # 4. Calculate loss on the original sequence.
        logits, target_ids = self.get_logits_and_targets(
            decoder_outputs.logits, suffix_ids
        )
        loss = self.loss_fn(logits, target_ids)
        # loss_mask = torch.randint_like(target_ids, 0, 2).to(logits.dtype)

        return {"loss": loss, "logits": logits}


    def generate(self, inputs: Tuple[torch.LongTensor, torch.LongTensor], max_new_tokens: int = 128, stop_list: set = {}) -> torch.LongTensor:

        '''
        Written only batch_size = 1 because of the stop_list implementation
        inputs is Tuple[LongTensor, LongTensor] - prefix and suffix
        prefix - is a context for the generation. Can be compressed by Autocompressor or used directly by the model
        suffix - the main sequence from which the generation is performed
        '''

        self.eval()
        stop_list.add(self.eos_id)
        prefix_ids, suffix_ids = inputs
        batch_size = 1
        summ_tokens = self.summ_tokens.repeat(batch_size, 1)

        (
            prefix_input_embeds,
            suffix_input_embeds,
            summ_input_embeds,
            sep_embed,
        ) = self.get_embeds(prefix_ids, suffix_ids, summ_tokens, batch_size)

        if self.task_type == "autocompressor":
            input_embeds = torch.cat([prefix_input_embeds, summ_input_embeds], dim=1)
            output = self.encoder(inputs_embeds=input_embeds, output_hidden_states=True)
            summary_embeds = output.hidden_states[-1][:, -self.num_summary :]
            summary_embeds = self.linear(summary_embeds)
            dec_input_embeds = torch.cat(
                [summary_embeds, sep_embed, suffix_input_embeds], dim=1
            )
        elif self.task_type == "base":
            dec_input_embeds = torch.cat(
                [prefix_input_embeds, suffix_input_embeds], dim=1
            )
        elif self.task_type == "base_no_context":
            dec_input_embeds = suffix_input_embeds

        generated_ids = []
        past_key_values = None

        n_symbols = 0
        for _ in range(max_new_tokens):
            decoder_outputs = self.decoder(
                inputs_embeds=dec_input_embeds,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = decoder_outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            generated_ids.append(next_token_id.unsqueeze(1))

            next_token_embed = self.embed_tokens(next_token_id).unsqueeze(1)
            dec_input_embeds = next_token_embed

            past_key_values = decoder_outputs.past_key_values

            if next_token_id.item() in stop_list and n_symbols>0:
                break
            n_symbols += 1

        generated_ids = torch.cat(generated_ids, dim=1)
        return generated_ids