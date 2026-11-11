"""ctu-aic/mbart25-multilingual-summarization-multilarge-cs, the summarizer for every phase.

The shared executor owns the loss, the optimizer, and generation. This class teaches it
mBART's layer names and packing. Both languages use ``en_XX``: articles and summaries are
``tokens </s> en_XX``, and the decoder starts on ``en_XX``. No language id tells the model
which language to write; the article's own script is the only cue (arm X). G4 adds a text
instruction instead.
"""

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from .model_seq2seq import Seq2SeqLMModel, Seq2SeqLoRA, _seed_for

MODEL_ID = "ctu-aic/mbart25-multilingual-summarization-multilarge-cs"

# facebook/mbart-large-cc25 language codes used by this checkpoint's tokenizer.
LANG_CODES = {"english": "en_XX", "hindi": "en_XX"}

_PROJECTION = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "out_proj": "o",
    "fc1": "wi_0", "fc2": "wo",
}
_PART = {"self_attn": "self_attention", "encoder_attn": "cross_attention"}


def bart_adapter_identity(name):
    """Map ``model.encoder.layers.0.self_attn.q_proj`` onto the executor's adapter names."""
    pieces = name.split(".")
    if len(pieces) < 5 or pieces[0] != "model" or pieces[2] != "layers":
        return None
    stack = pieces[1]
    projection = _PROJECTION.get(pieces[-1])
    if stack not in {"encoder", "decoder"} or projection is None:
        return None
    if pieces[-1] in {"fc1", "fc2"}:
        part = "feed_forward"
    else:
        part = _PART.get(pieces[-2])
    if part is None:
        return None
    return stack, int(pieces[3]) + 1, part, projection


def encode_summary(tokenizer, text, lang_code, limit):
    """mBART packing: tokens, then EOS, then the language id. The decoder start is that id."""
    body = tokenizer.encode(str(text), add_special_tokens=False)
    eos = int(tokenizer.eos_token_id)
    lang_id = int(tokenizer.convert_tokens_to_ids(lang_code))
    if lang_id == int(tokenizer.unk_token_id):
        raise ValueError(f"{lang_code} is not in the mBART vocabulary")
    return body[:max(limit - 2, 0)] + [eos, lang_id]


def _sdpa_factory(config):
    """Scaled-dot-product attention. The shared executor stays on eager attention for mT5."""
    return AutoModelForSeq2SeqLM.from_config(config, attn_implementation="sdpa")


SHARED_CODE = LANG_CODES["english"]


class MBart25Summarizer(Seq2SeqLMModel):
    model_id = MODEL_ID
    model_type = "mbart"
    stack_paths = {"encoder": "model.encoder", "decoder": "model.decoder"}
    blocks_attribute = "layers"

    def __init__(self, settings=None, **kwargs):
        kwargs.setdefault("model_factory", _sdpa_factory)
        super().__init__(settings, **kwargs)

    def _prepare(self):
        super()._prepare()
        # The executor counts T5's num_layers. mBART stores encoder_layers / decoder_layers.
        if getattr(self.config, "num_layers", None) is None:
            self.config.num_layers = int(self.config.encoder_layers)
        if getattr(self.config, "num_decoder_layers", None) is None:
            self.config.num_decoder_layers = int(self.config.decoder_layers)
        # encode() warns once the raw article exceeds 1024, before the probe truncates it.
        if self._tokenizer is not None:
            self._tokenizer.model_max_length = int(1e9)
            # Greedy writing starts on the shared language id for both languages.
            self.config.decoder_start_token_id = self.language_id()

    def language_id(self, lang_code=SHARED_CODE):
        tokenizer = self._tokenizer if self._tokenizer is not None else self.tokenizer
        lang_id = int(tokenizer.convert_tokens_to_ids(lang_code))
        if lang_id == int(tokenizer.unk_token_id):
            raise ValueError(f"{lang_code} is not in the mBART vocabulary")
        return lang_id

    def encode(self, text):
        """Plain tokens, then EOS. The preprocessing suffix adds the language id after it."""
        return self.tokenizer.encode(str(text), add_special_tokens=False) + [int(self.tokenizer.eos_token_id)]

    def preprocessing_kwargs(self, *, max_source=None, max_target=None):
        """mBART packing on both sides: ``tokens </s> en_XX``. Training moves ``en_XX`` to the decoder start."""
        return {**super().preprocessing_kwargs(max_source=max_source, max_target=max_target),
                "suffix": (self.language_id(),)}

    def activate(self):
        super().activate()
        # The Arc has 30 GB. Recomputing all 24 blocks each step is slower than storing them.
        if self.role != "serving" and self.module is not None and getattr(self.module, "is_gradient_checkpointing", False):
            self.module.gradient_checkpointing_disable()

    def _attach_lora(self, module, *, deterministic):
        for name, child in list(module.named_modules()):
            identity = bart_adapter_identity(name)
            if identity is None or not isinstance(child, nn.Linear):
                continue
            stack, block, part, projection = identity
            parent, attribute = name.rsplit(".", 1)
            setattr(module.get_submodule(parent), attribute, Seq2SeqLoRA(
                child, self.settings.rank, self.settings.alpha, block_key=f"{stack}.{block}",
                block_index=block - 1, part=part, projection=projection,
                deterministic=deterministic, seed=_seed_for(self.settings.seed, name)))
        self._mark_adapters(module)

    def set_decoder_language(self, lang_code):
        """Greedy decoding starts on the target language id. Training does not use this."""
        lang_id = int(self.tokenizer.convert_tokens_to_ids(lang_code))
        self.config.decoder_start_token_id = lang_id
        if self.module is not None:
            self.module.config.decoder_start_token_id = lang_id
        return lang_id

    def pad_id(self):
        pad = self.tokenizer.pad_token_id
        return 1 if pad is None else int(pad)
