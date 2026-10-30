"""LFM2.5-350M maps both attention and convolution projections to native Torch LoRA."""

from .model_causal_lm import CausalLMModel


class LFM25Model(CausalLMModel):
    model_id = "LiquidAI/LFM2.5-350M"
    model_type = "lfm2"
    # LFM2 uses out_proj for attention and in_proj/out_proj inside convolution blocks.
    lora_targets = ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj")
