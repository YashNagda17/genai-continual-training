"""Qwen3-0.6B with non-thinking instruction templates for every numerical role."""

from .model_causal_lm import CausalLMModel


class Qwen3Model(CausalLMModel):
    model_id = "Qwen/Qwen3-0.6B"
    model_type = "qwen3"
    template_options = {"enable_thinking": False}
