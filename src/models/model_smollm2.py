"""SmolLM2-135M-Instruct uses its native Llama architecture and instruction template."""

from .model_causal_lm import CausalLMModel


class SmolLM2Model(CausalLMModel):
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_type = "llama"
