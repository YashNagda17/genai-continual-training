from dataclasses import dataclass, replace


PHASE_NAMES = {
    1: "Observation of Forgetting Problem",
    2: "Replay Methods",
    3: "Block Selection",
    4: "Adapter Compression",
    5: "Knowledge Distillation Replay",
}
PHASE_LAYOUT = "report-five-phase-v1"


@dataclass(frozen=True)
class PhaseSettings:
    protocol: str = "smoke"
    seed: int = 0
    batch_size: int = 16
    replay_per_batch: int = 2
    english_passes: int = 4
    hindi_passes: int = 4
    task_prefix: str = "summarize: "
    analysis_batch: int = 4
    max_new_tokens: int = 256
    encoder_blocks: int = 2
    decoder_blocks: int = 2
    stability_threshold: float = -1.0
    sample_size: int = 1
    shrink_tolerance: float = 1.0
    shrink_grid: tuple = (12, 8, 6, 4)
    encoder_block_cost: int = 18432        # Adapter parameters per rank in one block; align_settings measures them
    decoder_block_cost: int = 26624
    learning_rate: float = 1e-3
    warmup_fraction: float = 0.01
    no_repeat_ngram: int = 3
    force_conditional: bool = False


def generation_spec(settings):
    return {"max_new_tokens": settings.max_new_tokens, "do_sample": False,
            "no_repeat_ngram_size": settings.no_repeat_ngram}


def align_settings(settings, model):
    """Measure block counts and adapter costs for the Phase 3 block comparisons."""
    encoder_cost, decoder_cost = model.rank_costs()
    return replace(settings, encoder_blocks=int(model.config.num_layers),
                   decoder_blocks=int(model.config.num_decoder_layers),
                   encoder_block_cost=encoder_cost, decoder_block_cost=decoder_cost)
