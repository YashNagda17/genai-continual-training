"""Protocol settings for smoke, pilot, and scientific runs."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PhaseSettings:
    protocol: str = "smoke"
    seed: int = 0
    hindi_seeds: tuple = (1, 2, 3)
    batch_size: int = 16
    replay_per_batch: int = 2
    english_passes: int = 4
    hindi_passes: int = 4
    task_prefix: str = "summarize: "
    analysis_batch: int = 16
    max_new_tokens: int = 3
    encoder_blocks: int = 2
    decoder_blocks: int = 2
    stability_threshold: float = -1.0
    sample_size: int = 1
    shrink_tolerance: float = 1.0
    shrink_grid: tuple = (12, 8, 6, 4)
    new_room_budget: int = 638976
    learning_rate: float = 1e-3
    warmup_fraction: float = 0.01
    no_repeat_ngram: int = 3
    max_seed_spread: float = 1.0
    g4_min_share: float = 0.25
    force_conditional: bool = False


def generation_spec(settings):
    return {"max_new_tokens": settings.max_new_tokens, "do_sample": False,
            "no_repeat_ngram_size": settings.no_repeat_ngram}


def align_settings(settings, model):
    return replace(settings, encoder_blocks=int(model.config.num_layers),
                   decoder_blocks=int(model.config.num_decoder_layers))

