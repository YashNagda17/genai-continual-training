"""Protocol settings for smoke, pilot, and scientific runs."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PhaseSettings:
    protocol: str = "smoke"
    seed: int = 0
    hindi_seeds: tuple = (1, 2, 3)
    batch_size: int = 2
    replay_per_batch: int = 2
    english_passes: int = 1
    hindi_passes: int = 1
    analysis_batch: int = 2
    max_new_tokens: int = 3
    encoder_blocks: int = 2
    decoder_blocks: int = 2
    stability_threshold: float = -1.0
    sample_size: int = 1
    shrink_tolerance: float = 1.0
    shrink_grid: tuple = (12, 8, 6, 4)
    new_room_budget: int = 638976


def generation_spec(settings):
    return {"max_new_tokens": settings.max_new_tokens, "do_sample": False}


def align_settings(settings, model):
    return replace(settings, encoder_blocks=int(model.config.num_layers),
                   decoder_blocks=int(model.config.num_decoder_layers))

