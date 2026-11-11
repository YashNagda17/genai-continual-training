"""Protocol settings, fixed before any Hindi run."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PhaseSettings:
    protocol: str = "scientific"
    device: str = "auto"
    seed: int = 0
    hindi_seeds: tuple = (1, 2, 3)        # The first is the screening seed
    train_rows: int = 2000
    validation_rows: int = 200
    test_rows: int = 500
    pool_rows: int = 500
    serving_rows: int = 200
    batch_size: int = 16
    replay_per_batch: int = 2
    first_passes: int = 4                 # English, or Hindi in the reverse order
    second_passes: int = 4
    task_prefix: str = "summarize: "
    analysis_batch: int = 16
    max_new_tokens: int = 64
    no_repeat: int = 3
    learning_rate: float = 1e-3
    warmup_fraction: float = 0.01         # Linear learning-rate warm-up, then constant
    checkpoint_every: int = 100
    rank_warmup: float = 0.1              # Random replay before the first rescoring
    top_fraction: float = 0.5
    past_filter_fraction: float = 0.2
    past_min_words: int = 5
    repair_beta: float = 1.0
    stop_similarity: float = 0.9
    refresh_steps: int = 200
    encoder_blocks: int = 8
    decoder_blocks: int = 8
    sample_size: int = 64
    stability_threshold: float = 0.5
    new_room_budget: int = 638976
    gamma: float = 1.0
    floor: float = 0.25
    match_tolerance: float = 0.03
    energy_fraction: float = 0.9
    shrink_tolerance: float = 1.0
    shrink_grid: tuple = (12, 8, 6, 4)
    csrel_rounds: int = 5                 # CSReL baseline: greedy selection rounds
    csrel_subset_passes: int = 4          # Passes to fit each round's subset model
    csrel_replay_weight: float = 1.0      # alpha in mean(current) + alpha mean(replay)
    tsr_probe_batches: int = 10           # TSR baseline: reference configuration
    tsr_routing_temperature: float = 0.1
    tsr_replay_probability: float = 0.5
    tsr_kd_weight: float = 0.5
    tsr_kd_temperature: float = 2.0
    max_seed_spread: float = 1.0
    g4_min_share: float = 0.25
    force_conditional: bool = False      # Smoke runs every conditional arm


def protocol_settings(protocol, device="auto", epochs=None):
    settings = PhaseSettings(protocol=protocol, device=device)
    if protocol == "pilot":
        settings = replace(settings, hindi_seeds=(1,))
    elif protocol == "smoke":  # Real mT5-small on a handful of rows; checks the code path, not science.
        settings = replace(settings, hindi_seeds=(1, 2), train_rows=8, validation_rows=4, test_rows=4, pool_rows=8,
                           serving_rows=2, batch_size=4, analysis_batch=4, max_new_tokens=8, first_passes=1,
                           second_passes=1, refresh_steps=1, past_min_words=0, sample_size=2, stability_threshold=-1.0,
                           shrink_tolerance=100.0, force_conditional=True,
                           csrel_rounds=2, csrel_subset_passes=1, tsr_probe_batches=1)
    return replace(settings, first_passes=epochs, second_passes=epochs) if epochs else settings
