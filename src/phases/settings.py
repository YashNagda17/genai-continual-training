"""Protocol settings and scientific pre-registration validation."""

from dataclasses import dataclass, replace
from pathlib import Path

from ..utils.state import read_json


@dataclass(frozen=True)
class PhaseSettings:
    protocol: str = "smoke"
    seed: int = 0
    hindi_seeds: tuple = (1, 2, 3)
    preregistration_hash: str = ""
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


WINNER_RULES = {"g3_parent": "lowest_english_validation_forgetting",
                "best_revision": "hindi_margin_then_forgetting_then_update_time",
                "best_blocks": "hindi_margin_then_forgetting_then_update_time",
                "overall_winner": "hindi_margin_then_forgetting_then_update_time"}
SCIENTIFIC_FIXED = {"model": "google/mt5-small", "rank": 16, "alpha": 32,
    "max_source_length": 256, "max_target_length": 64, "batch_size": 16, "replay_per_batch": 2,
    "seed": 0, "hindi_seeds": [1, 2, 3], "encoder_blocks": 8, "decoder_blocks": 8,
    "sample_size": 64, "max_new_tokens": 64, "shrink_tolerance": 1.0, "shrink_grid": [12, 8, 6, 4],
    "new_room_budget": 638976, "train_rows": 2000, "pool_rows": 500, "validation_rows": 200, "test_rows": 500,
    "top_fraction": 0.5, "warmup_fraction": 0.1, "past_filter_fraction": 0.2, "repair_beta": 1,
    "stop_similarity": 0.9, "refresh_steps": 200, "gamma": 1, "floor": 0.25, "energy_fraction": 0.9}


def registered_settings(path):
    """Validate the locked protocol before model creation or dataset loading."""
    from datetime import date
    import hashlib
    path = Path(path)
    if not path.is_file():
        raise ValueError("Scientific runs require preregistration.json")
    registration = read_json(path)
    for key in ("date", "settings", "predictions", "winner_rules", "gate_thresholds"):
        if key not in registration:
            raise ValueError(f"Pre-registration is missing {key}")
    date.fromisoformat(registration["date"])
    values = registration["settings"]
    for key, expected in SCIENTIFIC_FIXED.items():
        if values.get(key) != expected:
            raise ValueError(f"Pre-registered {key} must be {expected!r}")
    if registration["winner_rules"] != WINNER_RULES or registration["gate_thresholds"] != {
            "meaningful_difference": 1, "hindi_margin": 1, "shrink_tolerance": 1}:
        raise ValueError("Winner rules or gate thresholds differ from the approved experiment")
    predictions = registration["predictions"]
    if any(not isinstance(predictions.get(f"P{i}"), str) or not predictions[f"P{i}"].strip() for i in range(1, 11)):
        raise ValueError("Pre-register predictions P1 through P10")
    for key in ("english_passes", "hindi_passes"):
        if type(values.get(key)) is not int or values[key] < 1:
            raise ValueError(f"The pilot must fix positive {key}")
    if not isinstance(values.get("stability_threshold"), (int, float)) or not -1 <= values["stability_threshold"] <= 1:
        raise ValueError("Pre-register the C1 stability threshold")
    fields = {key: value for key, value in values.items() if key in PhaseSettings.__dataclass_fields__
              and key not in {"protocol", "preregistration_hash"}}
    fields["hindi_seeds"] = tuple(fields["hindi_seeds"])
    fields["shrink_grid"] = tuple(fields["shrink_grid"])
    return PhaseSettings(protocol="scientific", **fields,
                         preregistration_hash=hashlib.sha256(path.read_bytes()).hexdigest())
