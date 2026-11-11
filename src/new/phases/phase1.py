"""Phase 1: the English checkpoint, pre-registration, Hindi-only X (every Hindi seed), the G5 script share, and the gate."""

from dataclasses import asdict
from datetime import date

from ..datasets.dataset_instruction import LANGUAGES
from ..utils.numerics import write_json
from .serving import benchmark
from .support import (brief, forgetting, gain, gate_decision, lead_baseline, phase, revision_arm, score,
                      script_share_of_forgetting)
from .verdicts import PREDICTIONS


@phase(1)
def run(directory, dataset, settings, context):
    english = revision_arm(directory / "english", "english", settings, dataset, stage="first", language="english",
                           selector="none")
    registration = directory.parent / "preregistration.json"
    if not registration.is_file():  # Settings and predictions, dated before the first Hindi run
        write_json(registration, {"date": date.today().isoformat(), "settings": asdict(settings), "predictions": PREDICTIONS})
    x = revision_arm(directory / "hindi_only", "hindi_only", settings, dataset, english, settings.hindi_seeds,
                     selector="none")
    before, after = english["metrics"], x["metrics"]
    share = {split: script_share_of_forgetting(score(before, f"english/{split}/plain"), score(after, f"english/{split}/plain"),
                                               score(after, f"english/{split}/guarded")) for split in ("test", "validation")}
    lead = {language: lead_baseline(dataset, language) for language in LANGUAGES}
    spread = (x.get("metric_std") or {}).get("english/validation/plain", {}).get("chrf")
    forget, hindi_gain = forgetting(before, x), gain(before, x)
    serving = benchmark(directory / "serving", dataset, settings,
                        {"base": None, "english": english["checkpoint"], "hindi_only": x["checkpoint"]},
                        reference={"english": english["checkpoint"], "hindi": x["checkpoint"]})
    details = {"gate": gate_decision(forget, hindi_gain, score(before, "english/validation/plain"), lead["english"]["chrf"]),
               "forget": forget, "gain": hindi_gain, "lead_1": lead, "x_seed_spread": spread,
               "confirm_revision": spread is not None and spread > settings.max_seed_spread,
               "script_share_of_forgetting": share["test"], "script_share_validation": share["validation"],
               "decision_split": "validation", "serving": serving, "runs": brief([english, x])}
    return details, {"english": english, "hindi_only": x, "phase1": details}
