"""Phase 6: G4, a separate English checkpoint whose inputs name the language to write, then X from it.

G4 runs only when G5's guard recovered at least `g4_min_share` of the English drop on validation."""

from ..models.model_seq2seq import load_tokenizer
from ..models.model_mt5 import MT5SmallModel
from .support import brief, phase, revision_arm, score, screening, script_share_of_forgetting

PREFIX = "summarize in {language}: "


@phase(6)
def run(directory, dataset, settings, context):
    share = context["phase1"]["script_share_validation"]
    if not settings.force_conditional and (share is None or share < settings.g4_min_share):
        return {"status": "skipped", "script_share_validation": share,
                "reason": f"G5 recovered {share} of the English drop; G4 needs {settings.g4_min_share}"}, {}
    tagged = dataset.tokenize(load_tokenizer(MT5SmallModel.model_id), prefix=PREFIX)
    english = revision_arm(directory / "english", "english_prefix", settings, tagged, stage="first",
                           language="english", selector="none", prefix="tagged")
    hindi = revision_arm(directory / "hindi_only", "hindi_only_prefix", settings, tagged, english, selector="none")
    plain, x = context["english"]["metrics"], context["hindi_only"]
    task = "english/validation/plain"
    slipped = {name: 100 - score(screening(arm)["metrics"], key, "right_language") for name, arm, key in (
        ("x", x, task), ("x_guarded", x, "english/validation/guarded"), ("x_tagged", hindi, task))}
    details = {"status": "completed", "script_share_validation": share, "prefix": PREFIX,
               "tag_share_of_forgetting": script_share_of_forgetting(score(plain, task), score(screening(x)["metrics"], task),
                                                                    score(hindi["metrics"], task)),
               "english_chrf": {"before": score(plain, task), "x": score(screening(x)["metrics"], task),
                                "x_guarded": score(screening(x)["metrics"], "english/validation/guarded"),
                                "x_tagged": score(hindi["metrics"], task)}, "slipped_percent": slipped,
               "english": english["checkpoint"], "hindi": hindi["checkpoint"], "runs": brief([english, hindi])}
    return details, {"prefix_english": english, "prefix_hindi": hindi, "phase6": details}
