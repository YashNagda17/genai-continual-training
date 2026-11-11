"""Phase 7: the extensions (reverse order, G1 with G2, G2 without references, frozen E5), the blind manual-check
sheet, and the verdicts on P1-P10, the principles and the headlines."""

from ..pipelines.adapter_room import block_plan, room_plans
from ..utils.numerics import write_json
from .manual_check import rating_sheet
from .support import brief, phase, revision_arm, revision_of, score, screening
from .verdicts import judge

UNSUPPORTED = ({"name": "third_language",
                "reason": "Needs Marathi or Gujarati XL-Sum, and a rank rule for a third stage"},)


def reverse_order(directory, dataset, settings):
    """Hindi first, then English: X (no replay) and A (random Hindi replay)."""
    hindi = revision_arm(directory / "hindi", "reverse_hindi", settings, dataset, stage="first", language="hindi",
                         selector="none")
    arms = {name: revision_arm(directory / name, f"reverse_{name}", settings, dataset, hindi, language="english", **config)
            for name, config in (("english_only", {"selector": "none"}), ("random", {}))}
    before = hindi["metrics"]
    summary = {name: {"hindi_forget": score(before, "hindi/validation/plain") - score(arm["metrics"], "hindi/validation/plain"),
                      "english_gain": score(arm["metrics"], "english/validation/plain") - score(before, "english/validation/plain")}
               for name, arm in arms.items()}
    return summary, [hindi, *arms.values()]


@phase(7)
def run(directory, dataset, settings, context):
    english = context["english"]
    runs = [revision_arm(directory / name, name, settings, dataset, english, selector=selector, replay_text="past",
                         reference_free=free) for name, selector, free in (
                ("drift_past", "drift", False), ("past_without_reference", "random", True))]
    skipped = list(UNSUPPORTED)
    shared = block_plan(context["best_blocks"], settings)
    plans, reason = room_plans(settings, context["maps"], shared, settings.hindi_seeds[0], ("room_placed_frozen",))
    if context.get("room_status") == "completed" and plans:
        runs.append(revision_arm(directory / "room_placed_frozen", "room_placed_frozen", settings, dataset, english,
                                 plan=plans["room_placed_frozen"], **revision_of(context["best_revision"])))
    else:
        skipped.append({"name": "room_placed_frozen", "reason": context.get("room_reason") or reason})
    reverse, reverse_runs = reverse_order(directory / "reverse", dataset, settings)
    blind = {"hindi_only": context["hindi_only"], "random": context["arms"]["random"], "past": context["arms"]["past"],
             context["overall_winner"]["name"]: context["overall_winner"]}
    verdicts = judge(context)
    write_json(directory.parent / "verdicts.json", verdicts)
    details = {"skipped": skipped, "reverse_order": reverse,
               "past_without_reference_filtered": len(screening(runs[1])["filtered"]),
               "manual_check": rating_sheet(directory, dataset, blind), "verdicts": verdicts,
               "runs": brief(runs + reverse_runs)}
    return details, {"drift_past": runs[0], "phase7": details}
