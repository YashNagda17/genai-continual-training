"""Phase 3: C1 location maps and a written prediction, C2 block plans, D, and the full-English reference.

The best revision gets every Hindi seed only when X's seed spread is too wide. G1's compute-matched
control runs only when G1 is the best revision."""

from ..pipelines.adapter_room import AdapterRoomPipeline, block_plan
from ..pipelines.revision import load_pool
from ..utils.numerics import read_json, write_json
from .phase2 import baselines
from .support import (brief, choose_winner, compare_arms, gain, matched_extra_steps, phase, revision_arm, revision_of,
                      run_arm, score, screening)


@phase(3)
def run(directory, dataset, settings, context):
    english, arms = context["english"], dict(context["arms"])
    before = english["metrics"]
    maps = run_arm("maps", directory / "maps", settings, lambda model, output, seed: AdapterRoomPipeline(
        settings, model, dataset, output, load_pool(english["pool"]), dataset.view("hindi/train"),
        english["checkpoint"], context["hindi_only"]["checkpoint"], seed))["maps"]
    path = directory / "c1_prediction.json"
    if not path.is_file():  # Written before C2 trains
        decoder = sum(v < 0 for k, v in maps["blocks"].items() if k.startswith("decoder"))
        encoder = sum(v < 0 for k, v in maps["blocks"].items() if k.startswith("encoder"))
        write_json(path, {"keep_english": "late" if decoder >= encoder else "early_late", "learn_hindi": "all",
                          "stable": maps["stable"], "clash_in_decoder": decoder >= encoder})
    for name in ("late", "early_late"):
        arms[name] = revision_arm(directory / name, name, settings, dataset, english, plan=block_plan(name, settings))
    best_blocks = choose_winner(before, [arms["random"], arms["late"], arms["early_late"]], arms["random"])
    best_revision = choose_winner(before, [arms[n] for n in ("loss", "drift", "past", "repair_stop")], arms["random"])
    conditional = {}
    if context["phase1"]["confirm_revision"] or settings.force_conditional:
        name = best_revision["name"]
        best_revision = arms[name] = revision_arm(directory.parent / "phase2" / name, name, settings, dataset, english,
                                                  settings.hindi_seeds, **revision_of(best_revision))
        conditional["confirmed_revision"] = name
        arms.update(baselines(directory.parent / "phase2", dataset, settings, english, settings.hindi_seeds)[0])
    if best_revision["name"] == "drift" or settings.force_conditional:
        extra = matched_extra_steps(arms["random"], arms["drift"])
        arms["random_compute_matched"] = revision_arm(directory / "random_compute_matched", "random_compute_matched",
                                                      settings, dataset, english, extra_random_steps=extra)
        conditional["compute_matched_extra_steps"] = extra
    arms["combined"] = revision_arm(directory / "combined", "combined", settings, dataset, english,
                                    plan=block_plan(best_blocks["blocks"], settings), **revision_of(best_revision))
    arms["full_english"] = revision_arm(directory / "full_english", "full_english", settings, dataset, english,
                                        selector="full")

    def s(arm):
        return score(screening(arm)["metrics"], "english/validation/plain")
    frontier = [{"arm": name, "english_kept": s(arm), "hindi_gain": gain(before, screening(arm)),
                 "update_seconds": screening(arm)["costs"]["update_seconds"],
                 "total_seconds": screening(arm)["costs"]["update_seconds"] + sum(screening(arm)["costs"]["analysis_seconds"].values())
                 + arm.get("selection_seconds", 0)}
                for name, arm in arms.items()]
    write_json(directory / "frontier.json", frontier)  # English kept and Hindi gained against update time
    comparisons = [("late", "early_late"), ("combined", "random"), ("combined", best_revision["name"]),
                   (best_revision["name"], "csrel"), (best_revision["name"], "tsr"), ("combined", "csrel"), ("combined", "tsr")]
    comparisons += [("drift", "random_compute_matched")] if "random_compute_matched" in arms else []
    trained = [arms[n] for n in ("late", "early_late", "combined", "full_english", "random_compute_matched") if n in arms]
    details = {"prediction": read_json(path), "best_blocks": best_blocks["blocks"], "best_revision": best_revision["name"],
               "interaction": s(arms["combined"]) - s(best_revision) - s(best_blocks) + s(arms["random"]),
               "frontier": frontier, "maps_stable": maps["stable"], "map_correlation": maps["correlation"], "conditional": conditional,
               "comparisons": [compare_arms(arms[a], arms[b]) for a, b in comparisons], "runs": brief(trained)}
    return details, {"maps": maps, "best_blocks": best_blocks["blocks"],
                     "best_revision": {"name": best_revision["name"], **revision_of(best_revision)},
                     "combined": arms["combined"], "arms": arms, "phase3": details}
