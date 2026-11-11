"""P1-P10, written down before the first Hindi run and judged after the last; then the principle check and headlines.

Decisions use validation on the screening seed, like the phases. P7-P10 use the test-split bootstrap comparisons."""

from pathlib import Path
import statistics

from .support import beats, forgetting, gain, screening

PREDICTIONS = {
    "P1": "Late-only is faster and lighter than all blocks; early + late saves little",
    "P2": "B's scoring adds under 10% to the update time",
    "P3": "C1's maps are stable, clash in the decoder, and predict C2",
    "P4": "Merged serving is as fast as the base model; unmerged is slower",
    "P5": "F cuts storage and keeps both languages",
    "P6": "One merged adapter uses half the reference's storage, at some cost in English",
    "P7": "G1 and B's top 100 overlap by less than half",
    "P8": "G5 recovers at least a quarter of the English drop",
    "P9": "G2 keeps English within 1 point of A",
    "P10": "G3 retires a third of examples early, with no English loss",
}


def settle(function):
    """(supported, evidence), or None when the phase that decides it did not run."""
    try:
        supported, evidence = function()
        return {"supported": bool(supported), "evidence": evidence}
    except (KeyError, TypeError, IndexError, ZeroDivisionError, StopIteration):
        return None


def judge(context):
    arms, before = context.get("arms", {}), context["english"]["metrics"]

    def seconds(name):
        return screening(arms[name])["costs"]["update_seconds"]

    def english_delta(phase, left, right):
        found = next(c for c in context[phase]["comparisons"] if (c["left"], c["right"]) == (left, right))
        return found["metrics"]["english/chrf"]

    def serving(key):
        return statistics.mean(row["median_seconds"] for row in context["phase1"]["serving"][key].values())

    def adapter_bytes(arm):
        return (Path(screening(arm)["checkpoint"]) / "weights.pt").stat().st_size

    def late_matches():
        late, every = screening(arms["late"]), screening(arms["random"])
        return abs(forgetting(before, late) - forgetting(before, every)) <= 1 and gain(before, late) >= gain(before, every) - 1

    def p1():
        late, early_late = seconds("late") / seconds("random"), seconds("early_late") / seconds("random")
        peaks = [screening(arms[n])["peak_memory_mb"] for n in ("late", "random")]
        lighter = None if None in peaks else peaks[0] < peaks[1]
        return late < 0.9 and early_late > 0.9 and lighter is not False, {
            "late/all_time": late, "early_late/all_time": early_late, "late_lighter": lighter}

    def p3():
        prediction = context["phase3"]["prediction"]
        kept = prediction["keep_english"]
        other = "early_late" if kept == "late" else "late"
        correct = forgetting(before, screening(arms[kept])) <= forgetting(before, screening(arms[other]))
        return prediction["stable"] and prediction["clash_in_decoder"] and correct, {**prediction, "prediction_correct": correct}

    def p4():
        base, merged, unmerged = serving("base/merged"), serving("english/merged"), serving("english/unmerged")
        return merged <= 1.05 * base and unmerged > 1.05 * merged, {
            "merged/base": merged / base, "unmerged/merged": unmerged / merged}

    def p5():
        ranks = {name: {k: r[k] for k in ("k_E", "k_G", "full_rank")} for name, r in context["shrink"].items()}
        return all(r["k_G"] < r["full_rank"] for r in ranks.values()), ranks

    def p6():
        winner = context["overall_winner"]
        ratio = adapter_bytes(winner) / (adapter_bytes(context["english"]) + adapter_bytes(context["hindi_only"]))
        return ratio <= 0.55, {"storage_ratio": ratio, "english_cost": forgetting(before, screening(winner))}

    def p10():
        delta = english_delta("phase2", "repair_stop", context["phase2"]["g3_parent"])
        share = context["phase2"]["retired_share"]
        return share >= 1 / 3 and delta["delta"] >= -1, {"retired_share": share, "english_delta": delta["delta"]}

    def winner_ranks():
        name = context["overall_winner"]["name"]
        ranks = context["shrink"][name if name in context["shrink"] else "random"]
        return ranks["k_G"] <= ranks["full_rank"] / 2 and ranks["k_G"] < ranks["k_E"], ranks

    loss = arms.get("loss", {}).get("costs", {})
    predictions = {
        "P1": settle(p1),
        "P2": settle(lambda: (loss["analysis_seconds"].get("pool_scoring", 0) / loss["update_seconds"] < 0.1,
                              {"scoring/update": loss["analysis_seconds"].get("pool_scoring", 0) / loss["update_seconds"]})),
        "P3": settle(p3),
        "P4": settle(p4),
        "P5": settle(p5),
        "P6": settle(p6),
        "P7": settle(lambda: (context["phase2"]["top_overlap"] < 0.5, {"top_overlap": context["phase2"]["top_overlap"]})),
        "P8": settle(lambda: (context["phase1"]["script_share_of_forgetting"] >= 0.25,
                              {"script_share": context["phase1"]["script_share_of_forgetting"]})),
        "P9": settle(lambda: (english_delta("phase2", "past", "random")["delta"] >= -1,
                              {"english_delta": english_delta("phase2", "past", "random")["delta"]})),
        "P10": settle(p10),
    }

    def holds(key):
        return (predictions[key] or {}).get("supported")

    selectors_win = settle(lambda: (any(beats(before, arms[n], [arms["random"]]) for n in ("loss", "drift")), {}))
    any_beats_random = settle(lambda: (any(beats(before, arms[n], [arms["random"]])
                                           for n in ("loss", "drift", "past", "repair_stop")), {}))
    g1_or_g2 = settle(lambda: (beats(before, arms["drift"], [arms.get("random_compute_matched", arms["random"])])
                               or beats(before, arms["past"], [arms["random"]]), {}))
    placement = context.get("phase4", {}).get("placement_wins")
    late = settle(lambda: (late_matches(), {}))
    small_winner = settle(winner_ranks)
    narrow = settle(lambda: (any(r["k_G"] <= 0.75 * r["k_E"] for r in context["shrink"].values()), {}))
    shares = [s for s in (context.get("phase1", {}).get("script_share_of_forgetting"),
                          context.get("phase6", {}).get("tag_share_of_forgetting")) if s is not None]

    def value(item):
        return None if item is None else item["supported"]
    principles = {
        "M": {"late_matches_all": value(late), "c1_predicted": holds("P3"), "e1_beats_e2_e3": placement},
        "N": {"b_or_g1_beats_random": value(selectors_win), "g2_matches": holds("P9"), "g3_frees_slots": holds("P10")},
        "K": {"early_late_costs_like_all": None if holds("P1") is None else predictions["P1"]["evidence"]["early_late/all_time"] > 0.9,
              "merging_removes_serving_cost": holds("P4")},
        "L": {"k_G_well_below_training_rank_and_energy": value(small_winner)},
        "P": {"guard_or_tag_recovers_a_quarter": any(s >= 0.25 for s in shares) if shares else None},
    }
    for checks in principles.values():
        known = [v for v in checks.values() if v is not None]
        checks["supported"] = all(known) if known else None
    headlines = {
        "A summarizer that reads its own writing revises better": value(g1_or_g2),
        "Most forgetting here is the wrong language, and it's nearly free to fix": any(s > 0.5 for s in shares) if shares else None,
        "Cheap random replay is enough": None if any_beats_random is None else not any_beats_random["supported"],
        "One gradient pass finds where a new language needs room": placement,
        "Learn wide, remember narrow": value(narrow),
        "Update the late blocks only: same quality, less cost": value(late) and holds("P1"),
        "LoRA adds Hindi without forgetting English": context["phase1"]["gate"] == "stop_replay",
    }
    def against(name):
        """Does our best revision, or D, forget at least 1 point less English than the baseline?"""
        ours = {"best_revision": arms[context["best_revision"]["name"]], "combined": context["combined"]}
        return {key: {"beats": beats(before, arm, [arms[name]]), "forget": forgetting(before, screening(arm)),
                      "baseline_forget": forgetting(before, screening(arms[name])),
                      "hindi_gain": gain(before, screening(arm)), "baseline_hindi_gain": gain(before, screening(arms[name]))}
                for key, arm in ours.items()}
    baselines = {name: settle(lambda name=name: (any(v["beats"] for v in against(name).values()), against(name)))
                 for name in ("csrel", "tsr")}
    baselines["csrel_writer_beats_random_memory"] = settle(lambda: (beats(before, arms["csrel"], [arms["csrel_random"]]), {}))
    return {"baselines": baselines, "predictions": {key: {"text": PREDICTIONS[key], "result": result} for key, result in predictions.items()},
            "principles": principles, "headlines": [text for text, ok in headlines.items() if ok]}
