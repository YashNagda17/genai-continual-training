"""Phase 3: location maps, block plans, and the combined run.

Arms are screened on one seed. The best revision gets the other seeds only when X's seed spread is too wide,
and G1's compute-matched control runs only when G1 is the best revision."""

from ..pipelines.adapter_room import AdapterRoomPipeline
from ..utils.numerics import interaction
from ..utils.state import read_json, write_json
from .support import (combined_pipeline, hindi_pipeline, matched_extra_steps, revision_pipeline, run_hindi, room_config,
                      choose_winner, phase, run_pipeline, score, screening)
from ..pipelines.revision import load_pool
from .artifacts import annotate, compare_arms, save_report


@phase(3)
def run(directory, dataset, model_factory, settings, context, clock):
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    room = AdapterRoomPipeline(
        room_config(settings), pool=pool, hindi=dataset.training_view("hindi"),
        english_checkpoint=checkpoint, hindi_checkpoint=context["hindi_only"]["checkpoint"])
    maps_record = run_pipeline(
        room, model_factory, dataset, directory / "maps", "analysis", settings, 0, "maps",
        describe=lambda pipeline, attempt: {"maps": {key: value for key, value in read_json(
            attempt / "location_maps.json").items() if key != "samples"}})
    maps = maps_record["maps"]
    prediction_path = directory / "c1_prediction.json"
    if not prediction_path.is_file():
        blocks = maps["blocks"]
        decoder = [value for key, value in blocks.items() if key.startswith("decoder")]
        encoder = [value for key, value in blocks.items() if key.startswith("encoder")]
        clash_in_decoder = sum(value < 0 for value in decoder) >= sum(value < 0 for value in encoder)
        write_json(prediction_path, {"keep_english": "late" if clash_in_decoder else "early_late",
                                     "learn_hindi": "all", "stable": maps["stable"],
                                     "clash_in_decoder": clash_in_decoder})
    prediction = read_json(prediction_path)
    arms = dict(context["arms"])
    for name in ("late", "early_late"):
        pipeline = hindi_pipeline(settings, pool, checkpoint, selector="random")
        pipeline.adapter_plan = room.adapter_plan(name)
        arms[name] = run_hindi(directory / name, name, pipeline, dataset, model_factory, settings,
                              selector="random", replay_text="reference", repair_stop=False, blocks=name)
    before = context["english_metrics"]
    best_blocks = choose_winner(before, [arms["random"], arms["late"], arms["early_late"]], arms["random"])
    revision_names = ("loss", "drift", "past", "repair_stop")
    best_revision = choose_winner(before, [arms[name] for name in revision_names], arms["random"])
    conditional = {}
    if context["phase1"].get("confirm_revision") or settings.force_conditional:
        name = best_revision["name"]
        arms[name] = run_hindi(directory.parent / "phase2" / name, name,
                               revision_pipeline(settings, pool, checkpoint, best_revision), dataset, model_factory,
                               settings, seeds=settings.hindi_seeds, selector=best_revision["selector"],
                               replay_text=best_revision["replay_text"], repair_stop=best_revision["repair_stop"])
        best_revision = arms[name]
        conditional["confirmed_revision"] = name
    if best_revision["name"] == "drift" or settings.force_conditional:
        extra = matched_extra_steps(arms["random"], arms["drift"])
        arms["random_compute_matched"] = run_hindi(
            directory / "random_compute_matched", "random_compute_matched",
            hindi_pipeline(settings, pool, checkpoint, selector="random", extra_random_steps=extra),
            dataset, model_factory, settings, selector="random", replay_text="reference", repair_stop=False,
            extra_random_steps=extra)
        conditional["compute_matched_extra_steps"] = extra
    arms["combined"] = run_hindi(
        directory / "combined", "combined", combined_pipeline(settings, pool, checkpoint, best_revision,
                                                             best_blocks["name"]), dataset, model_factory, settings,
        selector=best_revision["selector"], replay_text=best_revision["replay_text"],
        repair_stop=best_revision["repair_stop"], blocks=best_blocks["name"])
    arms["full_english"] = run_hindi(
        directory / "full_english", "full_english", hindi_pipeline(settings, pool, checkpoint, selector="full"),
        dataset, model_factory, settings, selector="full", replay_text="reference", repair_stop=False)
    def english(arm):
        return score(screening(arm)["metrics"], "english/validation/plain")
    interaction_value = interaction(english(arms["random"]), english(best_revision), english(best_blocks),
                                    english(arms["combined"]))
    trained = [maps_record, arms["late"], arms["early_late"], arms["combined"], arms["full_english"]]
    trained += [arms[name] for name in ("random_compute_matched",) if name in arms]
    comparisons = [("late", "early_late"), ("combined", "random"), ("combined", best_revision["name"])]
    if "random_compute_matched" in arms:
        comparisons.append(("drift", "random_compute_matched"))
    details = annotate(
        {"prediction": prediction, "best_blocks": best_blocks["name"],
         "best_revision": best_revision["name"], "interaction": interaction_value,
         "maps_stable": maps["stable"], "conditional": conditional,
         "comparisons": [compare_arms(arms[a], arms[b]) for a, b in comparisons]},
        clock, trained, arms["combined"].get("final_model"))
    kept = maps
    update = {"maps": kept, "best_blocks": best_blocks["name"],
              "best_revision": {key: best_revision[key] for key in ("name", "selector", "replay_text", "repair_stop")},
              "combined": arms["combined"], "arms": arms, "phase3": details}
    return save_report(directory, 3, details, update)
