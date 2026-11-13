import copy
from pathlib import Path

import torch

from ..models.model_distillation import CachedDistillation
from ..pipelines.adapter_room import AdapterRoomPipeline
from ..pipelines.generation_shrink import GenerationShrinkPipeline, ShrinkConfig
from ..pipelines.revision import RevisionPipeline, load_pool
from ..utils.numerics import interaction, top_overlap
from ..utils.state import read_json, write_json
from .artifacts import annotate, compare_arms, save_report
from .data import with_prefix
from .serving import benchmark
from .settings import generation_spec
from .support import (
    beats, choose_winner, combined_pipeline, forgetting, gain, gate_decision, hindi_pipeline,
    lead_baseline, lower_forgetting, matched_extra_steps, phase, revision_config,
    room_config, room_plans, run_hindi, run_pipeline, score, script_share_of_forgetting,
)


@phase(1)
def run_phase1(directory, dataset, model_factory, settings, context, clock):
    english_dir = directory / "english"
    english_pipeline = RevisionPipeline(revision_config(settings, stage="english"), pool=dataset.pool_view())
    english_arm = run_pipeline(
        english_pipeline, model_factory, dataset, english_dir, "english", settings, settings.english_passes,
        "english")
    english_checkpoint = english_arm["checkpoint"]
    pool = english_dir / "pinned" / "pool"
    hindi_dir = directory / "hindi_only"
    hindi_arm = run_pipeline(
        RevisionPipeline(revision_config(settings, stage="hindi", selector="none"),
                         pool=load_pool(pool), english_checkpoint=english_checkpoint),
        model_factory, dataset, hindi_dir, "hindi", settings, settings.hindi_passes, "hindi_only",
        selector="none", replay_text="reference", repair_stop=False)
    before = english_arm["metrics"]
    after = hindi_arm["metrics"]
    forget = forgetting(before, hindi_arm)
    hindi_gain = gain(before, hindi_arm)
    share = script_share_of_forgetting(score(before, "english/test/plain"), score(after, "english/test/plain"),
                                       score(after, "english/test/guarded"))
    # G4 is decided on validation, so the test split never chooses an experiment.
    share_validation = script_share_of_forgetting(
        score(before, "english/validation/plain"), score(after, "english/validation/plain"),
        score(after, "english/validation/guarded"))
    lead = {language: lead_baseline(dataset, language) for language in ("english", "hindi")}
    english_score = score(before, "english/validation/plain")
    serving = benchmark(directory / "serving", dataset, model_factory, settings,
        {"base": None, "english": english_checkpoint, "hindi_only": hindi_arm["checkpoint"]},
        reference={"english": english_checkpoint, "hindi": hindi_arm["checkpoint"]})
    details = annotate(
        {"gate": gate_decision(forget, hindi_gain, english_score, lead["english"]["chrf"]),
         "forget": forget, "gain": hindi_gain, "lead_1": lead, "english_validation_chrf": english_score,
         "script_share_of_forgetting": share, "script_share_validation": share_validation,
         "decision_split": "validation", "serving": serving},
        clock, [english_arm, hindi_arm], hindi_arm.get("final_model"))
    update = {"english_checkpoint": str(english_checkpoint), "pool": str(pool),
              "english_metrics": before, "hindi_only": hindi_arm, "phase1": details}
    return save_report(directory, 1, details, update)


@phase(2)
def run_phase2(directory, dataset, model_factory, settings, context, clock):
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    specs = (
        ("random", dict(selector="random")),
        ("loss", dict(selector="loss")),
        ("drift", dict(selector="drift")),
        ("past", dict(selector="random", replay_text="past")),
    )
    arms = {}
    for name, overrides in specs:
        arms[name] = run_hindi(directory / name, name, hindi_pipeline(settings, pool, checkpoint, **overrides),
                              dataset, model_factory, settings)
    parent = lower_forgetting(context["english_metrics"], [arms["loss"], arms["drift"]])
    repair = run_hindi(directory / "repair_stop", "repair_stop",
                      hindi_pipeline(settings, pool, checkpoint, selector=parent["selector"], repair_stop=True),
                      dataset, model_factory, settings, selector=parent["selector"], replay_text="reference",
                      repair_stop=True, parent=parent["name"])
    arms["repair_stop"] = repair
    loss, drift = arms["loss"], arms["drift"]
    width = min(100, len(loss["ranking"]), len(drift["ranking"]))
    overlap = top_overlap(loss["ranking"], drift["ranking"], width) if width else None
    details = annotate(
        {"g3_parent": parent["name"], "top_overlap": overlap,
         "filter_drops": len(arms["past"]["filtered"]), "retired": len(arms["repair_stop"]["retired"]),
         "arms": list(arms), "comparisons": [compare_arms(arms[a], arms[b]) for a, b in (
             ("loss", "random"), ("drift", "loss"), ("past", "random"), ("repair_stop", parent["name"]))]},
        clock, list(arms.values()), arms[parent["name"]].get("final_model"))
    return save_report(directory, 2, details, {"arms": arms, "g3_parent": parent["name"], "phase2": details})


@phase(3)
def run_phase3(directory, dataset, model_factory, settings, context, clock):
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
        return score(arm["metrics"], "english/validation/plain")
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


def _room_runner(directory, dataset, model_factory, settings, context, pool, checkpoint):
    revision = context["best_revision"]

    def run(plan):
        name = plan.identity["arm"]
        pipeline = hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                                  replay_text=revision["replay_text"], repair_stop=revision["repair_stop"])
        pipeline.adapter_plan = plan

        return run_hindi(directory / name, name, pipeline, dataset, model_factory, settings,
                         selector=revision["selector"], replay_text=revision["replay_text"],
                         repair_stop=revision["repair_stop"])
    return run


def _frozen(directory, dataset, model_factory, settings, context, pool, checkpoint):
    """Placed extra room with shared adapters frozen in the chosen blocks."""
    plans, reason = room_plans(settings, context, names=("room_placed_frozen",))
    chosen = next((plan for plan in plans if plan.identity["arm"] == "room_placed_frozen"), None)
    if chosen is None:
        return None, reason
    revision = context["best_revision"]
    pipeline = hindi_pipeline(settings, pool, checkpoint, selector=revision["selector"],
                              replay_text=revision["replay_text"], repair_stop=revision["repair_stop"])
    pipeline.adapter_plan = chosen
    arm = run_hindi(directory / "room_placed_frozen", "room_placed_frozen", pipeline, dataset, model_factory,
                    settings, selector=revision["selector"], replay_text=revision["replay_text"],
                    repair_stop=revision["repair_stop"])
    return arm, None


@phase(4)
def run_phase4(directory, dataset, model_factory, settings, context, clock):
    pool = load_pool(context["pool"])
    checkpoint = context["english_checkpoint"]
    before = context["english_metrics"]
    plans, reason = room_plans(settings, context)
    if not plans:
        status = "skipped" if not context["maps"]["stable"] or not any(
            value < 0 for value in context["maps"]["blocks"].values()) else "rejected"
        winner = context["combined"]
        final = winner["checkpoint"]
        details = annotate({"status": status, "reason": reason, "overall_winner": winner["name"]},
                           clock, [winner], final)
        update = {"room_status": status, "room_reason": reason, "room_arms": {},
                  "overall_winner": winner, "phase4": details}
        return save_report(directory, 4, details, update)
    run_room = _room_runner(directory, dataset, model_factory, settings, context, pool, checkpoint)
    by_name = {plan.identity["arm"]: plan for plan in plans}
    trained = {name: run_room(plan) for name, plan in by_name.items()}
    placement_wins = beats(before, trained["room_placed"], [trained["room_random"], trained["room_spread"]])
    e4_reason = None
    if placement_wins or settings.force_conditional:
        slow, e4_reason = room_plans(settings, context, names=("room_placed_slow",))
        if slow:
            by_name["room_placed_slow"] = slow[0]
            trained["room_placed_slow"] = run_room(slow[0])
    else:
        e4_reason = "room_placed did not beat room_random and room_spread by 1 point"
    winner = choose_winner(before, [context["combined"], *trained.values()], context["arms"]["random"])
    if winner["name"] in trained:
        trained[winner["name"]] = winner
    frozen, frozen_reason = _frozen(directory, dataset, model_factory, settings, context, pool, checkpoint)
    comparisons = [compare_arms(trained["room_placed"], trained[other]) for other in ("room_random", "room_spread")]
    if winner["name"] != "random":
        comparisons.append(compare_arms(winner, context["arms"]["random"]))
    details = {"status": "completed", "arms": list(trained), "overall_winner": winner["name"],
               "placement_wins": placement_wins, "room_placed_slow": e4_reason or "ran",
               "comparisons": comparisons, "placement_counts": any(
                   result["counts"] for result in comparisons[1]["metrics"].values()),
               "room_placed_frozen": None if frozen is None else frozen["attempt"],
               "room_placed_frozen_reason": frozen_reason}
    final = winner["checkpoint"]
    details = annotate(details, clock, list(trained.values()) + ([winner] if winner["name"] not in trained else [])
                       + ([frozen] if frozen is not None else []), final)
    return save_report(directory, 4, details, {"room_status": "completed", "room_reason": "",
                                               "room_arms": trained, "overall_winner": winner,
                                               "room_placed_frozen": frozen, "phase4": details})


@phase(5)
def run_phase5(directory, dataset, model_factory, settings, context, clock):
    winner = context["overall_winner"]
    sources = [context["arms"]["random"]]
    if winner["name"] != "random":
        sources.append(winner)
    results = {}
    for source in sources:
        name = source["name"]
        checkpoint = source["checkpoint"]
        metadata = read_json(Path(checkpoint) / "metadata.json")
        rank = int(metadata["rank"]) + max((int(value) for value in metadata.get("new_rank", {}).values()), default=0)
        pipeline = GenerationShrinkPipeline(
            ShrinkConfig(rank, tolerance=settings.shrink_tolerance, grid=settings.shrink_grid,
                         analysis_batch=settings.analysis_batch, generation=generation_spec(settings)),
            checkpoint=checkpoint, state_kind=metadata["state_kind"])
        output = directory / name
        def shrink_fields(pipeline, attempt):
            result = read_json(attempt / "shrink.json")
            chosen = next(row for row in result["table"] if row["rank"] == result["k_G"])
            adapter = str(attempt / result["adapter"])
            return {"checkpoint": adapter, "adapter": adapter, "k_E": result["k_E"], "k_G": result["k_G"],
                    "metrics": {f"{language}/shrink": scores for language, scores in chosen["scores"].items()},
                    "rank_table": result["table"]}
        results[name] = run_pipeline(pipeline, model_factory, dataset, output, "analysis",
                    settings, 0, name, describe=shrink_fields)
        results[name]["ranks"] = {"k_E": results[name]["k_E"], "k_G": results[name]["k_G"]}
    checkpoints = {"best_blocks": context["arms"][context["best_blocks"]]["checkpoint"],
                   "winner": winner["checkpoint"], "shrunk_winner": results[sources[-1]["name"]]["checkpoint"]}
    checkpoints.update({name: arm["checkpoint"] for name, arm in context.get("room_arms", {}).items()
                        if name in {"room_placed", "room_spread"}})
    serving = benchmark(directory / "serving", dataset, model_factory, settings, checkpoints)
    details = annotate({"compared": list(results), "ranks": {name: item["ranks"] for name, item in results.items()},
                        "serving": serving,
                        "k_G_differs": len({item["k_G"] for item in results.values()}) > 1},
                       clock, list(results.values()), results[sources[-1]["name"]].get("final_model"))
    return save_report(directory, 5, details, {"shrink": results, "phase5": details})


PREFIX = "summarize in {language}: "


@phase(6)
def run_phase6(directory, dataset, model_factory, settings, context, clock):
    share = context["phase1"].get("script_share_validation")
    if not settings.force_conditional and (share is None or share < settings.g4_min_share):
        # Reuse the core experiment's winner when the prompt comparison is skipped.
        winner = context.get("overall_winner")
        final = None if winner is None else (
            winner["checkpoint"])
        details = annotate({"status": "skipped", "prefix": PREFIX, "script_share_validation": share,
                            "reason": f"G5 recovered {share} of the English drop; G4 needs {settings.g4_min_share}"},
                           clock, [], final)
        return save_report(directory, 6, details, {"phase6": details})
    token_model = model_factory()
    prefixed = with_prefix(dataset, token_model, PREFIX)
    token_model.release()
    english_dir = directory / "english"
    english_arm = run_pipeline(
        RevisionPipeline(revision_config(settings, stage="english", prefix="tagged"), pool=prefixed.pool_view()),
        model_factory, prefixed, english_dir, "english", settings, settings.english_passes, "english_prefix", selector="none",
        replay_text="reference", repair_stop=False)
    checkpoint = english_arm["checkpoint"]
    hindi = run_hindi(
        directory / "hindi_only", "hindi_only_prefix",
        RevisionPipeline(revision_config(settings, stage="hindi", selector="none"),
                         pool=load_pool(english_dir / "pinned" / "pool"), english_checkpoint=checkpoint),
        prefixed, model_factory, settings, selector="none")
    details = annotate(
        {"status": "completed", "script_share_validation": share, "prefix": PREFIX, "stored_prefix": prefixed.manifest["preprocessing"]["prefix"],
         "english": english_arm["checkpoint"], "hindi": hindi["checkpoint"]},
        clock, [english_arm, hindi], hindi.get("final_model"))
    return save_report(directory, 6, details, {"prefix_english": english_arm, "prefix_hindi": hindi, "phase6": details})


@phase(7)
def run_phase7(directory, dataset, model_factory, settings, context, clock):
    from ..distillation_pilot import main as run_distillation
    phase_data = copy.copy(dataset)
    phase_data.views, phase_data.manifest = dict(dataset.views), copy.deepcopy(dataset.manifest)
    for language in ('english', 'hindi'):
        key = f'{language}/validation'
        if key not in phase_data.views:
            phase_data.expose(key, [r.sample_id for r in dataset.views['validation'] if r.task_id == language])
    prepared = directory / 'prepared'
    phase_data.save(prepared)
    template = model_factory()
    previous_threads = torch.get_num_threads()
    try:
        template.tokenizer  # Resolve configuration without loading another resident model.

        class PhaseKD(CachedDistillation, type(template)):
            def __init__(self, numerical_settings, **kwargs):
                kwargs.update(config=copy.deepcopy(template.config), tokenizer=template.tokenizer,
                    model_factory=template.model_factory, pretrained=template.pretrained,
                    revision=template.revision, cache_dir=template.cache_dir,
                    gradient_checkpointing=template.gradient_checkpointing)
                super().__init__(numerical_settings, **kwargs)

        counts = [len(phase_data.views[f'{language}/validation']) for language in ('english', 'hindi')]
        if not counts[0] or counts[0] != counts[1]:
            raise ValueError('Phase 7 requires equal, nonempty English/Hindi validation views')
        argv = ['--prepared', str(prepared), '--checkpoint', context['english_checkpoint'],
                '--output', str(directory / 'kd'), '--epochs', str(settings.hindi_passes),
                '--hindi-batch', str(settings.batch_size), '--replay-batch', str(settings.replay_per_batch),
                '--eval-examples', str(counts[0]), '--max-new-tokens', str(settings.max_new_tokens),
                '--learning-rate', str(settings.learning_rate), '--device', str(template.device),
                '--threads', str(previous_threads), '--seed', str(settings.seed)]
        if template.gradient_checkpointing:
            argv.append('--gradient-checkpointing')
        report = run_distillation(argv, model_class=PhaseKD)
    finally:
        template.release()
        torch.set_num_threads(previous_threads)
    runs = []
    for arm, value in report['arms'].items():
        name = arm
        final = value['checkpoint']
        runs.append({'name': name, 'seed': settings.seed, 'status': 'completed', 'steps': value['event'],
            'epochs': {'requested': settings.hindi_passes, 'completed': settings.hindi_passes, 'steps': value['event']},
            'accuracies': {f'{language}/validation/plain': data['metrics'] for language, data in value['after'].items()},
            'costs': {'update_seconds': value['training_seconds']}, 'checkpoint': value['checkpoint'],
            'final_model': final})
    final = next(r['final_model'] for r in runs if r['name'] == 'replay_kd')
    details = annotate({'method': 'cached_teacher_kd', 'comparison': 'replay_kd_minus_replay',
                        'epochs_per_arm': settings.hindi_passes, 'aggregate': report['aggregate'],
                        'kd_report': str(directory / 'kd/report.json'), 'decision_split': 'validation'},
                       clock, runs, final)
    return save_report(directory, 7, details, {'kd_distillation': {
        'report': details['kd_report'], 'aggregate': report['aggregate']}, 'phase7': details})


PHASES = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
    7: run_phase7,
}
