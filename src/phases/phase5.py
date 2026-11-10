"""Phase 5: validation-based shrinking of A and the overall winner, without training.

Shrinking runs on the screening seed only; the seed-averaged claims stay with the trained adapters."""

from dataclasses import replace
from pathlib import Path

from ..pipelines.generation_shrink import GenerationShrinkPipeline, ShrinkConfig
from ..utils.state import read_json
from .support import phase, run_pipeline
from .artifacts import annotate, save_report
from .serving import benchmark
from .settings import generation_spec


@phase(5)
def run(directory, dataset, model_factory, settings, context, clock):
    winner = context["overall_winner"]
    sources = [context["arms"]["random"]]
    if winner["name"] != "random":
        sources.append(winner)
    results = {}
    for source in sources:
        name = source["name"]
        runs = []
        for original in source.get("seed_runs", [source])[:1]:
            checkpoint = original["checkpoint"]
            metadata = read_json(Path(checkpoint) / "metadata.json")
            rank = int(metadata["rank"]) + max((int(value) for value in metadata.get("new_rank", {}).values()), default=0)
            pipeline = GenerationShrinkPipeline(
                ShrinkConfig(rank, tolerance=settings.shrink_tolerance, grid=settings.shrink_grid,
                             analysis_batch=settings.analysis_batch, generation=generation_spec(settings)),
                checkpoint=checkpoint, state_kind=metadata["state_kind"])
            output = directory / name
            if source.get("seed_runs"):
                output = output / f"seed-{original['seed']}"
            def shrink_fields(pipeline, attempt):
                result = read_json(attempt / "shrink.json")
                chosen = next(row for row in result["table"] if row["rank"] == result["k_G"])
                adapter = str(attempt / result["adapter"])
                return {"checkpoint": adapter, "adapter": adapter, "k_E": result["k_E"], "k_G": result["k_G"],
                        "metrics": {f"{language}/shrink": scores for language, scores in chosen["scores"].items()},
                        "rank_table": result["table"]}
            runs.append(run_pipeline(pipeline, model_factory, dataset, output, "analysis",
                        replace(settings, seed=original["seed"]), 0, original["name"], describe=shrink_fields))
        results[name] = runs[0]
        results[name]["ranks"] = {str(run["seed"]): {"k_E": run["k_E"], "k_G": run["k_G"]} for run in runs}
    checkpoints = {"best_blocks": context["arms"][context["best_blocks"]]["checkpoint"],
                   "winner": winner["checkpoint"], "shrunk_winner": results[sources[-1]["name"]]["checkpoint"]}
    checkpoints.update({name: arm["checkpoint"] for name, arm in context.get("room_arms", {}).items()
                        if name in {"room_placed", "room_spread"}})
    serving = benchmark(directory / "serving", dataset, model_factory, settings, checkpoints)
    details = annotate({"compared": list(results), "ranks": {name: item["ranks"] for name, item in results.items()},
                        "serving": serving,
                        "k_G_differs": len({run["k_G"] for item in results.values()
                                             for run in item.get("seed_runs", [item])}) > 1},
                       clock, list(results.values()), results[sources[-1]["name"]].get("final_model"))
    return save_report(directory, 5, details, {"shrink": results, "phase5": details})
