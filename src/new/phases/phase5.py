"""Phase 5: shrink A and the overall winner on validation, without training, then compare serving."""

from pathlib import Path

from ..pipelines.generation_shrink import GenerationShrinkPipeline
from ..utils.numerics import read_json
from .serving import benchmark
from .support import brief, phase, run_arm, screening


@phase(5)
def run(directory, dataset, settings, context):
    winner, arms = context["overall_winner"], context["arms"]
    results = {}
    for source in [arms["random"]] + ([winner] if winner["name"] != "random" else []):
        checkpoint = screening(source)["checkpoint"]
        metadata = read_json(Path(checkpoint) / "metadata.json")
        full_rank = metadata["rank"] + max(metadata["new_rank"].values(), default=0)
        results[source["name"]] = run_arm(source["name"], directory / source["name"], settings,
                                          lambda model, output, seed: GenerationShrinkPipeline(
                                              settings, model, dataset, output, checkpoint, full_rank, seed))
    shrunk = results[winner["name"] if winner["name"] in results else "random"]
    checkpoints = {"best_blocks": screening(arms[{"all": "random"}.get(context["best_blocks"], context["best_blocks"])])["checkpoint"],
                   "winner": screening(winner)["checkpoint"], "shrunk_winner": shrunk["checkpoint"]}
    checkpoints.update({name: screening(arm)["checkpoint"] for name, arm in context.get("room_arms", {}).items()
                        if name in ("room_placed", "room_spread")})
    details = {"ranks": {name: {"k_E": r["k_E"], "k_G": r["k_G"]} for name, r in results.items()},
               "k_G_differs": len({r["k_G"] for r in results.values()}) > 1,
               "serving": benchmark(directory / "serving", dataset, settings, checkpoints),
               "runs": brief(results.values())}
    return details, {"shrink": results, "phase5": details}
