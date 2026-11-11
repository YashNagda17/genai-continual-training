"""C1 location maps, C2 block plans, and E's parameter-matched extra-rank plans."""

from dataclasses import dataclass, field, replace
import random

import torch

from .pipeline_base import PipelineBase
from ..utils.numerics import cosine, spearman, write_json

ROOM_ARMS = ("room_placed", "room_random", "room_spread")


@dataclass(frozen=True)
class AdapterPlan:
    """Which shared adapters can change (`encoder.7` keys), where extra rank goes, and per-block learning rates."""
    trainable: tuple
    new_rank: dict = field(default_factory=dict)
    lr_scale: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)


def cost(key):
    """Parameters per rank: one encoder block, or one decoder block with cross-attention."""
    return 8192 if key.startswith("encoder.") else 11776


def blocks(settings):
    return {stack: [f"{stack}.{i}" for i in range(1, count + 1)]
            for stack, count in (("encoder", settings.encoder_blocks), ("decoder", settings.decoder_blocks))}


def block_plan(name, settings):
    """C2: all blocks (= A), late (encoder and decoder 7-8), or early + late (1 and 8 of each)."""
    stacks = blocks(settings)
    trainable = {"all": stacks["encoder"] + stacks["decoder"],
                 "late": stacks["encoder"][-2:] + stacks["decoder"][-2:],
                 "early_late": sorted({ids[0] for ids in stacks.values()} | {ids[-1] for ids in stacks.values()})}[name]
    return AdapterPlan(tuple(trainable), identity={"arm": name, "trainable_parameters": 16 * sum(map(cost, trainable))})


def room_plans(settings, maps, shared, seed, names=ROOM_ARMS):
    """E: new rank where agreement < 0 (E1), at random in the same blocks per stack (E2), on all blocks (E3),
    with the shared adapter slowed (E4) or frozen (E5) there. Returns (plans by name, reason when none)."""
    if not maps["stable"]:
        return {}, "unstable_map"
    clashing = sorted(key for key, value in maps["blocks"].items() if value < 0)
    if not clashing:
        return {}, "no_clashing_blocks"
    stacks, rng, plans = blocks(settings), random.Random(seed), {}
    for arm in names:
        chosen = list(clashing)
        if arm == "room_random":
            chosen = [key for stack, ids in stacks.items()
                      for key in rng.sample(ids, sum(k.startswith(stack + ".") for k in clashing))]
        elif arm == "room_spread":
            chosen = stacks["encoder"] + stacks["decoder"]
        rank = max(1, round(settings.new_room_budget / sum(map(cost, chosen))))
        trainable = [key for key in shared.trainable if arm != "room_placed_frozen" or key not in chosen]
        scales = dict(shared.lr_scale)
        if arm == "room_placed_slow":
            scales.update({key: max(settings.floor, min(1.0, 1 + settings.gamma * maps["blocks"][key])) for key in chosen})
        plans[arm] = AdapterPlan(tuple(trainable), {key: rank for key in chosen}, scales,
                                 {"arm": arm, "new_parameters": rank * sum(map(cost, chosen)),
                                  "overlap": len(set(chosen) & set(clashing)) / len(clashing)})
    counts = [plan.identity["new_parameters"] for plan in plans.values()]
    if (max(counts) - min(counts)) / min(counts) > settings.match_tolerance:
        return {}, "Extra-room arms exceed the pre-registered parameter-match tolerance"
    return {name: replace(plan, identity={**plan.identity, "matched_counts": counts}) for name, plan in plans.items()}, None


class AdapterRoomPipeline(PipelineBase):
    """C1: change, pull and clash maps from a zero-start probe on the English checkpoint, on two row samples."""

    def __init__(self, settings, model, dataset, output, pool, hindi, english_checkpoint, hindi_checkpoint, seed=0):
        super().__init__(settings, model, dataset, output, seed)
        self.rows = {"english": tuple(pool), "hindi": tuple(hindi)}
        self.checkpoints = {"english": english_checkpoint, "hindi": hindi_checkpoint}

    def mean_gradients(self, rows):
        totals = {}
        for part in self.batches(rows):
            batch = self.dataset.collate(part)
            gradients = self.timed(lambda: self.model.probe_gradients(batch), "probe_pass", part, units=3)
            for name, gradient in gradients.items():
                totals[name] = totals.get(name, 0) + len(part) * gradient.detach()
        return {name: value / len(rows) for name, value in totals.items()}

    def run(self):
        change = {}
        for language, path in self.checkpoints.items():  # ||B A|| per cell, in the English checkpoint and in X
            self.model.load(path)
            cells = {}
            for _, m in self.model.adapters():
                cells[m.cell] = cells.get(m.cell, 0.0) + float(m.delta().norm())
            change[language] = cells
        self.model.load(self.checkpoints["english"])
        self.model.reset_probe()
        groups = {name: {"cell": m.cell, "block": m.block} for name, m in self.model.adapters()}
        n = self.settings.sample_size
        samples = {language: self.rng.sample([r.id for r in rows], 2 * n) for language, rows in self.rows.items()}
        lookup = {r.id: r for rows in self.rows.values() for r in rows}
        maps = []
        for index in range(2):
            means = {language: self.mean_gradients([lookup[i] for i in ids[index * n:(index + 1) * n]])
                     for language, ids in samples.items()}
            sample = {}
            for view in ("cell", "block"):
                keys = sorted({group[view] for group in groups.values()})
                vectors = {language: {key: torch.cat([g for name, g in values.items() if groups[name][view] == key])
                                      for key in keys} for language, values in means.items()}
                sample[view] = {key: {"pull": float(vectors["hindi"][key].norm()),
                                      "agreement": cosine(vectors["english"][key], vectors["hindi"][key])} for key in keys}
            maps.append(sample)
        cells = sorted(maps[0]["cell"])
        correlation = spearman([maps[0]["cell"][key]["agreement"] for key in cells],
                               [maps[1]["cell"][key]["agreement"] for key in cells])
        result = {"change": change, "samples": maps, "correlation": correlation,
                  "stable": correlation is not None and correlation >= self.settings.stability_threshold,
                  "blocks": {key: sum(sample["block"][key]["agreement"] for sample in maps) / 2 for key in maps[0]["block"]}}
        write_json(self.output / "location_maps.json", result)
        return {"maps": result, "costs": self.costs}
