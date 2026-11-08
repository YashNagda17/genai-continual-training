from dataclasses import replace
from pathlib import Path
import copy
import json
import math
import os
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..models.model_causal_lm import CausalLMSettings
from ..models.model_lfm25 import LFM25Model
from ..models.model_qwen3 import Qwen3Model
from ..models.model_smollm2 import SmolLM2Model

from .fakes import Controller, TinyDataset, TinyModel, record
from ..combined_pipeline import CombinedPipeline
from ..pipelines.al_sar import ALSARConfig, ALSARPipeline
from ..pipelines.csrel import CSReLConfig, CSReLPipeline
from ..pipelines.oasis import OASISConfig, OASISPipeline
from ..pipelines.tsr import TSRConfig, TSRPipeline
from ..utils.contracts import (
    ReplayBudget, ResourceBudget, RunContext, RunSpec, StepReport,
    ScoreResult,
)
from ..utils.numerics import (
    continual_metrics, move_to_device, quotas, resolve_device, sample,
)
from ..utils.state import ReplayStore


def oasis(rate=1, profile="current_only"):
    return OASISPipeline(OASISConfig(.2, 0, 1, 1e-8, rate, 8, "head", profile))


def tsr(alpha=.5):
    return TSRPipeline(TSRConfig(alpha, .5, .5, 2, True, "fixed-zero-B", "lora-b", 2, 2, False))


def sar(hybrid=False):
    return ALSARPipeline(ALSARConfig(1, 4, .2, 4, "subset", False, True, True,
                                    "active_population" if hybrid else "native_history"))


def csrel(variant="csrel_cl"):
    return CSReLPipeline(CSReLConfig(variant, 2, 1, 1, "reference-base", "subset-base", .7, 1, False))


def combined(rate=1, alpha=.5):
    return CombinedPipeline(oasis=oasis(rate), tsr=tsr(alpha), al_sar=sar(True), csrel=csrel())


def spec(**changes):
    base = RunSpec(("a", "b"), 17, 2, 5, 4, 1000, 2, "repeat_admitted", "per_example_mean",
                   ReplayBudget(4, 1000, 100000), ResourceBudget(1000000, 1000000),
                   1, {"protocol": "tiny-interface-check", "adaptations": ["synthetic_model"]},
                   {"metric": "negative_mse"})
    return replace(base, **changes)


def run_pipeline(pipeline, root, run_spec=None, controller=None, dataset=None):
    model = TinyModel(controller=controller)
    result = pipeline.run(RunContext(model, dataset or TinyDataset(), run_spec or spec(), root))
    return result, model



class EquationTests(unittest.TestCase):
    def test_siren_keeps_higher_order_and_negative_cosine(self):
        values = OASISPipeline.siren(((2, 0), (1, 0), (1, 1)), ("a", "b", "c"))
        self.assertAlmostEqual(values[0], 4, delta=1e-6)
        self.assertAlmostEqual(values[2], 2 - 4 / math.sqrt(2), delta=1e-6)
        self.assertAlmostEqual(values[1], 1 - 4 - 2 / math.sqrt(2) + 9 / math.sqrt(10), delta=1e-6)
        gradients = torch.tensor([[2., 0.], [-1., 0.]], requires_grad=True)
        self.assertEqual(OASISPipeline.siren(gradients, ("a", "b")), (4, 5))
        self.assertIsNone(gradients.grad)
        self.assertEqual(OASISPipeline.siren(((0, 0), (0, 0)), ("a", "b")), (0, 0))

    def test_oasis_prior_statistics_and_raw_batch_mean(self):
        pipeline = oasis(.5)
        batch = TinyDataset().collate((record("a0"), record("a1", x=(.5, 1))))
        model = TinyModel()
        result = pipeline.select_current(batch, model, "per_example_mean")
        self.assertEqual(result.provenance["previous_mean"], 0)
        average = (5 + 6.6125) / 2  # Hand-computed squared gradient norms.
        self.assertAlmostEqual(pipeline.policy["mean"], .2 * average, delta=1e-6)
        self.assertAlmostEqual(pipeline.policy["variance"], .2 * average ** 2 + .8, delta=1e-6)
        selected = tuple(i for i, p in zip(result.candidate_ids, result.probabilities)
                         if p > result.provenance["draw"])
        self.assertEqual(selected, result.selected_ids)

    def test_calibration_solves_equation(self):
        for rate, threshold in ((.0625, 3.1193), (.125, 2.2854), (.25, 1.3149)):
            computed = OASISPipeline.calibrate_threshold(rate)
            self.assertAlmostEqual(computed, threshold, places=4)
            self.assertAlmostEqual(OASISPipeline.expected_rate(computed), rate, places=10)

    def test_signed_sar_probabilities(self):
        pipeline = sar(True)
        rows = (record("a0", label="x"), record("a1", label="y"))
        pipeline.register(rows)
        pipeline.policy["counts"] = {"a0": 1., "a1": 3.}
        pipeline.policy["class_totals"] = {"x": 1., "y": 3.}
        pipeline.policy["similarity"] = {"x": {"y": -1.}}
        self.assertEqual(pipeline.effective_frequencies(rows), (1, 2))
        probabilities = pipeline.probabilities(rows)
        self.assertAlmostEqual(probabilities[0], 1 / (1 + math.exp(-1)), delta=1e-6)

    def test_sar_decay_multiplicity_and_eviction(self):
        pipeline = sar(True)
        rows = (record("a0"), record("a1"), record("a2", label="negative"))
        pipeline.register(rows)
        pipeline.policy["counts"] = dict.fromkeys(("a0", "a1", "a2"), 1.)
        pipeline.update_counts((rows[0], rows[0], rows[2]))
        self.assertEqual(pipeline.policy["counts"], {"a0": 2.75, "a1": .75, "a2": 1.75})
        pipeline.reconcile((rows[0],))
        self.assertEqual(pipeline.policy["class_totals"], {"positive": 2.75})
        native = sar()
        native.register(rows)
        native.update_counts(rows)
        native.reconcile((rows[0],))
        self.assertEqual(native.policy["class_totals"], {"positive": 2, "negative": 1})

    def test_sar_cold_start_bfc_cadence_and_skip(self):
        pipeline, info = sar(True), TinyModel().info
        self.assertEqual(pipeline.plan_update({"feature_norm_sq": 1}, info).prefix, 0)
        pipeline.policy.update(fisher=[1, 10], feature_square=1)
        self.assertEqual(pipeline.plan_update({"feature_norm_sq": 0}, info).prefix, 1)
        self.assertEqual(pipeline.plan_update({"feature_norm_sq": 100}, info).prefix, 0)
        pipeline.policy["updates"] = 3
        self.assertTrue(pipeline.plan_update({"feature_norm_sq": 0}, info).refresh)
        before = copy.deepcopy(pipeline.policy)
        report = StepReport(("a0",), False, 0, 2, {}, {}, skip_reason="overflow")
        pipeline.observe_update(report, (record("a0"),), info)
        self.assertEqual(before, pipeline.policy)

    def test_similarity_and_fisher_steps_can_be_composed(self):
        pipeline = sar(True)
        rows = (record("a0", label="x"), record("a1", label="y"))
        pipeline.update_similarities(rows, ((1, 0), (-1, 0)))
        torch.testing.assert_close(pipeline.policy["similarity"],
                                   {"x": {"y": -.2}, "y": {"x": -.2}}, rtol=1e-5, atol=1e-6)
        before = copy.deepcopy(pipeline.policy["similarity"])
        pipeline.update_similarities(rows, None)
        self.assertEqual(pipeline.policy["similarity"], before)
        pipeline.update_fisher((1, 10))
        pipeline.update_feature_statistics(1)
        torch.testing.assert_close(pipeline.information_per_cost(TinyModel().info),
                                   (11 / 7, 10 / 3), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pipeline.freezing_scores(0, TinyModel().info),
                                   (0., 40 / 3), rtol=1e-5, atol=1e-6)

    def test_tsr_distribution_coherent_sources_and_ties(self):
        pipeline = tsr()
        signature = {"values": [1., 0.], "initialization": "fixed", "view": "b",
                     "model": "probe", "reduction": "mean"}
        pipeline.policy["tasks"] = {
            "a": {"signature": signature, "teacher": "teacher-a"},
            "b": {"signature": dict(signature, values=[0., 1.]), "teacher": "teacher-b"}}
        replay = ReplayStore((record("a0"), record("b0", "b")), budget=spec().replay)
        plan = pipeline.route_replay(signature, replay)
        self.assertAlmostEqual(plan.probabilities[0], 1 / (1 + math.exp(-2)), delta=1e-6)
        self.assertEqual(plan.current_teacher, "a")
        pipeline.policy["tasks"]["b"]["signature"] = signature
        self.assertEqual(pipeline.route_replay(signature, replay).current_teacher, "a")
        self.assertEqual(pipeline.select_teacher(("b", "a"), (.5, .5)), "a")
        self.assertEqual(pipeline.route_replay(signature, ReplayStore(budget=spec().replay)).tasks, ())

    def test_quotas_and_negative_forgetting(self):
        self.assertEqual(quotas(("c", "a", "b"), 2), {"a": 1, "b": 1, "c": 0})
        self.assertIsNone(continual_metrics([[.5]], 0)["forgetting"])
        result = continual_metrics([[.5], [.7, .8]], 1)
        self.assertAlmostEqual(result["forgetting"], -.2, delta=1e-6)
        self.assertAlmostEqual(result["backward_transfer"], .2, delta=1e-6)


class CompositionTests(unittest.TestCase):
    def test_shared_model_weight_handoff_and_file_round_trip(self):
        pipeline = combined().set_model(TinyModel())
        pipeline.set_weights((2, 3))
        self.assertTrue(all(c.model is pipeline.model for c in pipeline.components().values()))
        self.assertEqual(pipeline.oasis.predict(((1, 1),), {}), [5])
        target = tsr().set_model(TinyModel())
        pipeline.model.steps = 7
        weights = pipeline.get_weights()
        target.set_weights(weights)
        weights[0] = 99
        self.assertEqual(pipeline.get_weights().tolist(), [2, 3])
        self.assertEqual(target.get_weights().tolist(), [2, 3])
        self.assertEqual((pipeline.model.steps, target.model.steps), (7, 0))
        with tempfile.TemporaryDirectory() as root:
            context = RunContext(pipeline.model, TinyDataset(), spec(), root)
            pipeline.bind(context)
            pipeline.set_model(target.model)
            self.assertIs(context.model, target.model)
            self.assertTrue(all(c.model is target.model for c in pipeline.components().values()))
            path = pipeline.save_weights(Path(root) / "weights")
            target.set_weights((0, 0))
            target.load_weights(path)
            self.assertEqual(target.get_weights().tolist(), [2, 3])

    def test_policy_and_rng_snapshots_are_independent_of_weights(self):
        pipeline = combined().set_model(TinyModel())
        pipeline.oasis.policy["mean"] = 7
        state = json.loads(json.dumps(pipeline.get_policy_state()))
        expected = {name: c.rng.random() for name, c in pipeline.components().items()}
        pipeline.oasis.policy["mean"] = -1
        pipeline.set_weights((2, 3))
        pipeline.set_policy_state(state)
        self.assertEqual(pipeline.oasis.policy["mean"], 7)
        self.assertEqual(pipeline.get_weights().tolist(), [2, 3])
        self.assertEqual({name: c.rng.random() for name, c in pipeline.components().items()}, expected)
        pipeline.oasis.policy["mean"] = 8
        self.assertEqual(state["oasis"]["policy"]["mean"], 7)

    def test_tsr_probe_steps_and_empty_replay(self):
        pipeline, model = tsr(), TinyModel("probe", "fixed-zero-B")
        batches = [TinyDataset().collate((record("a0"),)),
                   TinyDataset().collate((record("a1", x=(0, 1)),))]
        pipeline.policy["probe"] = {"cursor": 0, "batches": 0, "sum": None, "ids": []}
        for batch in batches:
            pipeline.accumulate_probe(pipeline.probe_batch(batch, model, "per_example_mean"), batch.ids)
        signature = pipeline.task_signature(model, "per_example_mean")
        self.assertEqual(signature["ids"], ["a0", "a1"])
        torch.testing.assert_close(signature["values"], [-1., -1.7], rtol=1e-5, atol=1e-6)
        self.assertEqual(pipeline.retrieve_replay((), 2), ())


class TorchTests(unittest.TestCase):
    def test_device_routing_and_cpu_fallback(self):
        # ROCm shares Torch's CUDA interface, so both use the same routing case.
        for cuda, xpu, expected in ((True, True, "cuda:0"), (False, True, "xpu:0"),
                                    (False, False, "cpu")):
            with self.subTest(cuda=cuda, xpu=xpu), \
                    patch.object(torch.cuda, "is_available", return_value=cuda), \
                    patch.object(torch.xpu, "is_available", return_value=xpu), \
                    patch.object(torch.cuda, "device_count", return_value=2), \
                    patch.object(torch.xpu, "device_count", return_value=2):
                self.assertEqual(str(resolve_device()), expected)
                self.assertEqual(str(resolve_device("cpu")), "cpu")
                for backend, available in (("cuda", cuda), ("xpu", xpu)):
                    self.assertEqual(str(resolve_device(backend + ":1")), backend + ":1" if available else "cpu")
                    self.assertEqual(str(resolve_device(backend + ":2")), "cpu")

    def test_nested_transfer_preserves_dtype_and_autograd(self):
        features = torch.tensor([1., 2.], requires_grad=True)
        payload = {"ids": torch.tensor([1, 2]), "mask": torch.tensor([True, False]),
                   "nested": ([features * 2], "metadata")}
        moved = move_to_device(payload, "cpu")
        self.assertEqual(moved["ids"].dtype, torch.int64)
        self.assertEqual(moved["mask"].dtype, torch.bool)
        moved["nested"][0][0].sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([2., 2.]))
        self.assertEqual(moved["nested"][1], "metadata")

    def test_sampling_restores_policy_rng_and_leaves_global_torch_rng_untouched(self):
        rng = random.Random(17)
        policy_state, global_state = rng.getstate(), torch.random.get_rng_state().clone()
        selected = sample(("a", "b", "c"), (.2, .3, .5), 20, True, rng)
        rng.setstate(policy_state)
        self.assertEqual(selected, sample(("a", "b", "c"), (.2, .3, .5), 20, True, rng))
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)
        self.assertEqual(set(sample(("a", "b", "c"), (.2, .3, .5), 3, False, rng)),
                         {"a", "b", "c"})
        state = rng.getstate()
        self.assertEqual(sample((), (), 0, False, rng), ())
        self.assertEqual(rng.getstate(), state)


class LifecycleTests(unittest.TestCase):
    def test_two_task_standalones_and_hybrid(self):
        factories = (oasis, tsr, sar, csrel, lambda: csrel("csrel_cl_prv"), combined)
        for factory in factories:
            with self.subTest(factory=factory), tempfile.TemporaryDirectory() as root:
                pipeline, control = factory(), Controller()
                result, _ = run_pipeline(pipeline, root, controller=control)
                self.assertEqual(result.status, "completed")
                self.assertEqual(len(result.metrics), 3)
                self.assertGreater(result.counters["steps"], 0)
                self.assertLessEqual(control.peak_resident, 1)
                self.assertTrue(Path(root, "metrics.csv").exists())
                retained = json.loads(Path(root, "latest.json").read_text())["checkpoints"]
                self.assertEqual(len(retained), 1)
                states = list(Path(root, "checkpoints").glob("*/state.json"))
                self.assertEqual(len(states), 1)
                self.assertFalse(control.resident)

    def test_empty_admission_and_zero_quota(self):
        cases = ((oasis(0, "historical_pool"), spec(replay=ReplayBudget(None, 1000, 100000)), 8),
                 (combined(0, 1), spec(), 4), (csrel(), spec(replay=ReplayBudget(0, 0, 0)), 0))
        for pipeline, run_spec, retained in cases:
            with self.subTest(pipeline=type(pipeline).__name__), tempfile.TemporaryDirectory() as root:
                control = Controller()
                result, _ = run_pipeline(pipeline, root, run_spec, control)
                self.assertEqual(len(pipeline.replay.records), retained)
                if retained == 0:
                    self.assertFalse(any(e[0] in {"reference", "subset"} for e in control.log))
                if isinstance(pipeline, (OASISPipeline, CombinedPipeline)):
                    self.assertEqual(result.counters["attempts"], 0)

    def test_alpha_extremes_and_skips_terminate(self):
        for alpha in (0., 1.):
            with self.subTest(alpha=alpha), tempfile.TemporaryDirectory() as root:
                pipeline = combined(1, alpha)
                result, _ = run_pipeline(pipeline, root, controller=Controller(skip=True))
                self.assertEqual(result.counters["steps"], 0)
                self.assertLessEqual(result.counters["attempts"], 2 * spec().max_attempts)
                self.assertEqual(pipeline.al_sar.policy["updates"], 0)
                self.assertEqual(set(pipeline.al_sar.policy["counts"].values()), {0.0})

    def test_tsr_replay_keeps_teacher_and_probe_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline, controller = combined(1, 1), Controller()
            run_pipeline(pipeline, root, controller=controller)
            for event in pipeline.events:
                if event["kind"] == "learner_update" and event["task"] == "b":
                    self.assertEqual(event["teacher"], "a")
                    self.assertTrue(all(i.startswith("a") for i in event["ids"]))
            probes = [entry for entry in controller.log if entry[:3] == ("probe", "score", "lora_b_gradient")]
            self.assertEqual(len(probes), 4)
            self.assertEqual({entry[4] for entry in probes}, {(.1, -.2)})

    def test_csrel_loss_difference_and_round_dependent_ranking(self):
        self.assertEqual(CSReLPipeline.select_top({"b": 3, "a": 3}, 1), ["a"])
        original_score = TinyModel.score

        def scored(model, batch, score_spec):
            if model.role not in {"reference", "subset"}:
                return original_score(model, batch, score_spec)
            if model.role == "reference":
                losses = {"a0": 8., "a1": .4, "a2": .1}
            elif model.steps == 0:
                losses = {"a0": 8.2, "a1": 3., "a2": 2.}
            else:
                losses = {"a0": 8.2, "a1": .5, "a2": .15}
            return ScoreResult(batch.ids, [losses[i] for i in batch.ids],
                               f"{model.role}-{model.steps}", "", score_spec.reduction)

        with tempfile.TemporaryDirectory() as root, patch.object(TinyModel, "score", scored):
            pipeline = csrel()
            dataset = TinyDataset((record("a0"), record("a1"), record("a2")))
            run_pipeline(pipeline, root, spec(task_ids=("a",), replay=ReplayBudget(2, 100, 1000)),
                         dataset=dataset)
            # Highest raw loss would choose a0 first; a one-shot ReL ranking would
            # choose a1,a2. Refitting the subset changes the second selection.
            self.assertEqual([r.sample_id for r in pipeline.replay.records], ["a1", "a0"])

    def test_resume_matches_uninterrupted_at_multiple_phases(self):
        with tempfile.TemporaryDirectory() as baseline_root:
            baseline = combined(.5)
            _, expected_model = run_pipeline(baseline, baseline_root)
            cases = (("learner.train", 2), ("reference.train", 2), ("subset.train", 1),
                     ("reference.score", 2), ("probe.score", 2), ("evaluate", 2), ("learner.save", 5))
            for operation, occurrence in cases:
                with self.subTest(operation=operation), tempfile.TemporaryDirectory() as root:
                    with self.assertRaises(KeyboardInterrupt):
                        run_pipeline(combined(.5), root, controller=Controller(operation, occurrence))
                    restored = combined(.5)
                    result, model = run_pipeline(restored, root, spec(resume="required"))
                    self.assertEqual(model.weights, expected_model.weights)
                    self.assertEqual(result.counters, baseline.counters)
                    self.assertEqual(restored.metrics, baseline.metrics)
                    for name in ("oasis", "al_sar"):
                        self.assertEqual(getattr(restored, name).policy, getattr(baseline, name).policy)
                    self.assertEqual(restored.replay.state_dict(), baseline.replay.state_dict())
                    ids = [event["event"] for event in restored.events]
                    self.assertEqual(ids, list(range(1, len(ids) + 1)))
                    self.assertFalse(list(Path(root, "checkpoints").glob(".stage-*")))

    def test_corrupt_latest_falls_back_without_mixing_states(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyboardInterrupt):
                run_pipeline(csrel(), root, controller=Controller("learner.train", 3))
            retained = json.loads(Path(root, "latest.json").read_text())["checkpoints"]
            self.assertEqual(len(retained), 2)
            Path(root, "checkpoints", retained[0], "learner", "model.json").write_text("corrupt")
            pipeline = csrel()
            result, _ = run_pipeline(pipeline, root)
            self.assertEqual(result.status, "completed")
            recovery = json.loads(Path(root, "recovery.json").read_text())
            self.assertEqual(recovery["restored"], retained[1])

    def test_budget_preflight_and_700m_ceiling(self):
        model = TinyModel()
        with self.assertRaises(ValueError):
            replace(model.info, total_parameters=700_000_001).validate()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run_pipeline(oasis(), root, spec(resources=ResourceBudget(1, 0)))
            self.assertFalse(Path(root, "latest.json").exists())

    def test_replay_denies_evicted_records_and_all_units(self):
        store = ReplayStore((record("a0"), record("a1")), budget=ReplayBudget(2, 4, 10000))
        replacement = store.proposal((record("b0", "b"),))
        self.assertFalse(replacement.task("a"))
        self.assertEqual(len(store.task("a")), 2)
        for budget in (ReplayBudget(0, 10, 1000), ReplayBudget(10, 1, 1000), ReplayBudget(10, 10, 1)):
            with self.assertRaises(ValueError):
                ReplayStore((record("a0"),), budget=budget)


@unittest.skipUnless(os.environ.get("GENAI_SETUP_TESTS") == "1",
                     "setup-only check of cached weights and device placement")
class SetupModelTests(unittest.TestCase):
    """Load each cached causal LM and run one short forward on the resolved device.

    Skipped unless GENAI_SETUP_TESTS=1. Routine algorithm runs must not enable it.
    """

    def test_models_load_and_forward_on_device(self):
        settings = CausalLMSettings(rank=4, alpha=8, max_length=32, max_batch_size=1,
                                    working_bytes=64 * 1024 ** 2, max_score_bytes=64 * 1024 ** 2)
        for cls in (SmolLM2Model, LFM25Model, Qwen3Model):
            model = cls(settings, device="auto")
            print(f"setup {cls.__name__} device={model.device}", flush=True)
            model.activate()
            try:
                parameter = next(model.module.parameters())
                self.assertEqual(parameter.device.type, model.device.type)
                self.assertEqual(parameter.dtype, torch.float32)
                token_ids = torch.randint(1, 32, (1, 8), device=model.device)
                mask = torch.ones_like(token_ids, dtype=torch.bool)
                with torch.no_grad():
                    logits = model.module(input_ids=token_ids, attention_mask=mask, use_cache=False).logits
                self.assertEqual(logits.device.type, model.device.type)
                self.assertTrue(bool(torch.isfinite(logits).all()))
            finally:
                model.release()


if __name__ == "__main__":
    unittest.main()
