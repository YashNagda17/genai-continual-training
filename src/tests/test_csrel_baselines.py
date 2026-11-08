"""CSReL selection checks and the matched baselines. The model is the tiny fixture."""

from pathlib import Path
from dataclasses import replace
from random import Random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from .fakes import Controller, TinyDataset
from .test_pipelines import run_pipeline, spec
from ..csrel_experiment import BalancedViews, performance_report, parse_args, resolved_config
from ..datasets.dataset_sources import CLINC_DOMAINS, clinc_domain_rows
from ..pipelines.csrel import CSReLConfig, CSReLPipeline
from ..pipelines.memory_baselines import MemoryBaselineConfig, MemoryBaselinePipeline
from ..utils.contracts import ReplayBudget
from ..utils.runtime import PipelineRuntime
from ..utils.state import Checkpoints, checksums, replace_artifact


def csrel(variant="csrel_cl"):
    return CSReLPipeline(CSReLConfig(variant, 2, 1, 1, "reference-base", "subset-base", 1.0, 1, False))


def baseline(method):
    return MemoryBaselinePipeline(MemoryBaselineConfig(method, 1.0, 1, False))


class SelectionTests(unittest.TestCase):
    def test_shuffled_pass_membership_and_independent_replay_rng(self):
        runtime = PipelineRuntime()
        runtime.spec = SimpleNamespace(seed=17, identity={"shuffle_training": True})
        runtime.task_index = 1
        runtime.rng = Random(17)
        before = runtime.rng.getstate()
        items = list(range(40))
        first = runtime.learner_pass_order(items, 1)
        self.assertEqual(sorted(first), items)
        self.assertEqual(first, runtime.learner_pass_order(items, 1))
        self.assertNotEqual(first, runtime.learner_pass_order(items, 2))
        self.assertEqual(before, runtime.rng.getstate())
        self.assertEqual(items, list(range(40)))
        runtime.spec.identity["shuffle_training"] = False
        self.assertEqual(runtime.learner_pass_order(items, 1), items)

    def test_shuffled_resume_matches_uninterrupted_updates(self):
        run_spec = spec(max_steps=20, max_attempts=20, passes=3)
        run_spec = replace(run_spec, identity={**run_spec.identity, "shuffle_training": True})
        with tempfile.TemporaryDirectory() as expected_root, tempfile.TemporaryDirectory() as actual_root:
            expected = baseline("random")
            _, expected_model = run_pipeline(expected, expected_root, run_spec)
            with self.assertRaises(KeyboardInterrupt):
                run_pipeline(baseline("random"), actual_root, run_spec,
                             controller=Controller("learner.train", 3))
            restored = baseline("random")
            _, restored_model = run_pipeline(restored, actual_root, replace(run_spec, resume="required"))
            self.assertEqual(restored_model.weights, expected_model.weights)
            expected_ids = [e["ids"] for e in expected.events if e["kind"] == "learner_update"]
            actual_ids = [e["ids"] for e in restored.events if e["kind"] == "learner_update"]
            self.assertEqual(actual_ids, expected_ids)
            self.assertEqual(len(expected_ids), 12)

    def test_checkpoint_checksums_ignore_explorer_metadata_only(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "state.json").write_text("{}")
            expected = checksums(folder)
            (folder / "desktop.ini").write_text("Explorer metadata")
            (folder / "Thumbs.db").write_bytes(b"thumbnail cache")
            self.assertEqual(checksums(folder), expected)
            (folder / "state.json").write_text('{"changed": true}')
            self.assertNotEqual(checksums(folder), expected)

    def test_atomic_publication_retries_windows_lock_without_copying(self):
        with patch("src.utils.state.os.name", "nt"):
            with patch("src.utils.state.os.replace", side_effect=[PermissionError(), None]) as publish:
                with patch("src.utils.state.time.sleep"):
                    replace_artifact("stage", "published")
            self.assertEqual(publish.call_count, 2)
            with patch("src.utils.state.os.replace", side_effect=PermissionError()) as publish:
                with patch("src.utils.state.time.sleep"):
                    with self.assertRaises(PermissionError):
                        replace_artifact("stage", "published")
            self.assertEqual(publish.call_count, 5)

    def test_cleanup_retries_permission_errors_and_rejects_attempt_root(self):
        with tempfile.TemporaryDirectory() as root:
            manager = Checkpoints(root, "fixture")
            obsolete = Path(root) / "obsolete"
            with patch("src.utils.state.shutil.rmtree", side_effect=[PermissionError(), None]) as remove:
                with patch("src.utils.state.time.sleep"):
                    manager._remove_owned(obsolete)
            self.assertEqual(remove.call_count, 2)
            with self.assertRaises(ValueError):
                manager._remove_owned(root)
            with self.assertRaises(ValueError):
                manager._remove_owned(Path(root).parent)

    def test_balanced_sampling_covers_classes_and_preserves_arrival(self):
        records = tuple(replace(row, class_id=str(i % 2))
                        for i, row in enumerate(TinyDataset().records))
        dataset = SimpleNamespace(fingerprint="fixture",
                                  training_view=lambda task: records,
                                  evaluation_view=lambda task: records)
        view = BalancedViews(dataset, 2, 1, 17)
        train = view.training_view("a")
        self.assertEqual([sum(r.class_id == label for r in train) for label in ("0", "1")], [2, 2])
        self.assertEqual(list(train), sorted(train, key=lambda r: (r.arrival, r.sample_id)))
        self.assertEqual(train, view.training_view("a"))
        with self.assertRaises(ValueError):
            BalancedViews(dataset, 100, 1, 17).training_view("a")

    def test_pilot_refuses_prefix_limits(self):
        with self.assertRaises(SystemExit):
            resolved_config(parse_args(["--protocol", "pilot", "--max-train-examples", "8"]))

    def test_cumulative_full_pass_guard_rejects_truncated_budget(self):
        pipeline = MemoryBaselinePipeline(MemoryBaselineConfig("cumulative", 1., 0, False, True))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "cannot cover"):
                run_pipeline(pipeline, root, spec(max_steps=1, passes=1, max_attempts=1))

    def test_reducible_loss_prefers_the_learnable_example(self):
        scores = CSReLPipeline.reducible_losses(
            {"easy": {"loss": 3.0}, "noisy": {"loss": 8.0}},
            {"easy": {"loss": 0.4}, "noisy": {"loss": 7.8}})
        self.assertAlmostEqual(scores["easy"], 2.6, places=5)
        self.assertAlmostEqual(scores["noisy"], 0.2, places=5)
        self.assertEqual(CSReLPipeline.select_top({"b": 3.0, "a": 3.0}, 1), ["a"])

    def test_domain_map_drops_oos_and_rejects_unknown_intents(self):
        names = [intent for intents in CLINC_DOMAINS.values() for intent in intents] + ["oos"]
        self.assertEqual(len(names) - 1, 150)
        by_split = {"train": [], "validation": [], "test": []}
        for intents in CLINC_DOMAINS.values():
            by_split["train"].append((f"ask about {intents[0]}", names.index(intents[0])))
        by_split["train"].append(("out of scope", names.index("oos")))
        rows, report = clinc_domain_rows(by_split, names)
        self.assertEqual(report["oos_dropped"], 1)
        self.assertEqual(report["task_order"], list(CLINC_DOMAINS))
        self.assertEqual(rows[0]["task"], "banking")
        self.assertTrue(all(row["class"] != "oos" for row in rows))
        with self.assertRaises(ValueError):
            clinc_domain_rows(by_split, names + ["not-an-intent"])

    def test_memory_capacity_and_historical_reselection_stay_inside_retained_ids(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = csrel()
            run_pipeline(pipeline, root, spec(replay=ReplayBudget(4, 1000, 100000)))
            commits = [event for event in pipeline.events if event["kind"] == "memory_commit"]
            first = set(commits[0]["ids"])
            self.assertEqual(len(first), 4)
            later = [sample_id for event in pipeline.events
                     if event["kind"] == "coreset_round" and event["task"] == "b"
                     and event["selection_task"] == "a" for sample_id in event["scores"]]
            self.assertTrue(set(later) <= first)
            retained = {record.sample_id for record in pipeline.replay.records}
            self.assertEqual(len(retained), 4)
            self.assertTrue({record.sample_id for record in pipeline.replay.records
                             if record.task_id == "a"} <= first)
            train_ids = {item.sample_id for item in TinyDataset().records}
            for event in pipeline.events:
                if event["kind"] not in {"learner_update", "auxiliary_update", "auxiliary_scoring"}:
                    continue
                self.assertTrue(set(event["ids"]) <= train_ids)
            task_a = [event for event in pipeline.events
                      if event["kind"] == "learner_update" and event["task"] == "a"]
            self.assertTrue(all(sample_id.startswith("a") for event in task_a for sample_id in event["ids"]))

    def test_reference_pool_matches_the_variant(self):
        for variant, expects_history in (("csrel_cl", False), ("csrel_cl_prv", True)):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as root:
                pipeline = csrel(variant)
                run_pipeline(pipeline, root)
                ids = [sample_id for event in pipeline.events
                       if event["kind"] == "auxiliary_update" and event["task"] == "b"
                       and event["phase"] == "reference_fit" for sample_id in event["ids"]]
                self.assertTrue(any(sample_id.startswith("b") for sample_id in ids))
                self.assertEqual(any(sample_id.startswith("a") for sample_id in ids), expects_history)

    def test_reference_model_starts_from_its_own_initialization(self):
        with tempfile.TemporaryDirectory() as root:
            control = Controller()
            _, model = run_pipeline(csrel(), root, controller=control)
            reference = [entry for entry in control.log if entry[0] == "reference" and entry[1] == "train"]
            self.assertEqual(reference[0][3], (0.1, -0.2))
            self.assertNotEqual(tuple(model.weights), (0.1, -0.2))

    def test_new_only_stores_nothing_and_random_replays_retained_ids(self):
        with tempfile.TemporaryDirectory() as root:
            control = Controller()
            new_only = baseline("new_only")
            run_pipeline(new_only, root, controller=control)
            self.assertEqual(new_only.replay.records, ())
            self.assertFalse(any(entry[0] == "reference" for entry in control.log))
            task_b = [event for event in new_only.events
                      if event["kind"] == "learner_update" and event["task"] == "b"]
            self.assertTrue(all(sample_id.startswith("b") for event in task_b for sample_id in event["ids"]))
        with tempfile.TemporaryDirectory() as root:
            control = Controller()
            random_run = baseline("random")
            run_pipeline(random_run, root, controller=control)
            self.assertEqual(len(random_run.replay.records), 4)
            task_b = [event for event in random_run.events
                      if event["kind"] == "learner_update" and event["task"] == "b"]
            self.assertTrue(any(sample_id.startswith("a") for event in task_b for sample_id in event["ids"]))
            self.assertFalse(any(entry[0] == "reference" for entry in control.log))

    def test_cumulative_retrains_from_the_initial_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline = baseline("cumulative")
            _, model = run_pipeline(pipeline, root, spec(max_steps=1, passes=1, max_attempts=2))
            opened = [event for event in pipeline.events if event["kind"] == "cumulative_retrain"]
            self.assertEqual([event["training_examples"] for event in opened], [4, 8])
            self.assertEqual(model.steps, 1)
            self.assertEqual(pipeline.replay.records, ())
            self.assertTrue((Path(root) / "initial_weights" / "model.json").is_file())

    def test_random_resume_matches_the_uninterrupted_memory(self):
        with tempfile.TemporaryDirectory() as baseline_root:
            expected = baseline("random")
            _, expected_model = run_pipeline(expected, baseline_root)
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(KeyboardInterrupt):
                    run_pipeline(baseline("random"), root, controller=Controller("learner.train", 2))
                restored = baseline("random")
                _, model = run_pipeline(restored, root, spec(resume="required"))
                self.assertEqual(model.weights, expected_model.weights)
                self.assertEqual(restored.replay.state_dict(), expected.replay.state_dict())

    def test_forgetting_uses_the_best_earlier_score(self):
        metrics = [
            {"checkpoint_task": "a", "task": "a", "metrics": {"exact_match": 1.0}},
            {"checkpoint_task": "b", "task": "a", "metrics": {"exact_match": 0.5}},
            {"checkpoint_task": "b", "task": "b", "metrics": {"exact_match": 0.8}},
        ]
        report = performance_report(metrics, ("a", "b"), "exact_match")
        self.assertAlmostEqual(report["forgetting"], 0.5)
        self.assertAlmostEqual(report["final_average"], 0.65)
        self.assertEqual(report["current_task"], {"a": 1.0, "b": 0.8})


if __name__ == "__main__":
    unittest.main()
