"""Phase runner checks. The smoke protocol uses a tiny local model and does not download."""

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
from ..phases.phase0 import summary_word_coverage
from ..phases.data import build_smoke_dataset, build_smoke_model, choose_ids, with_prefix
from ..phases.settings import PhaseSettings
from ..phases.serving import measure
from ..phases.support import execute, hindi_pipeline, phase, run_hindi
from ..pipelines.revision import load_pool
from ..phases.support import choose_winner, gate_decision, script_share_of_forgetting
from ..summarization_experiment import main


def arm(name, english, hindi, seconds=1):
    return {"name": name, "wall_clock_seconds": seconds,
            "metrics": {"english/validation/plain": {"chrf": english},
                        "hindi/validation/plain": {"chrf": hindi}}}


class ServingTimingTests(unittest.TestCase):
    def test_one_model_stays_loaded_for_every_article(self):
        class Model:
            def __init__(self):
                self.module = None
                self.built = 0

            def activate(self):
                if self.module is None:
                    self.built += 1
                    self.module = True

            def predict(self, batch, spec):
                if self.module is None:
                    raise RuntimeError("model was released")
                return {"token_ids": [[1, 2]], "cost": {}}

        class Data:
            def collate_sources(self, records):
                return records

        rows = [SimpleNamespace(sample_id=str(i)) for i in range(3)]
        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "english-unmerged.json"
            path.write_text(json.dumps({"requests": [{"language": "english", "seconds": 9}], "status": "completed"}), encoding="utf-8")
            saved = measure(path, Data(), {"english": tuple(rows)}, PhaseSettings(), lambda language: model)
        self.assertEqual(model.built, 1)
        self.assertEqual(len(saved["requests"]), 3)
        self.assertTrue(saved["resident"])
        self.assertEqual(saved["status"], "completed")


class PhaseRuleTests(unittest.TestCase):
    def test_epoch_override_reaches_both_stage_budgets(self):
        from ..summarization_experiment import parse_args, prepare
        from ..phases.support import run_spec
        args = parse_args(["--pilot", "--epochs", "5"])
        dataset, factory, settings = prepare("smoke", "cpu", epochs=args.epochs)
        self.assertEqual((settings.english_passes, settings.hindi_passes), (5, 5))
        role = factory()
        try:
            for stage in ("english", "hindi"):
                spec = run_spec(stage, settings, getattr(settings, stage + "_passes"), stage, dataset, role)
                rows = len(dataset.training_view(stage))
                self.assertEqual(spec.max_steps, ((rows + settings.batch_size - 1) // settings.batch_size) * 5)
                self.assertEqual(spec.passes, 5)
        finally:
            role.release()

    def test_truncation_keeps_words_from_the_token_prefix(self):
        article = "alpha beta gamma delta epsilon"
        words = article.split()
        row = SimpleNamespace(task_id="english", inputs={"text": article}, target={"text": "beta epsilon"})
        coverage = summary_word_coverage(
            [row], lambda text: [words.index(word) + 1 for word in text.split()],
            lambda ids: " ".join(words[index - 1] for index in ids), 3)
        self.assertEqual(coverage["english"]["full"], 2)
        self.assertEqual(coverage["english"]["kept"], 1)
        self.assertEqual(coverage["english"]["coverage"], 0.5)

    def test_draw_is_stable_and_keeps_arrival_order(self):
        records = [SimpleNamespace(sample_id=str(index), arrival=index) for index in range(10)]
        chosen = choose_ids(records, 4, 0)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(chosen, choose_ids(records, 4, 0))
        self.assertEqual(chosen, sorted(chosen, key=int))

    def test_winner_and_gate_follow_the_plan(self):
        before = {"english/validation/plain": {"chrf": 30}, "hindi/validation/plain": {"chrf": 8}}
        random = arm("random", 22, 25)
        loss = arm("loss", 28, 25, 2)
        drift = arm("drift", 24, 25, 3)
        past = arm("past", 29, 10)
        self.assertEqual(choose_winner(before, [loss, drift, past], random)["name"], "loss")
        self.assertEqual(choose_winner(before, [past], random)["name"], "random")
        # Decisions use the screening seed, so a confirmed arm is judged on the same seed as the others.
        confirmed = dict(arm("loss", 10, 25), seed_runs=[loss, arm("loss", 10, 25)])
        self.assertEqual(choose_winner(before, [confirmed, drift], random)["name"], "loss")
        self.assertEqual(gate_decision(8, 17), "continue")
        self.assertEqual(gate_decision(0.5, 17), "stop_replay")
        self.assertEqual(gate_decision(8, 0), "fix_setup")
        self.assertEqual(gate_decision(8, 17, english_score=11.6, lead_score=22.0), "fix_english")
        self.assertEqual(gate_decision(8, 17, english_score=25.0, lead_score=22.0), "continue")
        self.assertEqual(script_share_of_forgetting(30, 22, 27), 0.625)
        self.assertIsNone(script_share_of_forgetting(22, 22, 27))

    def test_default_command_resumes_scientific_from_the_last_phase(self):
        from ..summarization_experiment import RUN_ORDER, parse_args, selected_phases
        args = parse_args([])
        self.assertEqual(args.protocol, "scientific")
        self.assertFalse(args.pilot)
        self.assertEqual(args.output, ".")
        self.assertIsNone(args.phase)
        self.assertTrue(parse_args(["--pilot"]).pilot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(RUN_ORDER, (0, 1, 2, 6, 7, 3, 4, 5))
            self.assertEqual(selected_phases(root, None), RUN_ORDER)
            (root / "phase2").mkdir()
            (root / "progress.json").write_text(json.dumps({"last_phase": 2}))
            (root / "phase2" / "report.json").write_text("{}")
            self.assertEqual(selected_phases(root, None), (6, 7, 3, 4, 5))
            (root / "phase2" / "report.json").unlink()
            self.assertEqual(selected_phases(root, None), (2, 6, 7, 3, 4, 5))
            (root / "progress.json").write_text(json.dumps({"last_phase": 7}))
            self.assertEqual(selected_phases(root, None), (7, 3, 4, 5))
            self.assertEqual(selected_phases(root, 4), (4,))

    def test_interruption_keeps_progress_without_a_completed_report(self):
        @phase(0)
        def interrupted(directory, dataset, factory, settings, context, clock):
            (directory / "retained.txt").write_text("existing checkpoint")
            raise KeyboardInterrupt("stopped")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                interrupted(directory, SimpleNamespace(fingerprint="data-v1"), None, PhaseSettings(), {})
            folder = Path(directory) / "phase0"
            self.assertEqual(json.loads((folder / "progress.json").read_text())["status"], "interrupted")
            self.assertEqual((folder / "retained.txt").read_text(), "existing checkpoint")
            self.assertFalse((folder / "report.json").exists())
            self.assertEqual([json.loads(line)["status"] for line in (folder / "events.jsonl").read_text().splitlines()],
                             ["running", "interrupted"])
            with self.assertRaisesRegex(ValueError, "different settings"):
                interrupted(directory, SimpleNamespace(fingerprint="data-v1"), None, replace(PhaseSettings(), batch_size=4), {})

    def test_prefix_preserves_selected_subsets(self):
        model = build_smoke_model()
        try:
            dataset = build_smoke_dataset(model)
            for language in ("english", "hindi"):
                dataset.expose(f"{language}/train", [row.sample_id for row in dataset.training_view(language)[:2]])
                dataset.expose(f"{language}/validation", [row.sample_id for row in dataset.views["validation"] if row.task_id == language][:1])
            dataset.expose("english/pool", [dataset.training_view("english")[0].sample_id])
            prefixed = with_prefix(dataset, model, "summarize in {language}: ")
            self.assertEqual(prefixed.manifest["protocol"], dataset.manifest["protocol"])
            self.assertEqual(len(prefixed.views["train"]), 4)
            self.assertEqual(len(prefixed.views["validation"]), 2)
        finally:
            model.release()


class DecodingTests(unittest.TestCase):
    def test_a_repeated_trigram_is_blocked(self):
        import torch
        from ..models.model_seq2seq import Seq2SeqLMModel
        generated = torch.tensor([[0, 5, 6, 7, 5, 6], [0, 1, 2, 3, 4, 5]])
        scores = Seq2SeqLMModel._block_repeats(generated, torch.zeros(2, 10), 3)
        self.assertEqual(scores[0, 7].item(), float("-inf"))
        self.assertEqual(int(torch.isinf(scores[0]).sum()), 1)
        self.assertEqual(int(torch.isinf(scores[1]).sum()), 0)

    def test_lead_sentence_splits_english_and_hindi(self):
        from ..phases.support import lead_sentence
        self.assertEqual(lead_sentence("One here.  Two there."), "One here.")
        self.assertEqual(lead_sentence("क ख। ग"), "क ख।")


class PhaseSmokeTests(unittest.TestCase):
    def test_smoke_writes_every_phase_report_and_keeps_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main(["--protocol", "smoke", "--output", str(root)])
            for index in range(8):
                report = json.loads((root / f"phase{index}" / "report.json").read_text())
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["phase"], index)
                self.assertGreaterEqual(report["details"]["time"]["phase_seconds"], 0)
                self.assertIn("runs", report["details"])
                final = Path(report["details"]["final_model"])
                self.assertTrue((final / "metadata.json").is_file(), index)
                self.assertTrue((root / f"phase{index}" / "report.csv").is_file())
                self.assertTrue((root / f"phase{index}" / "report.md").is_file())
                self.assertEqual(json.loads((root / f"phase{index}" / "progress.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((root / "progress.json").read_text())["completed_phases"], list(range(8)))
            phase0 = json.loads((root / "phase0" / "report.json").read_text())["details"]
            self.assertTrue(phase0["trained"])
            self.assertTrue(phase0["generated"])
            self.assertIn("english", phase0["tokens_per_character"])
            self.assertIn("hindi", phase0["tokens_per_character"])
            phase1 = json.loads((root / "phase1" / "report.json").read_text())
            self.assertIn(phase1["details"]["gate"], {"continue", "stop_replay", "fix_setup", "fix_english"})
            self.assertIn("chrf", phase1["details"]["lead_1"]["english"])
            self.assertIn("script_share_of_forgetting", phase1["details"])
            self.assertEqual(phase1["context"]["hindi_only"]["status"], "completed")
            self.assertEqual(phase1["details"]["decision_split"], "validation")
            serving = json.loads((root / "phase1" / "serving" / "hindi_only-unmerged.json").read_text())
            self.assertEqual(len(serving["requests"]), 4)
            self.assertEqual(serving["summary"]["english"]["repeats"], 1)
            self.assertFalse(serving["summary"]["english"]["warmup"])
            predictions = root / "phase1" / "hindi_only" / "predictions" / "english-test-plain.json"
            self.assertGreater(len(json.loads(predictions.read_text())), 0)
            self.assertEqual(json.loads((root / "phase1" / "english" / "run.json").read_text())["status"], "completed")
            phase2 = json.loads((root / "phase2" / "report.json").read_text())["details"]
            self.assertEqual(set(phase2["arms"]), {"random", "loss", "drift", "past", "repair_stop"})
            self.assertIn(phase2["g3_parent"], {"loss", "drift"})
            phase4 = json.loads((root / "phase4" / "report.json").read_text())["details"]
            self.assertIn(phase4["status"], {"completed", "skipped", "rejected"})
            shrink = json.loads((root / "phase5" / "report.json").read_text())["context"]["shrink"]["random"]
            self.assertGreaterEqual(shrink["k_E"], 1)
            self.assertGreaterEqual(shrink["k_G"], 1)
            phase6 = json.loads((root / "phase6" / "report.json").read_text())["details"]
            self.assertEqual(phase6["status"], "completed")
            self.assertEqual(phase6["prefix"], phase6["stored_prefix"])
            phase3 = json.loads((root / "phase3" / "report.json").read_text())["details"]
            self.assertIn("compute_matched_extra_steps", phase3["conditional"])
            phase7 = json.loads((root / "phase7" / "report.json").read_text())["details"]
            self.assertIn("reverse_order", {item["name"] for item in phase7["skipped"]})
            self.assertTrue((root / "phase7" / "drift_past" / "run.json").is_file())
            original = (root / "phase0" / "report.json").read_bytes()
            later = (root / "phase7" / "report.json").read_bytes()
            main(["--protocol", "smoke", "--output", str(root)])
            self.assertEqual((root / "phase0" / "report.json").read_bytes(), original)
            self.assertEqual((root / "phase7" / "report.json").read_bytes(), later)
            (root / "phase1" / "report.csv").unlink()
            main(["--protocol", "smoke", "--output", str(root), "--phase", "1"])
            self.assertIn("epochs/completed", (root / "phase1" / "report.csv").read_text())
            hindi = next(item for item in phase1["details"]["runs"] if item["name"] == "hindi_only")
            self.assertGreaterEqual(hindi["epochs"]["completed"], 1)
            self.assertIn("chrf", next(iter(hindi["accuracies"].values())))
            random_run = (root / "phase2" / "random" / "run.json").read_bytes()
            weights = root / "phase0" / "final" / "batch" / "weights.pt"
            stamp = weights.stat().st_mtime_ns
            (root / "phase2" / "report.json").unlink()
            (root / "phase0" / "report.json").unlink()
            main(["--protocol", "smoke", "--output", str(root), "--phase", "2"])
            main(["--protocol", "smoke", "--output", str(root), "--phase", "0"])
            self.assertEqual((root / "phase2" / "random" / "run.json").read_bytes(), random_run)
            self.assertEqual(weights.stat().st_mtime_ns, stamp)
            resumed = json.loads((root / "phase2" / "report.json").read_text())
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(set(resumed["details"]["arms"]), {"random", "loss", "drift", "past", "repair_stop"})
            self.assertTrue((Path(resumed["details"]["final_model"]) / "metadata.json").is_file())
            self.assertEqual(json.loads((root / "phase0" / "report.json").read_text())["details"]["trained"], True)
            model = build_smoke_model()
            dataset = build_smoke_dataset(model)
            model.release()
            settings = PhaseSettings(protocol="scientific", batch_size=2, analysis_batch=2, english_passes=1,
                                     hindi_passes=1, max_new_tokens=8, shrink_tolerance=100.0)
            pool = load_pool(phase1["context"]["pool"])
            checkpoint = phase1["context"]["english_checkpoint"]
            pipeline = hindi_pipeline(settings, pool, checkpoint, selector="random")
            output = root / "three_seed"
            screened = run_hindi(output, "random", pipeline, dataset, build_smoke_model, settings)
            self.assertEqual([run["seed"] for run in screened["seed_runs"]], [1])
            with patch("src.phases.support.execute", wraps=execute) as trained:
                seeded = run_hindi(output, "random", pipeline, dataset, build_smoke_model, settings,
                                   seeds=settings.hindi_seeds)
            self.assertEqual(trained.call_count, 2)
            self.assertEqual([run["seed"] for run in seeded["seed_runs"]], [1, 2, 3])
            self.assertEqual(seeded["steps"], sum(run["steps"] for run in seeded["seed_runs"]))
            self.assertEqual(seeded["costs"]["forward_pass_units"],
                             sum(run["costs"]["forward_pass_units"] for run in seeded["seed_runs"]))
            for final in seeded["final_models"].values():
                self.assertTrue((Path(final) / "metadata.json").is_file())
            with patch("src.phases.support.execute", side_effect=AssertionError("finished seeds retrained")):
                cached = run_hindi(output, "random", pipeline, dataset, build_smoke_model, settings,
                                   seeds=settings.hindi_seeds)
            self.assertEqual(seeded["metrics"], cached["metrics"])


if __name__ == "__main__":
    unittest.main()
