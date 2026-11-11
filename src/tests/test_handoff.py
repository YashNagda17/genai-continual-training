"""Portable, explicitly exploratory handoff; no model loading or training."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ..phase2_from_checkpoint import main, rebase_paths, write_training_report
from ..phases.settings import PhaseSettings


class HandoffTests(unittest.TestCase):
    def test_phase2_report_keeps_the_failed_source_gate(self):
        from ..phases.phase2 import run
        with tempfile.TemporaryDirectory() as directory:
            settings = PhaseSettings(protocol="pilot", seed=1)
            metrics = {"english/validation/plain": {"chrf": 20.}, "hindi/validation/plain": {"chrf": 22.}}
            provenance = {"exploratory": True, "source_gate": "fix_english"}
            context = {"pool": "pool", "english_checkpoint": "english", "english_metrics": metrics,
                       "phase1_handoff": provenance}
            def fake_run(output, name, pipeline, *args, **kwargs):
                return {"name": name, "seed": 1, "metrics": metrics, "selector": pipeline.config.selector,
                        "ranking": ["a", "b"], "filtered": [], "retired": [], "update_seconds": 1.}
            with patch("src.phases.phase2.load_pool", return_value=()), \
                    patch("src.phases.phase2.run_hindi", side_effect=fake_run), \
                    patch("src.phases.phase2.compare_arms", return_value={}):
                run(Path(directory), SimpleNamespace(fingerprint="prepared-source"), None, settings, context)
            saved = json.loads((Path(directory) / "phase2/report.json").read_text())
            self.assertEqual(saved["details"]["phase1_handoff"], provenance)
            self.assertEqual(saved["status"], "completed")

    def test_only_source_owned_paths_are_rebased_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {"checkpoint": "/home/yash/run/phase1/english",
                        "seeds": [{"attempt": "/home/yash/run/phase1/seed-1"}],
                        "other": "/home/yash/run-other/weights", "text": "ordinary summary"}
            actual = rebase_paths(original, "/home/yash/run", Path(directory))
            self.assertEqual(actual["checkpoint"], str((Path(directory) / "phase1/english").resolve()))
            self.assertEqual(actual["other"], original["other"])
            self.assertEqual(actual["text"], original["text"])
            self.assertEqual(original["checkpoint"], "/home/yash/run/phase1/english")

    def inputs(self):
        return (SimpleNamespace(), PhaseSettings(protocol="scientific"), {}, {},
                {"phase1": {"gate": "fix_english"}}, {"weights.pt": "source-hash"})

    def test_prepare_only_preserves_failed_gate_and_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["--exploratory", "--source", str(root / "source"), "--output", str(root / "output")]
            with patch("src.phase2_from_checkpoint.inspect_source", return_value=self.inputs()), \
                    patch("src.phase2_from_checkpoint.write_training_report"), \
                    patch("src.phase2_from_checkpoint.run_phase2") as run, redirect_stdout(io.StringIO()):
                main(argv)
                run.assert_not_called()
                path = root / "output/handoff.json"
                before = path.read_bytes()
                saved = json.loads(before)
                self.assertEqual(saved["source_root"], "../source")
                self.assertEqual(saved["provenance"]["source_gate"], "fix_english")
                self.assertEqual(saved["settings"]["protocol"], "pilot")
                self.assertEqual(saved["settings"]["seed"], 1)
                changed = list(self.inputs())
                changed[-1] = {"weights.pt": "changed-hash"}
                with patch("src.phase2_from_checkpoint.inspect_source", return_value=tuple(changed)):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        main(argv)
                self.assertEqual(path.read_bytes(), before)

    def test_launch_is_explicit_and_passes_exploratory_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["--exploratory", "--run", "--source", str(root / "source"), "--output", str(root / "output")]
            with patch("src.phase2_from_checkpoint.inspect_source", return_value=self.inputs()), \
                    patch("src.phase2_from_checkpoint.write_training_report"), \
                    patch("src.phase2_from_checkpoint.run_phase2") as run, \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.device_count", return_value=1), redirect_stdout(io.StringIO()):
                main(argv)
                run.assert_called_once()
                settings, context = run.call_args.args[-2:]
                self.assertEqual(settings.protocol, "pilot")
                self.assertEqual(settings.seed, 1)
                self.assertTrue(context["phase1_handoff"]["exploratory"])
                self.assertEqual(context["phase1"]["gate"], "fix_english")

    def test_explicit_exploratory_flag_is_required(self):
        with patch("src.phase2_from_checkpoint.inspect_source") as inspect, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main([])
            inspect.assert_not_called()

    def test_training_report_does_not_mark_interrupted_phase_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            folder = source / "phase1"
            folder.mkdir()
            progress = folder / "progress.json"
            progress.write_text('{"status":"interrupted"}')
            metrics = {f"{language}/{split}/plain": {"chrf": 20., "rouge_l": 15., "right_language": 100.}
                       for language in ("english", "hindi") for split in ("validation", "test")}
            english = {"metrics": metrics, "time": {}, "steps": 500,
                       "update_seconds": 60., "wall_clock_seconds": 120.}
            hindi = dict(english, steps=1500, metric_std={key: {m: .2 for m in value} for key, value in metrics.items()})
            context = {"hindi_only": hindi, "phase1": {"gate": "fix_english", "english_validation_chrf": 20.,
                       "lead_1": {"english": {"chrf": 22.}}, "forget": 6., "gain": 8., "past_replay_eligible": 400}}
            write_training_report(source, PhaseSettings(protocol="scientific"), english, context)
            self.assertEqual(json.loads(progress.read_text())["status"], "interrupted")
            self.assertFalse((folder / "report.json").exists())
            saved = json.loads((folder / "training-report.json").read_text())
            self.assertEqual(saved["scientific_gate"], "fix_english")
            self.assertIn("serving_incomplete", saved["status"])
            self.assertIn("fix_english", (folder / "training-report.md").read_text(encoding="utf-8"))
