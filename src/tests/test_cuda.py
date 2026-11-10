"""NVIDIA execution checks with tiny local weights; no downloads or CPU substitution."""

from pathlib import Path
import tempfile
import unittest

import torch

from ..summarization_experiment import prepare
from ..utils.contracts import Objective, ScoreSpec, TokenTargets, TrainLimits, TrainingBatch, UpdateHooks


@unittest.skipUnless(torch.version.cuda is not None and torch.cuda.is_available(),
                     "An NVIDIA CUDA build and device are required")
class CUDAExecutionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset, self.factory, _ = prepare("smoke", "cuda:0")
        self.role = self.factory()
        self.addCleanup(self.role.release)
        self.role.bind_runtime(self.root / "learner", None)
        self.role.bind_dataset(self.dataset)
        rows = (self.dataset.training_view("english")[0], self.dataset.training_view("hindi")[0])
        self.batch = self.dataset.collate(rows)

    def train_once(self, role, hooks=None):
        return role.train(1, batch_factory=lambda: [TrainingBatch(self.batch, ("current", "current"))],
                          objective=Objective("per_example_mean", token_mode="weighted" if hooks else "all"),
                          hooks=hooks, limits=TrainLimits(1, 1, self.batch.valid_tokens))[0]

    def test_weighted_update_scoring_and_generation_use_cuda(self):
        self.assertEqual(self.role.device, torch.device("cuda:0"))
        self.role.activate()
        payload = self.role._payload(self.batch)
        self.assertTrue(all(value.is_cuda for value in payload.values()))
        self.assertEqual(payload["supervised_indices"].dtype, torch.int64)
        self.assertEqual(payload["attention_mask"].dtype, torch.bool)

        def targets(_batch, nll):
            self.assertTrue(nll.is_cuda)
            self.assertFalse(nll.requires_grad)
            return TokenTargets(torch.zeros_like(nll), torch.ones_like(nll), all_positive=True)

        report = self.train_once(self.role, UpdateHooks(token_targets=targets))
        self.assertTrue(report.success)
        self.assertIsInstance(report.loss_sums["supervised"], float)
        self.assertEqual(report.loss_counts["supervised"], len(self.batch.records))
        self.assertTrue(report.diagnostics["token_surprise"].is_cuda)
        score = self.role.score(self.batch, ScoreSpec("token_loss", "none"))
        self.assertTrue(score.values.is_cuda)
        self.assertEqual(score.values.numel(), self.batch.supervised_tokens)
        self.assertTrue(torch.isfinite(score.values).all().item())
        generated = self.role.predict(self.dataset.collate_sources(self.batch.records),
                                      {"max_new_tokens": 3, "no_repeat_ngram_size": 2})
        self.assertEqual(generated["ids"], self.batch.ids)
        self.assertEqual(len(generated["texts"]), len(self.batch.records))
        self.assertTrue(self.train_once(self.role).success)

    def test_positive_weight_reduction_does_not_synchronize(self):
        flat = torch.tensor([2., 4., 6.], device="cuda:0", requires_grad=True)
        weights = torch.ones_like(flat)
        counts = torch.tensor([1, 2], device="cuda:0")
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            loss, _, count = self.role._weighted_loss(flat, TokenTargets(None, weights, all_positive=True), counts)
            loss.backward()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        self.assertEqual(count, 2)
        self.assertEqual(loss.item(), 3.5)

    def test_resume_preserves_next_dropout_update_and_cuda_rng(self):
        self.role.config.dropout_rate = 0.1
        self.role.activate()
        caller_rng = torch.cuda.get_rng_state(self.role.device).clone()
        self.train_once(self.role)
        self.assertTrue(torch.equal(caller_rng, torch.cuda.get_rng_state(self.role.device)))
        checkpoint = self.root / "checkpoint"
        self.role.save(checkpoint, "resume")
        restored = self.factory()
        self.addCleanup(restored.release)
        restored.config.dropout_rate = 0.1
        restored.bind_runtime(self.root / "restored", None)
        restored.bind_dataset(self.dataset)
        restored.load(checkpoint, "resume")
        restored.activate()
        self.train_once(self.role)
        self.train_once(restored)
        for name, parameter in self.role.module.named_parameters():
            torch.testing.assert_close(parameter, restored.module.get_parameter(name), msg=name)
        self.assertEqual(self.role.steps, restored.steps)
        self.assertEqual(self.role.scheduler.state_dict(), restored.scheduler.state_dict())
        self.assertTrue(torch.equal(self.role.rng_state["device"], restored.rng_state["device"]))
        self.assertTrue(torch.equal(caller_rng, torch.cuda.get_rng_state(self.role.device)))


if __name__ == "__main__":
    unittest.main()
