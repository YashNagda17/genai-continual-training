"""Numerical checks for compressed teacher distributions."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from ..models.model_distillation import compress_probabilities, bucket_kl


class DistillationTests(unittest.TestCase):
    def test_all_but_one_matches_full_kl_and_gradient(self):
        torch.manual_seed(8)
        teacher = torch.randn(5, 9)
        student = torch.randn(5, 9, requires_grad=True)
        indices, probabilities = compress_probabilities(teacher, 8)
        actual = bucket_kl(student, indices, probabilities).mean()
        q = teacher.softmax(-1)
        expected = (q * (q.log() - student.log_softmax(-1))).sum(-1).mean()
        torch.testing.assert_close(actual, expected)
        first, = torch.autograd.grad(actual, student, retain_graph=True)
        second, = torch.autograd.grad(expected, student)
        torch.testing.assert_close(first, second)

    def test_matching_teacher_zero_and_tail_keeps_mass(self):
        torch.manual_seed(2)
        logits = torch.randn(5, 100, requires_grad=True)
        indices, q = compress_probabilities(logits.detach(), 4)
        torch.testing.assert_close(q.sum(-1), torch.ones(5))
        self.assertTrue(bool((q[:, -1] > 0).all()))
        loss = bucket_kl(logits, indices, q).mean()
        self.assertLess(abs(float(loss.detach())), 1e-6)
        loss.backward()
        self.assertLess(float(logits.grad.abs().max()), 1e-6)

    def test_paired_runner_and_completed_run_reuse(self):
        from .. import distillation_pilot
        from ..phases.data import build_smoke_model, build_smoke_dataset, _seeded_smoke_module
        from ..models.model_distillation import DistillationMT5
        from ..utils.state import read_json
        smoke = build_smoke_model()
        dataset = build_smoke_dataset(smoke)
        for language in ('english', 'hindi'):
            dataset.expose(f'{language}/validation', [r.sample_id for r in dataset.views['validation']
                                                     if r.task_id == language])

        class TinyKD(DistillationMT5):
            model_id = smoke.model_id
            expected_parameters = None

            def __init__(self, settings, **kwargs):
                super().__init__(settings, pretrained=False, config=smoke.config,
                    tokenizer=smoke.tokenizer, model_factory=_seeded_smoke_module, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset.save(root / 'prepared')
            smoke.save(root / 'english', 'english')
            smoke.release()
            argv = ['--prepared', str(root / 'prepared'), '--checkpoint', str(root / 'english'),
                    '--output', str(root / 'run'), '--steps', '2', '--hindi-batch', '1',
                    '--eval-examples', '1', '--max-new-tokens', '4', '--top-k', '4', '--threads', '1']
            with patch.object(distillation_pilot, 'DistillationMT5', TinyKD):
                original_evaluate = distillation_pilot.evaluate
                calls = []

                def interrupted_evaluation(*args):
                    calls.append(1)
                    if len(calls) == 2:
                        raise RuntimeError('simulated interruption after committed training')
                    return original_evaluate(*args)

                with patch.object(distillation_pilot, 'evaluate', interrupted_evaluation):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        distillation_pilot.main(argv)
                distillation_pilot.main(argv)
                report = read_json(root / 'run/report.json')
                self.assertEqual(report['seeds'][0]['arms']['replay_kd']['event'], 2)
                self.assertEqual(len(report['seeds'][0]['arms']['replay']['after']['english']['predictions']), 1)
                distillation_pilot.main(argv)
                self.assertEqual(report, read_json(root / 'run/report.json'))


if __name__ == '__main__':
    unittest.main()
