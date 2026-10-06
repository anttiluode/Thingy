import tempfile
from pathlib import Path
import unittest

from thingy.benchmark import benchmark, mechanism_gate, write_report, text_feedback, latent_feedback
from thingy.engine import load_default_controller
from thingy.tasks import parse_episode


class BenchmarkTests(unittest.TestCase):
    def test_actual_feedback_arms_and_one_read_limit(self):
        model = load_default_controller()
        ep = parse_episode('Zapp means tired\ntired helps rest', 'Zapp means helps')
        self.assertEqual(ep.names[text_feedback(model, ep)['prediction']], 'rest')
        self.assertEqual(ep.names[latent_feedback(model, ep)['prediction']], 'rest')
        self.assertEqual(ep.names[text_feedback(model, ep, one_read=True)['prediction']], 'tired')
        self.assertGreater(text_feedback(model, ep)['bytes'], 0)

    def test_paired_denominators_and_frozen_reproducibility(self):
        model = load_default_controller()
        a = benchmark(model, seed=101, episodes=35)
        b = benchmark(model, seed=101, episodes=35)
        self.assertEqual(a, b)
        self.assertEqual(a['held_out']['self_ping']['total'], 35)
        self.assertEqual(a['causal']['intact']['total'], 25)
        for name in ('delete', 'wrong', 'freeze', 'replay'):
            self.assertEqual(a['causal'][name]['total'], 25)
        self.assertEqual(a['memory']['codebook+residue']['total'], 25)
        self.assertEqual(a['memory']['workspace+residue']['correct'], 0)
        self.assertEqual(a['memory']['all']['accuracy'], 1)
        self.assertEqual(a['memory']['codebook+residue']['accuracy'], 1)
        self.assertTrue(a['gate']['passed'])

    def test_gate_does_not_hide_failures(self):
        passed = mechanism_gate(0.95, 0.95, 0.6, 0.5, 1.0)
        self.assertTrue(passed['passed'])
        for values in [(0.5, 0.95, 0.6, 0.5, 1), (0.95, 0.95, 0.9, 0.5, 1),
                       (0.95, 0.95, 0.6, 0.9, 1), (0.95, 0.95, 0.6, 0.5, 0.8)]:
            self.assertFalse(mechanism_gate(*values)['passed'])

    def test_report_writes_json_and_honest_markdown(self):
        report = benchmark(load_default_controller(), seed=111, episodes=14)
        with tempfile.TemporaryDirectory() as d:
            write_report(report, d)
            self.assertTrue((Path(d) / 'benchmark.json').exists())
            text = (Path(d) / 'REPORT.md').read_text()
            self.assertIn('controller ablations', text)
            self.assertIn('designed', text)


if __name__ == '__main__':
    unittest.main()
