import tempfile
from pathlib import Path
import unittest
import numpy as np

from thingy.neural import Controller
from thingy.training import make_batch, train


class NeuralTests(unittest.TestCase):
    def test_exact_recurrent_gradients_match_finite_differences(self):
        model = Controller(seed=2)
        batch = make_batch(np.random.default_rng(17), 3, max_hops=2)
        loss, grads, _ = model.loss_and_grad(batch, teacher=0.2, auxiliary=0.3, ping_cost=0.004)
        self.assertTrue(np.isfinite(loss))
        for name, parameter in model.params.items():
            indices = [tuple(0 for _ in parameter.shape), tuple(s - 1 for s in parameter.shape)]
            for index in indices:
                original = parameter[index]
                parameter[index] = original + 1e-5
                plus = model.loss_and_grad(batch, teacher=0.2, auxiliary=0.3, ping_cost=0.004)[0]
                parameter[index] = original - 1e-5
                minus = model.loss_and_grad(batch, teacher=0.2, auxiliary=0.3, ping_cost=0.004)[0]
                parameter[index] = original
                self.assertAlmostEqual(grads[name][index], (plus - minus) / 2e-5, delta=1e-6)

    def test_training_improves_on_independent_fresh_graphs(self):
        batch = make_batch(np.random.default_rng(123456), 128, max_hops=4)
        initial = Controller(seed=5)
        before = initial.loss_and_grad(batch)[2]['accuracy']
        model, receipt = train(seed=5, steps=300, batch_size=48)
        after = model.loss_and_grad(batch)[2]['accuracy']
        self.assertGreater(after, 0.9)
        self.assertGreater(after, before + 0.6)
        self.assertEqual(receipt['phases'][-1]['teacher'], 0)
        self.assertEqual(receipt['phases'][-1]['auxiliary'], 0)

    def test_checkpoint_roundtrip_is_exact_and_rejects_nonfinite(self):
        model = Controller(seed=3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.json'
            model.save(path)
            restored = Controller.load(path)
            for k in model.params:
                np.testing.assert_equal(model.params[k], restored.params[k])
            data = model.to_dict()
            data['params']['W1'][0][0] = float('nan')
            with self.assertRaises(ValueError):
                Controller.from_dict(data)
            data = model.to_dict()
            data['version'] = 999
            with self.assertRaises(ValueError):
                Controller.from_dict(data)

    def test_inference_does_not_update_permanent_weights(self):
        model = Controller(seed=0)
        copies = {k: v.copy() for k, v in model.params.items()}
        p = model.predict(np.zeros((1, model.input_size)))
        self.assertAlmostEqual(float(p.sum()), 1)
        model.decode(np.eye(16)[0])
        for k in copies:
            np.testing.assert_equal(copies[k], model.params[k])

    def test_extreme_checkpoint_values_fail_recoverably(self):
        model = Controller(seed=3)
        for value in (1e308, 10 ** 1000):
            data = model.to_dict()
            data['params']['W2'][0][0] = value
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                Controller.from_dict(data)

    def test_poisoned_runtime_weights_cannot_become_a_normal_address(self):
        model = Controller(seed=3)
        model.params['W2'].fill(1e308)
        with self.assertRaises(ValueError):
            model.predict(np.ones((1, model.input_size)))


if __name__ == '__main__':
    unittest.main()
