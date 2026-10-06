import unittest
import numpy as np

from thingy.tasks import Episode, generate_episode, parse_episode


class TaskTests(unittest.TestCase):
    def test_composition_uses_episode_definitions(self):
        ep = parse_episode('Zapp means tired\ntired helps rest', 'Zapp means helps')
        self.assertEqual(ep.names[ep.target()], 'rest')
        other = parse_episode('Zapp means cook\ncook helps soup', 'Zapp means helps')
        self.assertEqual(other.names[other.target()], 'soup')

    def test_zero_hops_and_synonyms(self):
        ep = parse_episode('Zapp means tired', 'Zapp')
        self.assertEqual(ep.names[ep.target()], 'Zapp')
        ep = parse_episode('Zapp means tired', 'Zapp defines')
        self.assertEqual(ep.names[ep.target()], 'tired')

    def test_missing_edges_are_unknown(self):
        ep = parse_episode('Zapp means tired', 'tired helps')
        self.assertIsNone(ep.target())

    def test_rejects_conflicting_facts_and_unknown_queries(self):
        for facts, query in [('Zapp means tired\nZapp means rest', 'Zapp means'),
                             ('Zapp flies tired', 'Zapp'), ('Zapp means tired', 'missing means'),
                             ('', 'Zapp'), ('Zapp means tired', 'Zapp flies')]:
            with self.subTest(facts=facts, query=query), self.assertRaises(ValueError):
                parse_episode(facts, query)

    def test_rejects_capacity_overflow(self):
        with self.assertRaises(ValueError):
            parse_episode('\n'.join(f'n{i} means n{i+1}' for i in range(17)), 'n0 means')
        with self.assertRaises(ValueError):
            parse_episode('Zapp means tired', 'Zapp ' + 'means '*33)

    def test_rng_and_serialization_are_reproducible(self):
        a = generate_episode(np.random.default_rng(7), 4)
        b = generate_episode(np.random.default_rng(7), 4)
        c = generate_episode(np.random.default_rng(8), 4)
        np.testing.assert_equal(a.bindings, b.bindings)
        self.assertFalse(np.array_equal(a.bindings, c.bindings))
        restored = Episode.from_dict(a.to_dict())
        self.assertEqual(a.target(), restored.target())
        self.assertFalse(a.bindings.flags.writeable)

    def test_nonfinite_or_invalid_graph_is_rejected(self):
        ep = generate_episode(np.random.default_rng(0), 1)
        data = ep.to_dict()
        data['bindings'][0][0][0] = float('nan')
        with self.assertRaises(ValueError):
            Episode.from_dict(data)


if __name__ == '__main__':
    unittest.main()
