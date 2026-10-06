import copy
import unittest
from unittest.mock import patch
import numpy as np

from thingy.engine import Agent, load_default_controller
from thingy.tasks import parse_episode, Episode

FACTS = 'Zapp means tired\ntired helps rest\nZapp next cook\ncook helps soup'


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_default_controller()

    def agent(self, query='Zapp means helps', **kwargs):
        return Agent(self.model, parse_episode(FACTS, query), **kwargs)

    def test_packets_write_geometry_before_the_next_state(self):
        agent = self.agent()
        before = agent.fast.copy()
        packet = agent.propose()
        self.assertEqual(packet['relation'], 'means')
        agent.step()
        self.assertGreater(np.linalg.norm(agent.fast - before), 0.9)
        self.assertEqual(agent.episode.names[agent.h.argmax()], 'tired')
        self.assertIsNone(agent.view()['public'])
        agent.run()
        self.assertEqual(agent.view()['answer'], 'rest')
        self.assertEqual(agent.status, 'complete')

    def test_wrong_ping_changes_answer_and_replay_restores(self):
        agent = self.agent()
        saved = agent.snapshot()
        agent.step({'kind': 'replace', 'relation': 'next'})
        agent.run()
        self.assertEqual(agent.view()['answer'], 'soup')
        agent.restore(saved)
        agent.run()
        self.assertEqual(agent.view()['answer'], 'rest')

    def test_delete_and_freeze_are_causal(self):
        for kind in ('drop', 'freeze'):
            agent = self.agent()
            before = agent.fast.copy()
            agent.step({'kind': kind})
            np.testing.assert_equal(agent.fast, before)
            self.assertEqual(agent.episode.names[agent.h.argmax()], 'Zapp')
            agent.run()
            self.assertEqual(agent.status, 'unknown')
            self.assertIsNone(agent.view()['answer'])

    def test_source_address_is_writable(self):
        agent = self.agent()
        agent.step({'source': 'Zapp', 'relation': 'next'})
        agent.run()
        self.assertEqual(agent.view()['answer'], 'soup')

    def test_codebook_residue_restore_after_interrupt(self):
        agent = self.agent()
        agent.step()
        saved = agent.snapshot()
        agent.interrupt()
        self.assertEqual(float(agent.bindings.sum()), 0)
        agent.restore(saved, ('codebook', 'residue'))
        agent.run()
        self.assertEqual(agent.view()['answer'], 'rest')
        agent.interrupt()
        agent.restore(saved, ('workspace', 'residue'))
        agent.run()
        self.assertEqual(agent.status, 'unknown')

    def test_snapshots_branch_without_aliasing_and_validate_atomically(self):
        agent = self.agent()
        saved = agent.snapshot()
        broken = copy.deepcopy(saved)
        broken['codebook']['fast'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            agent.restore(broken)
        self.assertEqual(agent.snapshot(), saved)
        agent.step()
        self.assertEqual(saved['residue']['cursor'], 0)
        alien = self.agent('tired helps').snapshot()
        with self.assertRaises(ValueError):
            agent.restore(alien)
        with self.assertRaises(ValueError):
            agent.restore(saved, ('unrecognized',))

    def test_zero_hop_silence_and_bounded_unfinished_work(self):
        agent = self.agent('Zapp')
        agent.run()
        self.assertEqual(agent.view()['answer'], 'Zapp')
        self.assertEqual(agent.ping_count, 0)
        capped = self.agent(max_cycles=1)
        capped.run()
        self.assertEqual(capped.status, 'budget_exhausted')
        self.assertIsNone(capped.view()['answer'])

    def test_runtime_never_uses_target_or_mutates_theta(self):
        agent = self.agent()
        theta = {k: v.copy() for k, v in self.model.params.items()}
        with patch.object(Episode, 'target', side_effect=AssertionError('oracle leak')):
            agent.run()
        for k in theta:
            np.testing.assert_equal(theta[k], self.model.params[k])

    def test_unknown_edges_abstain_instead_of_inventing_facts(self):
        agent = Agent(self.model, parse_episode('Zapp means tired', 'tired helps'))
        agent.run()
        self.assertEqual(agent.status, 'unknown')
        self.assertIsNone(agent.view()['answer'])

    def test_inconsistent_terminal_snapshot_is_rejected_without_mutation(self):
        agent = self.agent()
        saved = agent.snapshot()
        for status in ('complete', 'halted_early', 'unknown', 'budget_exhausted'):
            invalid = copy.deepcopy(saved)
            invalid['status'] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                agent.restore(invalid)
            self.assertEqual(agent.snapshot(), saved)
        done = self.agent()
        done.run()
        invalid = done.snapshot()
        invalid['status'] = 'halted_early'
        with self.assertRaises(ValueError):
            done.restore(invalid)

    def test_invalid_empty_interventions_are_rejected_without_a_step(self):
        for value in ([], '', False, 0):
            agent = self.agent()
            saved = agent.snapshot()
            with self.subTest(value=value), self.assertRaises(ValueError):
                agent.step(value)
            self.assertEqual(agent.snapshot(), saved)

    def test_numeric_snapshot_overflow_is_a_validation_error(self):
        agent = self.agent()
        saved = agent.snapshot()
        invalid = copy.deepcopy(saved)
        invalid['codebook']['fast'][0][0] = 10 ** 1000
        with self.assertRaises(ValueError):
            agent.restore(invalid)
        self.assertEqual(agent.snapshot(), saved)

    def test_full_length_queries_keep_addresses_in_the_trained_input_range(self):
        ep = parse_episode('a helps b\nb helps a', 'a ' + 'supports ' * 32)
        agent = Agent(self.model, ep)
        agent.run()
        self.assertEqual(agent.status, 'complete')
        self.assertEqual(agent.view()['answer'], 'a')
        self.assertEqual(agent.ping_count, 32)


if __name__ == '__main__':
    unittest.main()
