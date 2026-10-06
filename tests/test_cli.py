import subprocess
import sys
import unittest


class CLITests(unittest.TestCase):
    def test_demo_runs_trained_packets_without_an_api(self):
        p = subprocess.run([sys.executable, '-m', 'thingy', 'demo'], capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('self-ping', p.stdout)
        self.assertIn('rest', p.stdout)

    def test_bad_query_is_a_useful_cli_error(self):
        p = subprocess.run([sys.executable, '-m', 'thingy', 'demo', '--query', 'bad gibberish'],
                           capture_output=True, text=True, timeout=20)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('Query', p.stderr)


if __name__ == '__main__':
    unittest.main()
