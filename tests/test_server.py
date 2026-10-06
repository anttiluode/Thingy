import json
import threading
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from thingy.server import create_server
from thingy.engine import load_default_controller

FACTS = 'Zapp means tired\ntired helps rest\nZapp next cook\ncook helps soup'


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(port=0, controller=load_default_controller())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, route, data):
        request = Request(self.base + route, data=json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as r:
            return json.load(r)

    def new(self):
        return self.request('/api/new', {'facts': FACTS, 'query': 'Zapp means helps'})

    def test_real_sessions_edits_and_separate_public_output(self):
        a, b = self.new(), self.new()
        self.assertNotEqual(a['session'], b['session'])
        sid = a['session']
        first = self.request('/api/step', {'session': sid, 'intervention': {'relation': 'next'}})
        self.assertIsNone(first['view']['public'])
        done = self.request('/api/run', {'session': sid})
        self.assertEqual(done['view']['answer'], 'soup')
        other = self.request('/api/run', {'session': b['session']})
        self.assertEqual(other['view']['answer'], 'rest')

    def test_snapshot_restore_drops_facts_then_restores_selected_state(self):
        sid = self.new()['session']
        self.request('/api/step', {'session': sid})
        saved = self.request('/api/snapshot', {'session': sid})['snapshot']
        restored = self.request('/api/restore', {'session': sid, 'snapshot': saved,
                                               'components': ['codebook', 'residue'], 'interrupt': True})
        self.assertEqual(restored['view']['active_bindings'], 4)
        result = self.request('/api/run', {'session': sid})
        self.assertEqual(result['view']['answer'], 'rest')
        self.request('/api/restore', {'session': sid, 'snapshot': saved,
                                     'components': ['workspace', 'residue'], 'interrupt': True})
        result = self.request('/api/run', {'session': sid})
        self.assertIsNone(result['view']['answer'])

    def test_bad_requests_and_unknown_routes_are_errors(self):
        for route, payload in [('/api/new', {'facts': '', 'query': 'Zapp'}),
                               ('/api/run', {'session': 'missing'}), ('/api/missing', {}),
                               ('/api/new', {'facts': 'x' * 70000, 'query': 'Zapp'})]:
            with self.subTest(route=route), self.assertRaises(HTTPError) as e:
                self.request(route, payload)
            self.assertIn(e.exception.code, (400, 404, 413))
        bad = Request(self.base + '/api/new', data=b'{', headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError):
            urlopen(bad, timeout=5)

    def test_page_and_health_are_packaged(self):
        with urlopen(self.base, timeout=5) as r:
            self.assertIn('Thingy', r.read().decode())
        with urlopen(self.base + '/api/health', timeout=5) as r:
            self.assertGreater(json.load(r)['parameters'], 0)

    def test_invalid_restore_does_not_destroy_live_state(self):
        sid = self.new()['session']
        saved = self.request('/api/snapshot', {'session': sid})['snapshot']
        with self.assertRaises(HTTPError):
            self.request('/api/restore', {'session': sid, 'snapshot': saved,
                                        'components': ['bad'], 'interrupt': True})
        result = self.request('/api/run', {'session': sid})
        self.assertEqual(result['view']['answer'], 'rest')

    def test_reloading_a_world_reuses_its_session_without_exhaustion(self):
        sid = self.new()['session']
        for _ in range(65):
            result = self.request('/api/new', {'session': sid, 'facts': FACTS, 'query': 'Zapp means helps'})
            self.assertEqual(result['session'], sid)
        result = self.request('/api/run', {'session': sid})
        self.assertEqual(result['view']['answer'], 'rest')

    @unittest.skipUnless(os.environ.get('CODEX_PRIMARY_RUNTIME_NODE') or shutil.which('node'),
                         'Optional explorer control test needs Node; engine tests need only Python/NumPy.')
    def test_real_watch_handler_cancels_the_old_sleeping_loop(self):
        sid = self.request('/api/new', {'random': True, 'seed': 11, 'hops': 6})['session']
        root = Path(__file__).resolve().parents[1]
        node = os.environ.get('CODEX_PRIMARY_RUNTIME_NODE') or shutil.which('node')
        result = subprocess.run([node, str(root/'tests/ui_watch_probe.js'),
                                 str(root/'thingy/web/index.html'), self.base, sid],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['cycles'], 2)


if __name__ == '__main__':
    unittest.main()
