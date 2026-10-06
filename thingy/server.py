"""Loopback explorer. Each browser session owns its agent, lock, and snapshots."""

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse
import uuid
import numpy as np

from .engine import Agent, COMPONENTS, load_default_controller
from .tasks import parse_episode, generate_episode

DEFAULT_FACTS = 'Zapp means tired\ntired helps rest\nZapp next cook\ncook helps soup'
DEFAULT_QUERY = 'Zapp means helps'
MAX_BODY = 65536


@dataclass
class Session:
    agent: Agent
    snapshots: dict = field(default_factory=dict)
    seen: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.RLock)


def create_server(host='127.0.0.1', port=8765, controller=None):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('The explorer binds to loopback: use 127.0.0.1 or localhost.')
    controller = controller or load_default_controller()
    sessions, table_lock = {}, threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200, content_type='application/json; charset=utf-8'):
            raw = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                self.reply((Path(__file__).parent / 'web' / 'index.html').read_bytes(),
                           content_type='text/html; charset=utf-8')
            elif path == '/api/health':
                self.reply(dict(ok=True, parameters=controller.parameter_count, kind='trained NumPy controller'))
            else:
                self.reply(dict(error='Unknown route.'), 404)

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    self.reply(dict(error='Request body must be 1–65536 bytes.'), 413)
                    return
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Send JSON with Content-Type: application/json.')
                origin = self.headers.get('Origin')
                allowed = (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}')
                if origin and origin not in allowed:
                    raise ValueError('Origin must match this local explorer.')
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object.')
                route = urlparse(self.path).path
                if route == '/api/new':
                    if data.get('random', False):
                        seed = data.get('seed', 7)
                        if type(seed) is not int or not 0 <= seed <= 2 ** 32:
                            raise ValueError('Random seed must be a nonnegative integer.')
                        ep = generate_episode(np.random.default_rng(seed), data.get('hops', 4))
                    else:
                        ep = parse_episode(data.get('facts', DEFAULT_FACTS), data.get('query', DEFAULT_QUERY))
                    agent = Agent(controller, ep)
                    previous = data.get('session')
                    sid = previous if isinstance(previous, str) else uuid.uuid4().hex
                    with table_lock:
                        expired = [key for key, s in sessions.items() if time.monotonic() - s.seen > 1800]
                        for key in expired:
                            del sessions[key]
                        if sid in sessions:
                            current = sessions[sid]
                            with current.lock:
                                current.agent = agent
                                current.snapshots.clear()
                                current.seen = time.monotonic()
                        else:
                            sid = uuid.uuid4().hex
                            if len(sessions) >= 64:
                                raise ValueError('Session limit reached. Restart the local server or wait for expiry.')
                            sessions[sid] = Session(agent)
                    self.reply(dict(session=sid, view=agent.view(), facts=ep.facts_text(), query=ep.query_text()))
                    return
                if route not in ('/api/step', '/api/run', '/api/snapshot', '/api/restore', '/api/interrupt'):
                    self.reply(dict(error='Unknown route.'), 404)
                    return
                sid = data.get('session')
                if not isinstance(sid, str):
                    raise ValueError('Create a session first.')
                with table_lock:
                    if sid not in sessions:
                        raise ValueError('Session is missing or expired; start a new episode.')
                    session = sessions[sid]
                    session.seen = time.monotonic()
                with session.lock:
                    extra = {}
                    if route == '/api/step':
                        session.agent.step(data.get('intervention'))
                    elif route == '/api/run':
                        session.agent.run()
                    elif route == '/api/interrupt':
                        session.agent.interrupt()
                    elif route == '/api/snapshot':
                        if len(session.snapshots) >= 16:
                            del session.snapshots[next(iter(session.snapshots))]
                        snapshot_id = uuid.uuid4().hex
                        session.snapshots[snapshot_id] = session.agent.snapshot()
                        extra['snapshot'] = snapshot_id
                    elif route == '/api/restore':
                        snapshot_id = data.get('snapshot')
                        if not isinstance(snapshot_id, str) or snapshot_id not in session.snapshots:
                            raise ValueError('Snapshot is missing or belongs to another session.')
                        if type(data.get('interrupt', False)) is not bool:
                            raise ValueError('interrupt must be true or false.')
                        # Build a branch and validate before replacing live state.
                        trial = Agent(controller, session.agent.episode, session.agent.max_cycles)
                        if data.get('interrupt', False):
                            trial.interrupt()
                        else:
                            trial.restore(session.agent.snapshot())
                        trial.restore(session.snapshots[snapshot_id], data.get('components', COMPONENTS))
                        session.agent = trial
                    self.reply(dict(session=sid, view=session.agent.view(), **extra))
            except (ValueError, TypeError, KeyError, UnicodeError) as e:
                self.reply(dict(error=str(e)), 400)

    return ThreadingHTTPServer((host, port), Handler)
