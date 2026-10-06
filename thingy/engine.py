"""The causal workspace -> self-ping -> fast geometry -> private state loop."""

import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from .neural import Controller, HALT, features
from .tasks import N_OBJECTS, RELATIONS, INSTRUCTIONS

COMPONENTS = ('codebook', 'workspace', 'residue', 'private')
STATUSES = ('thinking', 'complete', 'unknown', 'halted_early', 'budget_exhausted')


def load_default_controller():
    return Controller.load(Path(__file__).parent / 'assets' / 'controller.json')


def sparse(x, k=1):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    out = np.zeros_like(x)
    indices = np.argsort(x)[-k:]
    out[indices] = x[indices]
    mass = float(out.sum())
    return out / mass if mass > 1e-12 else out


def _array(value, shape, nonnegative=False):
    try:
        a = np.array(value, dtype=float, copy=True)
    except (OverflowError, TypeError) as e:
        raise ValueError('Snapshot contains invalid or out-of-range numbers.') from e
    if (a.shape != shape or not np.isfinite(a).all() or (np.abs(a) > 1e6).any()
            or (nonnegative and (a < -1e-9).any())):
        raise ValueError('Invalid or nonfinite snapshot array.')
    return a


class Agent:
    def __init__(self, controller, episode, max_cycles=40):
        if type(max_cycles) is not int or not 1 <= max_cycles <= 128:
            raise ValueError('max_cycles must be an integer from 1 to 128.')
        self.controller, self.episode, self.max_cycles = controller, episode, max_cycles
        self.signature = hashlib.sha256(json.dumps(episode.to_dict(), sort_keys=True).encode()).hexdigest()
        self.model_id = hashlib.sha256(json.dumps(controller.to_dict(), sort_keys=True).encode()).hexdigest()
        self.anchor = np.eye(N_OBJECTS)[episode.source]
        self.bindings = episode.bindings.copy()
        self.fast = np.eye(N_OBJECTS)
        self.h = self.anchor.copy()
        self.workspace = None
        self.cursor, self.cycles, self.ping_count = 0, 0, 0
        self.status, self.reason, self.trace = 'thinking', '', []

    def propose(self):
        if self.status != 'thinking':
            return None
        token = self.episode.program[self.cursor] if self.cursor < len(self.episode.program) else -1
        x = features(np.array([token]), self.h[None, :], np.array([max(0, len(self.episode.program) - self.cursor)]))
        probabilities = self.controller.predict(x)[0]
        action = int(probabilities.argmax())
        return dict(source=sparse(self.h).tolist(), relation='HALT' if action == HALT else RELATIONS[action],
                    probabilities=probabilities.tolist(), instruction=None if token < 0 else INSTRUCTIONS[token])

    def step(self, intervention=None):
        if self.status != 'thinking':
            return self.view()
        if self.cycles >= self.max_cycles:
            self.status, self.reason = 'budget_exhausted', 'Internal cycle budget reached with work unfinished.'
            return self.view()
        intervention = {} if intervention is None else intervention
        if not isinstance(intervention, dict) or intervention.get('kind', 'normal') not in ('normal', 'replace', 'drop', 'freeze'):
            raise ValueError('Intervention must be normal, replace, drop, or freeze.')
        if set(intervention) - {'kind', 'source', 'relation'}:
            raise ValueError('Unrecognized workspace edit.')
        packet = self.propose()
        kind = intervention.get('kind', 'normal')
        if 'relation' in intervention:
            if intervention['relation'] not in (*RELATIONS, 'HALT'):
                raise ValueError('Unknown relation edit.')
            packet['relation'] = intervention['relation']
        if 'source' in intervention:
            if intervention['source'] not in self.episode.names:
                raise ValueError('Unknown source edit.')
            packet['source'] = np.eye(N_OBJECTS)[self.episode.names.index(intervention['source'])].tolist()
        before = self.h.copy()
        old_fast = self.fast.copy()
        self.cycles += 1
        if packet['relation'] == 'HALT':
            self.status = 'complete' if self.cursor >= len(self.episode.program) else 'halted_early'
            self.reason = 'Learned HALT address.' if self.status == 'complete' else 'Controller halted with unfinished residue.'
            packet['candidate'] = self.h.tolist()
        else:
            self.ping_count += 1
            relation = RELATIONS.index(packet['relation'])
            key = np.array(packet['source'])
            candidate = self.bindings[relation] @ key
            packet['candidate'] = candidate.tolist()
            if kind not in ('drop', 'freeze'):
                if candidate.sum() < 1 - 1e-8:
                    self.status, self.reason = 'unknown', 'No supplied binding for this source and relation.'
                else:
                    # A real rank-one write; private state is read AFTER the write.
                    self.fast += np.outer(candidate - self.fast @ self.anchor, self.anchor) / float(self.anchor @ self.anchor)
                    self.h = self.fast @ self.anchor
            # Interventions miss one internal operation while keeping task timing fixed.
            self.cursor = min(self.cursor + 1, len(self.episode.program))
        self.workspace = copy.deepcopy(packet)
        self.trace.append(dict(cycle=self.cycles, kind=kind, packet=copy.deepcopy(packet),
                               before=before.tolist(), after=self.h.tolist(),
                               geometry_delta=float(np.linalg.norm(self.fast - old_fast)),
                               residue=len(self.episode.program) - self.cursor, status=self.status))
        return self.view()

    def run(self):
        while self.status == 'thinking':
            self.step()
        return self.view()

    def snapshot(self):
        return dict(version=1, episode=self.signature, model=self.model_id,
                    codebook=dict(bindings=self.bindings.tolist(), fast=self.fast.tolist()),
                    workspace=copy.deepcopy(self.workspace), residue=dict(cursor=self.cursor),
                    private=self.h.tolist(), cycles=self.cycles, ping_count=self.ping_count,
                    status=self.status, reason=self.reason)

    def interrupt(self):
        """Discard working state and facts. Keep only original public query metadata."""
        self.bindings = np.zeros_like(self.bindings)
        self.fast = np.eye(N_OBJECTS)
        self.h = self.anchor.copy()
        self.workspace = None
        self.cursor, self.cycles, self.ping_count = 0, 0, 0
        self.status, self.reason, self.trace = 'thinking', '', []

    def restore(self, snapshot, components=COMPONENTS):
        """Validate everything first, then restore selected state without aliasing."""
        try:
            selected = tuple(components)
            if not selected or set(selected) - set(COMPONENTS):
                raise ValueError('Unknown or empty restore component selection.')
            if (snapshot['version'] != 1 or snapshot['episode'] != self.signature
                    or snapshot['model'] != self.model_id):
                raise ValueError('Snapshot belongs to another episode or controller.')
            bindings = _array(snapshot['codebook']['bindings'], (4, N_OBJECTS, N_OBJECTS), True)
            if (bindings.sum(axis=1) > 1 + 1e-9).any():
                raise ValueError('Invalid codebook probability mass.')
            fast = _array(snapshot['codebook']['fast'], (N_OBJECTS, N_OBJECTS))
            private = _array(snapshot['private'], (N_OBJECTS,), True)
            if abs(private.sum() - 1) > 1e-8:
                raise ValueError('Private state must have unit mass.')
            cursor = snapshot['residue']['cursor']
            if type(cursor) is not int or not 0 <= cursor <= len(self.episode.program):
                raise ValueError('Invalid residue cursor.')
            workspace = copy.deepcopy(snapshot['workspace'])
            if workspace is not None:
                for field, size in (('source', N_OBJECTS), ('candidate', N_OBJECTS), ('probabilities', 5)):
                    a = _array(workspace[field], (size,), True)
                    if a.sum() > 1 + 1e-8:
                        raise ValueError('Invalid workspace probability mass.')
                if workspace['relation'] not in (*RELATIONS, 'HALT'):
                    raise ValueError('Invalid workspace relation.')
            cycles, pings = snapshot['cycles'], snapshot['ping_count']
            if (type(cycles) is not int or not 0 <= cycles <= 128 or type(pings) is not int
                    or not 0 <= pings <= cycles or snapshot['status'] not in STATUSES
                    or not isinstance(snapshot['reason'], str)):
                raise ValueError('Invalid snapshot counters/status.')
            if ((snapshot['status'] != 'thinking' and cycles == 0)
                    or (snapshot['status'] == 'complete' and cursor != len(self.episode.program))
                    or (snapshot['status'] == 'halted_early' and cursor >= len(self.episode.program))):
                raise ValueError('Snapshot status conflicts with unfinished residue or cycle counters.')
            restored_h = self.h.copy()
            if 'codebook' in selected:
                restored_h = fast @ self.anchor
            elif 'workspace' in selected and workspace is not None:
                candidate = np.array(workspace['candidate'])
                restored_h = candidate if abs(candidate.sum() - 1) < 1e-8 else np.array(workspace['source'])
            if 'private' in selected:
                restored_h = private
            if not np.isfinite(restored_h).all() or (restored_h < -1e-9).any() or abs(restored_h.sum() - 1) > 1e-8:
                raise ValueError('Selected state cannot reconstruct a valid private vector.')
        except (KeyError, TypeError, IndexError) as e:
            raise ValueError('Malformed snapshot.') from e
        if 'codebook' in selected:
            self.bindings, self.fast = bindings, fast
        if 'workspace' in selected:
            self.workspace = workspace
        if 'residue' in selected:
            self.cursor = cursor
        self.h = restored_h.copy()
        self.cycles, self.ping_count, self.trace = 0, 0, []
        self.status, self.reason = 'thinking', ''
        if set(selected) == set(COMPONENTS):
            self.cycles, self.ping_count = cycles, pings
            self.status, self.reason = snapshot['status'], snapshot['reason']
        return self.view()

    def view(self):
        probabilities = self.controller.decode(self.h)
        answer = self.episode.names[int(probabilities.argmax())] if self.status == 'complete' else None
        return dict(names=list(self.episode.names), query=self.episode.query_text(),
                    status=self.status, reason=self.reason, answer=answer,
                    public=f'I get {answer}.' if answer is not None else None,
                    public_confidence=float(probabilities.max()) if answer is not None else None,
                    private=self.h.tolist(), fast=self.fast.tolist(), workspace=copy.deepcopy(self.workspace),
                    next_packet=self.propose(), cursor=self.cursor,
                    remaining=[INSTRUCTIONS[i] for i in self.episode.program[self.cursor:]],
                    cycles=self.cycles, ping_count=self.ping_count,
                    trace=copy.deepcopy(self.trace), parameters=self.controller.parameter_count,
                    active_bindings=int(np.count_nonzero(self.bindings)))
