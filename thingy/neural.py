"""Small trainable permanent machinery; exact gradients, no autodiff dependency."""

import json
from pathlib import Path
import numpy as np

from .tasks import N_OBJECTS, INSTRUCTIONS, INSTRUCTION_RELATIONS

EMBED_SIZE = 8
HIDDEN_SIZE = 48
HALT = 4
EMBEDDINGS = np.random.default_rng(20261006).normal(size=(len(INSTRUCTIONS), EMBED_SIZE))


def softmax(x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Controller produced nonfinite logits; check its weights and inputs.')
    with np.errstate(under='ignore'):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def features(tokens, h, remaining):
    tokens = np.asarray(tokens)
    embedding = EMBEDDINGS[np.maximum(tokens, 0)].copy()
    embedding[tokens < 0] = 0
    # Keep the learned feature in the 0..4 range seen during training. The host
    # residue still retains the entire query and exact remaining count.
    return np.concatenate((embedding, h, np.clip(np.asarray(remaining), 0, 4)[:, None] / 8,
                           (1 - (h * h).sum(axis=1))[:, None]), axis=1)


class Controller:
    """Permanent weights are frozen during an Agent episode."""
    input_size = EMBED_SIZE + N_OBJECTS + 2

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.params = dict(
            W1=rng.normal(0, 0.15, (self.input_size, HIDDEN_SIZE)),
            b1=np.zeros(HIDDEN_SIZE),
            W2=rng.normal(0, 0.12, (HIDDEN_SIZE, 5)),
            b2=np.zeros(5),
            Wo=rng.normal(0, 0.04, (N_OBJECTS, N_OBJECTS)),
            bo=np.zeros(N_OBJECTS))

    def predict(self, x):
        try:
            with np.errstate(over='raise', invalid='raise'):
                a = np.tanh(x @ self.params['W1'] + self.params['b1'])
                return softmax(a @ self.params['W2'] + self.params['b2'])
        except FloatingPointError as e:
            raise ValueError('Controller arithmetic overflow; check its weights and inputs.') from e

    def decode(self, h):
        try:
            with np.errstate(over='raise', invalid='raise'):
                return softmax(np.asarray(h) @ self.params['Wo'] + self.params['bo'])
        except FloatingPointError as e:
            raise ValueError('Public decoder arithmetic overflow; check its weights and state.') from e

    def loss_and_grad(self, batch, teacher=0.0, auxiliary=0.0, ping_cost=0.002):
        """Differentiate soft recurrent packet feedback and a separate public head.

        Teacher mixing and auxiliary packet supervision decay to zero. The host
        supplies the remaining query; it never supplies the answer to the runtime.
        """
        if not 0 <= teacher <= 1 or auxiliary < 0 or ping_cost < 0:
            raise ValueError('Invalid teacher, auxiliary, or cost.')
        b, tokens, sources, targets = (batch[k] for k in ('bindings', 'tokens', 'sources', 'targets'))
        count, cycles = tokens.shape
        h = np.eye(N_OBJECTS)[sources].copy()
        cache, auxiliary_loss, ping_loss = [], 0.0, 0.0
        lengths = (tokens >= 0).sum(axis=1)
        for t in range(cycles):
            x = features(tokens[:, t], h, np.maximum(lengths - t, 0))
            a = np.tanh(x @ self.params['W1'] + self.params['b1'])
            p = softmax(a @ self.params['W2'] + self.params['b2'])
            labels = np.where(tokens[:, t] < 0, HALT,
                              INSTRUCTION_RELATIONS[np.maximum(tokens[:, t], 0)])
            teach = np.eye(5)[labels]
            mixed = teacher * teach + (1 - teacher) * p
            candidates = np.concatenate((np.einsum('brij,bj->bri', b, h), h[:, None, :]), axis=1)
            cache.append((h, x, a, p, mixed, candidates, labels))
            h = np.einsum('br,bri->bi', mixed, candidates)
            auxiliary_loss -= auxiliary * np.log(p[np.arange(count), labels] + 1e-30).mean() / cycles
            ping_loss += ping_cost * (1 - p[:, HALT]).mean() / cycles
        public = self.decode(h)
        loss = -np.log(public[np.arange(count), targets] + 1e-30).mean() + auxiliary_loss + ping_loss
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dpublic = (public - np.eye(N_OBJECTS)[targets]) / count
        grads['Wo'] = h.T @ dpublic
        grads['bo'] = dpublic.sum(axis=0)
        dh = dpublic @ self.params['Wo'].T
        for old_h, x, a, p, mixed, candidates, labels in reversed(cache):
            dp = (1 - teacher) * np.einsum('bi,bri->br', dh, candidates)
            dp[np.arange(count), labels] -= auxiliary / (count * cycles) / (p[np.arange(count), labels] + 1e-30)
            dp[:, HALT] -= ping_cost / (count * cycles)
            dz = p * (dp - (dp * p).sum(axis=1, keepdims=True))
            grads['W2'] += a.T @ dz
            grads['b2'] += dz.sum(axis=0)
            da = (dz @ self.params['W2'].T) * (1 - a * a)
            grads['W1'] += x.T @ da
            grads['b1'] += da.sum(axis=0)
            dx = da @ self.params['W1'].T
            dh = (np.einsum('br,brij,bi->bj', mixed[:, :4], b, dh)
                  + mixed[:, HALT, None] * dh
                  + dx[:, EMBED_SIZE:EMBED_SIZE + N_OBJECTS]
                  - 2 * old_h * dx[:, -1, None])
        metrics = dict(accuracy=float((public.argmax(axis=1) == targets).mean()),
                       answer_loss=float(loss - auxiliary_loss - ping_loss),
                       auxiliary_loss=float(auxiliary_loss), ping_loss=float(ping_loss))
        return float(loss), grads, metrics

    def to_dict(self):
        return dict(version=1, kind='thingy-numpy-controller', instructions=list(INSTRUCTIONS),
                    params={k: v.tolist() for k, v in self.params.items()})

    @classmethod
    def from_dict(cls, data):
        try:
            if (data['version'] != 1 or data['kind'] != 'thingy-numpy-controller'
                    or data['instructions'] != list(INSTRUCTIONS)):
                raise ValueError('Incompatible controller checkpoint.')
            model = cls()
            for k, expected in model.params.items():
                array = np.asarray(data['params'][k], dtype=float)
                if (array.shape != expected.shape or not np.isfinite(array).all()
                        or (np.abs(array) > 1e6).any()):
                    raise ValueError(f'Invalid checkpoint tensor {k}.')
                model.params[k] = array.copy()
            return model
        except (KeyError, TypeError, OverflowError) as e:
            raise ValueError('Invalid controller checkpoint.') from e

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), separators=(',', ':'), allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def parameter_count(self):
        return sum(p.size for p in self.params.values())
