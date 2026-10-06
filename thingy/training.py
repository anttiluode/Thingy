"""Developmental curriculum: guided packet feedback, then autonomous task loss."""

import time
import numpy as np

from .neural import Controller
from .tasks import generate_episode, N_OBJECTS


def make_batch(rng, batch_size=64, max_hops=4):
    if batch_size < 1 or not 0 <= max_hops <= 32:
        raise ValueError('Invalid batch size or hop count.')
    episodes = [generate_episode(rng, int(rng.integers(max_hops + 1))) for _ in range(batch_size)]
    tokens = np.full((batch_size, max_hops + 1), -1, dtype=int)
    for i, ep in enumerate(episodes):
        tokens[i, :len(ep.program)] = ep.program
    return dict(bindings=np.stack([e.bindings for e in episodes]), tokens=tokens,
                sources=np.array([e.source for e in episodes]),
                targets=np.array([e.target() for e in episodes]))


def train(seed=7, steps=1000, batch_size=64, max_hops=4, learning_rate=0.02, callback=None):
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('Positive steps, batch size, and learning rate are required.')
    model = Controller(seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    m = {k: np.zeros_like(v) for k, v in model.params.items()}
    v = {k: np.zeros_like(p) for k, p in model.params.items()}
    phases, started = [], time.monotonic()
    for step in range(steps):
        guided = max(0.0, 1 - step / max(1, int(steps * 0.45)))
        teacher, auxiliary = guided * 0.8, guided
        batch = make_batch(rng, batch_size, max_hops)
        loss, grads, metrics = model.loss_and_grad(batch, teacher, auxiliary)
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        for k in model.params:
            g = grads[k] * min(1.0, 5.0 / max(norm, 1e-12))
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.999 * v[k] + 0.001 * g * g
            model.params[k] -= learning_rate * (m[k] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[k] / (1 - 0.999 ** (step + 1))) + 1e-8)
        if step == 0 or (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
            row = dict(step=step + 1, teacher=float(teacher), auxiliary=float(auxiliary), loss=loss, **metrics)
            phases.append(row)
            if callback:
                callback(row)
    validation = make_batch(np.random.default_rng(np.random.SeedSequence([seed, 2])), 512, max_hops)
    _, _, metrics = model.loss_and_grad(validation)
    return model, dict(seed=seed, steps=steps, batch_size=batch_size, max_hops=max_hops,
                       learning_rate=learning_rate, parameters=model.parameter_count,
                       training_rng='SeedSequence([seed, 1])', validation_rng='SeedSequence([seed, 2])',
                       phases=phases, soft_validation=metrics, seconds=time.monotonic() - started,
                       learning='Exact BPTT through soft packet feedback; Adam. Guided supervision fades to zero.')
