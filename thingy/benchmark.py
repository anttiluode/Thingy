"""Frozen, paired causal experiments. These are controller ablations, not LLMs."""

import json
import math
from pathlib import Path
import numpy as np

from .engine import Agent, COMPONENTS
from .neural import HALT, features
from .tasks import generate_episode, Episode, RELATIONS, N_OBJECTS


def text_feedback(model, episode, one_read=False):
    """Serialize an actual selected address, parse it, and feed it back."""
    h = np.eye(N_OBJECTS)[episode.source]
    transcript, reads, status = [], 0, 'unfinished'
    for cursor in range(len(episode.program) + 3):
        token = episode.program[cursor] if cursor < len(episode.program) else -1
        p = model.predict(features(np.array([token]), h[None], np.array([max(0, len(episode.program) - cursor)])))[0]
        action = int(p.argmax())
        if action == HALT:
            transcript.append('HALT')
            status = 'complete' if cursor >= len(episode.program) else 'early_halt'
            break
        message = f'READ {int(h.argmax())} {RELATIONS[action]}'
        transcript.append(message)
        verb, source, relation = message.split()
        assert verb == 'READ'
        key = np.eye(N_OBJECTS)[int(source)]
        h = episode.bindings[RELATIONS.index(relation)] @ key
        reads += 1
        if h.sum() < 1 - 1e-8:
            status = 'unknown'
            break
        if one_read:
            status = 'complete'
            break
    return dict(prediction=int(model.decode(h).argmax()) if status == 'complete' else None,
                bytes=len('\n'.join(transcript).encode('utf-8')), reads=reads, status=status)


def latent_feedback(model, episode):
    """Continuous relation mixture and private recurrence, with no sparse packet."""
    h = np.eye(N_OBJECTS)[episode.source]
    status, reads = 'unfinished', 0
    for cursor in range(len(episode.program) + 3):
        token = episode.program[cursor] if cursor < len(episode.program) else -1
        p = model.predict(features(np.array([token]), h[None], np.array([max(0, len(episode.program) - cursor)])))[0]
        if int(p.argmax()) == HALT:
            status = 'complete' if cursor >= len(episode.program) else 'early_halt'
            break
        h = np.einsum('r,rij,j->i', p[:4], episode.bindings, h) + p[HALT] * h
        reads += 1
        if h.sum() < 0.9:
            status = 'unknown'
            break
    return dict(prediction=int(model.decode(h).argmax()) if status == 'complete' else None,
                reads=reads, status=status)


def _prediction(agent):
    return int(agent.controller.decode(agent.h).argmax()) if agent.status == 'complete' else None


def _summary(values):
    total, correct = len(values), sum(values)
    accuracy = correct / total if total else 0.0
    if total:
        z = 1.96
        denom = 1 + z * z / total
        center = (accuracy + z * z / (2 * total)) / denom
        margin = z * math.sqrt(accuracy * (1 - accuracy) / total + z * z / (4 * total * total)) / denom
        interval = [max(0.0, center - margin), min(1.0, center + margin)]
    else:
        interval = [0.0, 1.0]
    return dict(correct=correct, total=total, accuracy=accuracy, wilson95=interval)


def mechanism_gate(held_accuracy, intact_accuracy, delete_accuracy, wrong_accuracy, replay_accuracy):
    checks = dict(held_out_at_least_90pct=held_accuracy >= 0.9,
                  delete_drop_at_least_20pp=intact_accuracy - delete_accuracy >= 0.2,
                  wrong_drop_at_least_20pp=intact_accuracy - wrong_accuracy >= 0.2,
                  correct_replay_at_least_95pct=replay_accuracy >= 0.95)
    return dict(passed=all(checks.values()), checks=checks,
                thresholds=dict(held_out=0.9, causal_drop=0.2, replay=0.95))


def benchmark(model, seed=101, episodes=420):
    if type(episodes) is not int or not 7 <= episodes <= 100000:
        raise ValueError('episodes must be an integer from 7 to 100000.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
    alternate = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    distractor_rng = np.random.default_rng(np.random.SeedSequence([seed, 102]))
    held = {k: [] for k in ('one_read', 'text_feedback', 'latent_recurrence', 'self_ping')}
    by_hops = {str(i): [] for i in range(7)}
    causal = {k: [] for k in ('intact', 'delete', 'wrong', 'freeze', 'replay', 'display_only')}
    restore_sets = {'codebook': ('codebook',), 'workspace': ('workspace',), 'residue': ('residue',),
                    'codebook+residue': ('codebook', 'residue'),
                    'workspace+residue': ('workspace', 'residue'), 'all': COMPONENTS}
    memory = {k: [] for k in restore_sets}
    memory_prefix_correct = {k: [] for k in restore_sets}
    replay_eligible, replay_affected, same_present = [], [], []
    geometry, pings, zero_pings, text_bytes, packet_bytes = [], [], [], [], []
    rows = []
    for i in range(episodes):
        hops = i % 7  # Each length receives a declared, known denominator.
        episode = generate_episode(rng, hops)
        target = episode.target()
        normal = Agent(model, episode)
        initial = normal.snapshot()
        proposed = normal.propose()
        normal.run()
        intact = _prediction(normal) == target
        held['self_ping'].append(intact)
        by_hops[str(hops)].append(intact)
        pings.append(normal.ping_count)
        if hops == 0:
            zero_pings.append(normal.ping_count == 0)
        for label, result in (('one_read', text_feedback(model, episode, True)),
                              ('text_feedback', text_feedback(model, episode)),
                              ('latent_recurrence', latent_feedback(model, episode))):
            held[label].append(result['prediction'] == target)
            if label == 'text_feedback':
                text_bytes.append(result['bytes'])
        packet_bytes.append(sum(len(json.dumps(event['packet'], separators=(',', ':')).encode())
                                for event in normal.trace))
        changed = generate_episode(alternate, hops)
        other = Episode(episode.names, changed.bindings, episode.source, episode.program)
        other_target = other.target()
        if other_target != target:
            other_agent = Agent(model, other)
            other_agent.run()
            same_present.append(intact and _prediction(other_agent) == other_target)
        row = dict(case=i, hops=hops, target=target, prediction=_prediction(normal))
        if hops >= 2:
            causal['intact'].append(intact)
            variants = {}
            first_relation = RELATIONS.index(proposed['relation']) if proposed['relation'] in RELATIONS else 0
            for label, intervention in (('delete', {'kind': 'drop'}),
                                        ('wrong', {'kind': 'replace', 'relation': RELATIONS[(first_relation + 1) % 4]}),
                                        ('freeze', {'kind': 'freeze'})):
                trial = Agent(model, episode)
                trial.restore(initial)
                trial.step(intervention)
                if label == 'wrong':
                    wrong_fast = trial.fast.copy()
                trial.run()
                variants[label] = _prediction(trial)
                causal[label].append(variants[label] == target)
            # Reset to the identical PRE-intervention state; no damaged-branch carryover.
            replay = Agent(model, episode)
            replay.restore(initial)
            replay.step()
            geometry.append(float(np.linalg.norm(wrong_fast - replay.fast)))
            replay.run()
            replay_ok = _prediction(replay) == target
            causal['replay'].append(replay_ok)
            if intact:
                replay_eligible.append(replay_ok)
                if variants['wrong'] != target:
                    replay_affected.append(replay_ok)
            # Editing a diagnostic copy is the noncausal placebo.
            display = normal.view()
            display['trace'][0]['packet']['relation'] = RELATIONS[(first_relation + 1) % 4]
            causal['display_only'].append(_prediction(normal) == target)
            prefix = Agent(model, episode)
            prefix_length = max(1, hops // 2)
            for _ in range(prefix_length):
                prefix.step()
            saved = prefix.snapshot()
            prefix_episode = Episode(episode.names, episode.bindings, episode.source, episode.program[:prefix_length])
            prefix_ok = int(prefix.h.argmax()) == prefix_episode.target() and prefix.status == 'thinking'
            # A real unrelated task runs between save and restoration, with fresh state.
            Agent(model, generate_episode(distractor_rng, 3)).run()
            for label, components in restore_sets.items():
                trial = Agent(model, episode)
                trial.interrupt()  # Zero the facts too. Do not replay the original transcript.
                trial.restore(saved, components)
                trial.run()
                correct = _prediction(trial) == target
                memory[label].append(correct)
                if prefix_ok:
                    memory_prefix_correct[label].append(correct)
            row['intervention_predictions'] = variants
        rows.append(row)
    held_summary = {k: _summary(v) for k, v in held.items()}
    causal_summary = {k: _summary(v) for k, v in causal.items()}
    codebook_scalars, workspace_scalars = 4 * 16 * 16 + 16 * 16, 16 + 16 + 5
    sizes = {'codebook': codebook_scalars, 'workspace': workspace_scalars, 'residue': 1, 'private': 16}
    memory_summary = {}
    for label, values in memory.items():
        memory_summary[label] = {**_summary(values), 'correct_prefix_subset': _summary(memory_prefix_correct[label]),
                                 'retained_numeric_scalars': sum(sizes[c] for c in restore_sets[label])}
    return dict(version=1, seed=seed, episodes=episodes, parameters=model.parameter_count,
                model_sha256=normal.model_id,
                evaluation_rng='SeedSequence([seed, 100]); alternate=101; distractor=102',
                training_hops='0..4', evaluation_hops='0..6; uniform cyclic allocation',
                held_out=held_summary, by_hops={k: _summary(v) for k, v in by_hops.items()},
                causal=causal_summary, replay_eligible=_summary(replay_eligible), replay_affected=_summary(replay_affected),
                same_present_different_history=_summary(same_present), memory=memory_summary,
                silence=_summary(zero_pings), mean_pings=float(np.mean(pings)),
                mean_wrong_geometry_delta=float(np.mean(geometry)),
                payload_bytes=dict(text_mean=float(np.mean(text_bytes)),
                                   instrumented_packet_json_mean=float(np.mean(packet_bytes))),
                gate=mechanism_gate(held_summary['self_ping']['accuracy'], causal_summary['intact']['accuracy'],
                                    causal_summary['delete']['accuracy'], causal_summary['wrong']['accuracy'],
                                    _summary(replay_eligible)['accuracy']),
                limitations=[
                    'Small shared-controller ablations, not trained transformer/CoT baselines.',
                    'The parser, address basis, remaining-query cursor, fact matrices, and delta write are engineered.',
                    'Codebook includes all facts plus fast operator; C+r reconstructs h by design, not learned compression.',
                    'Soft recurrent training differs from hard sparse packet evaluation.',
                    'Payload compares full instrumented JSON with short addresses; no compression advantage is claimed.',
                    'No evidence of consciousness, spontaneous private language, or a replicated Jacobian lens.'],
                cases=rows)


def write_report(report, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    pct = lambda v: f"{v['accuracy']:.1%} ({v['correct']}/{v['total']})"
    lines = ['# Thingy measured results', '',
             f"Frozen model, evaluation seed {report['seed']}, {report['episodes']} fresh fact graphs. "
             f"{report['parameters']:,} learned parameters. Training chain lengths 0–4; evaluation 0–6.", '',
             'These are shared **controller ablations**, not four language transformers.', '',
             '| Arm | Exact accuracy |', '|---|---:|']
    lines += [f'| {k} | {pct(v)} |' for k, v in report['held_out'].items()]
    lines += ['', '## Paired interventions on multi-hop episodes', '', '| Intervention | Exact accuracy |', '|---|---:|']
    lines += [f'| {k} | {pct(v)} |' for k, v in report['causal'].items()]
    lines += ['', f"Correct replay on intact-correct cases: {pct(report['replay_eligible'])}. "
              f"Wrong-ping-affected subset: {pct(report['replay_affected'])}.", '',
              f"Different codebooks under the identical query: {pct(report['same_present_different_history'])} paired success.", '',
              '## Interruption and selective restoration', '',
              'The saved prefix is followed by an unrelated task. Facts and working state are cleared before each restore. '
              'Original public query metadata is retained in every arm. C includes all episodic facts and the fast operator. '
              '**C+r recovery is designed into this state split**, not a learned compression result. '
              'Numeric state counts exclude fixed theta, public query metadata, and diagnostic trace text.', '',
              '| Restored state | Accuracy | Retained numeric scalars |', '|---|---:|---:|']
    lines += [f"| {k} | {pct(v)} | {v['retained_numeric_scalars']} |" for k, v in report['memory'].items()]
    lines += ['', f"Zero-hop silent halt: {pct(report['silence'])}. Mean pings per episode: {report['mean_pings']:.2f}.", '',
              f"**Predeclared mechanism gate: {'PASS' if report['gate']['passed'] else 'FAIL'}.**", '']
    lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in report['gate']['checks'].items()]
    lines += ['', '## What this establishes', '',
              'A genuinely trained tiny controller uses a readable address channel that causally changes a fast operator '
              'and the next private state. Deleting, replacing, freezing, and replaying that channel produce measurable effects. '
              'Text and latent recurrence can also solve this deliberately simple world; these results do not establish '
              'an accuracy advantage over either.', '', '## Limits', '']
    lines += [f'- {v}' for v in report['limitations']]
    lines += ['', 'Reproduce with `python -m thingy benchmark --seed ' + str(report['seed'])
              + ' --episodes ' + str(report['episodes']) + ' --out results/reproduced`.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
