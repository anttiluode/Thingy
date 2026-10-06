"""`python -m thingy`: watch, train, measure, or open the local explorer."""

import argparse
import json
from pathlib import Path
import sys
import threading
import webbrowser
import numpy as np

from .engine import Agent, load_default_controller
from .neural import Controller
from .tasks import generate_episode, parse_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description='Thingy — learned causal self-pings, inspectable fast memory.')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Print genuine packets and public output.')
    demo.add_argument('--query', default='Zapp means helps')
    demo.add_argument('--facts', type=Path, help='UTF-8 file of source relation target facts.')
    demo.add_argument('--random', action='store_true')
    demo.add_argument('--seed', type=int, default=7)
    demo.add_argument('--hops', type=int, default=4)
    demo.add_argument('--intervention', choices=['drop', 'wrong', 'freeze'])
    demo.add_argument('--model', type=Path)
    train_parser = commands.add_parser('train', help='Train a new small controller on fresh episodes.')
    train_parser.add_argument('--seed', type=int, default=7)
    train_parser.add_argument('--steps', type=int, default=1000)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--out', type=Path, default=Path('runs/training'))
    bench = commands.add_parser('benchmark', help='Run frozen, paired causal and memory experiments.')
    bench.add_argument('--seed', type=int, default=101)
    bench.add_argument('--episodes', type=int, default=420)
    bench.add_argument('--model', type=Path)
    bench.add_argument('--out', type=Path, default=Path('runs/benchmark'))
    serve = commands.add_parser('serve', help='Start the local live workspace explorer.')
    serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--open', action='store_true', help='Open your browser automatically.')
    serve.add_argument('--model', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            from .training import train
            def progress(row):
                print(f"step {row['step']:4d}  answer accuracy {row['accuracy']:.1%}  "
                      f"teacher {row['teacher']:.2f}  loss {row['loss']:.4f}", flush=True)
            model, receipt = train(args.seed, args.steps, args.batch_size, callback=progress)
            args.out.mkdir(parents=True, exist_ok=True)
            model.save(args.out / 'controller.json')
            (args.out / 'training.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            print(f"Saved {args.out / 'controller.json'}; fresh soft validation {receipt['soft_validation']['accuracy']:.1%}.")
            return 0
        model = Controller.load(args.model) if args.model else load_default_controller()
        if args.command == 'benchmark':
            from .benchmark import benchmark, write_report
            report = benchmark(model, args.seed, args.episodes)
            write_report(report, args.out)
            print(f"Held-out: {report['held_out']['self_ping']['accuracy']:.1%}; "
                  f"mechanism gate: {'PASS' if report['gate']['passed'] else 'FAIL'}; {args.out / 'REPORT.md'}")
            return 0
        if args.command == 'serve':
            from .server import create_server
            server = create_server(port=args.port, controller=model)
            url = f'http://127.0.0.1:{server.server_port}'
            print(f'Thingy: {url}  ({model.parameter_count:,} learned parameters; CPU, no API key)', flush=True)
            if args.open:
                threading.Timer(0.4, lambda: webbrowser.open(url)).start()
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0
        from .server import DEFAULT_FACTS
        episode = (generate_episode(np.random.default_rng(args.seed), args.hops) if args.random else
                   parse_episode(args.facts.read_text(encoding='utf-8') if args.facts else DEFAULT_FACTS, args.query))
        agent = Agent(model, episode)
        print(f'Thingy / {model.parameter_count:,} learned parameters\nQuery: {episode.query_text()}')
        first = True
        while agent.status == 'thinking':
            intervention = None
            if first and args.intervention:
                if args.intervention == 'wrong':
                    from .tasks import RELATIONS
                    proposal = agent.propose()['relation']
                    relation = RELATIONS[(RELATIONS.index(proposal) + 1) % 4] if proposal in RELATIONS else 'means'
                    intervention = dict(kind='replace', relation=relation)
                else:
                    intervention = dict(kind=args.intervention)
            before = len(agent.trace)
            agent.step(intervention)
            if len(agent.trace) > before:
                event = agent.trace[-1]
                packet = event['packet']
                source = episode.names[int(np.argmax(packet['source']))]
                current = episode.names[int(agent.h.argmax())]
                print(f"  self-ping {event['cycle']}: {source} / {packet['relation']} -> {current}  "
                      f"geometry Δ {event['geometry_delta']:.3f}; residue {event['residue']} [{event['kind']}]")
            first = False
        view = agent.view()
        print('Public:', view['public'] or f"[{view['status']}] {view['reason']}")
        return 0
    except (ValueError, OSError) as e:
        print(f'Thingy: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
