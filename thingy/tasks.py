"""Fresh episodic worlds and a small, explicit language/address interface."""

from dataclasses import dataclass
import re
import numpy as np

N_OBJECTS = 16
MAX_HOPS = 32
RELATIONS = ('means', 'next', 'helps', 'opposes')
INSTRUCTIONS = ('means', 'defines', 'is', 'next', 'after', 'follows',
                'helps', 'supports', 'aids', 'opposes', 'blocks', 'hinders')
INSTRUCTION_RELATIONS = np.repeat(np.arange(4), 3)
DEFAULT_NAMES = ('Zapp', 'tired', 'rest', 'cook', 'soup', 'forest', 'bike',
                 'safe', 'danger', 'joke', 'app', 'water', 'home', 'light',
                 'sound', 'quiet')


@dataclass(frozen=True)
class Episode:
    names: tuple
    bindings: np.ndarray
    source: int
    program: tuple

    def __post_init__(self):
        names = tuple(self.names)
        if (len(names) != N_OBJECTS or len(set(names)) != N_OBJECTS
                or any(not isinstance(n, str) or not re.fullmatch(r'[\w-]{1,40}', n) for n in names)):
            raise ValueError('Use 16 distinct object names (letters, numbers, underscore or dash).')
        b = np.array(self.bindings, dtype=float, copy=True)
        if (b.shape != (4, N_OBJECTS, N_OBJECTS) or not np.isfinite(b).all()
                or (b < 0).any() or (b.sum(axis=1) > 1 + 1e-9).any()):
            raise ValueError('Bindings must be finite nonnegative 4 x 16 x 16 matrices with column mass <= 1.')
        if (type(self.source) is not int or not 0 <= self.source < N_OBJECTS
                or len(self.program) > MAX_HOPS
                or any(type(i) is not int or not 0 <= i < len(INSTRUCTIONS) for i in self.program)):
            raise ValueError('Invalid query address or query length (maximum 32 relations).')
        b.setflags(write=False)
        object.__setattr__(self, 'names', names)
        object.__setattr__(self, 'bindings', b)
        object.__setattr__(self, 'program', tuple(self.program))

    def target(self):
        """Oracle for training/scoring only. The engine never calls this."""
        current = self.source
        for token in self.program:
            column = self.bindings[INSTRUCTION_RELATIONS[token], :, current]
            if column.sum() < 1 - 1e-9:
                return None
            current = int(np.argmax(column))
        return current

    def to_dict(self):
        return dict(names=list(self.names), bindings=self.bindings.tolist(),
                    source=self.source, program=list(self.program))

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(tuple(data['names']), data['bindings'], data['source'], tuple(data['program']))
        except (KeyError, TypeError) as e:
            raise ValueError('Invalid episode record.') from e

    def query_text(self):
        return ' '.join([self.names[self.source]] + [INSTRUCTIONS[i] for i in self.program])

    def facts_text(self):
        lines = []
        for r, name in enumerate(RELATIONS):
            for source in range(N_OBJECTS):
                column = self.bindings[r, :, source]
                if column.sum() > 0.999:
                    lines.append(f'{self.names[source]} {name} {self.names[int(column.argmax())]}')
        return '\n'.join(lines)


def generate_episode(rng, hops=3):
    if type(hops) is not int or not 0 <= hops <= MAX_HOPS:
        raise ValueError('hops must be an integer from 0 to 32.')
    b = np.zeros((4, N_OBJECTS, N_OBJECTS))
    for r in range(4):
        b[r, rng.permutation(N_OBJECTS), np.arange(N_OBJECTS)] = 1
    return Episode(DEFAULT_NAMES, b, int(rng.integers(N_OBJECTS)),
                   tuple(int(x) for x in rng.integers(len(INSTRUCTIONS), size=hops)))


def parse_episode(facts, query):
    """Facts: `Zapp means tired`; query: `Zapp means helps`."""
    if not isinstance(facts, str) or not isinstance(query, str) or len(facts) > 32000 or len(query) > 2000:
        raise ValueError('Facts/query must be bounded text.')
    triples, names, seen = [], [], {}
    for line in facts.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.strip().split()
        if len(fields) != 3 or fields[1] not in RELATIONS:
            raise ValueError('Each fact is: source means|next|helps|opposes target.')
        source, relation, target = fields
        key = (source, relation)
        if key in seen and seen[key] != target:
            raise ValueError(f'Conflicting definitions for {source} {relation}.')
        seen[key] = target
        triples.append((source, RELATIONS.index(relation), target))
        for name in (source, target):
            if name not in names:
                names.append(name)
    if not names or len(names) > N_OBJECTS:
        raise ValueError('Supply facts containing 1 to 16 objects.')
    tokens = query.strip().split()
    if not tokens or tokens[0] not in names:
        raise ValueError('Query must start with an object defined in the facts.')
    if any(t not in INSTRUCTIONS for t in tokens[1:]) or len(tokens) - 1 > MAX_HOPS:
        raise ValueError('Use known relation words, at most 32 hops.')
    source = names.index(tokens[0])
    for i in range(N_OBJECTS):
        if len(names) == N_OBJECTS:
            break
        name = f'unused_{i}'
        if name not in names:
            names.append(name)
    b = np.zeros((4, N_OBJECTS, N_OBJECTS))
    for start, relation, end in triples:
        b[relation, names.index(end), names.index(start)] = 1
    return Episode(tuple(names), b, source, tuple(INSTRUCTIONS.index(t) for t in tokens[1:]))
