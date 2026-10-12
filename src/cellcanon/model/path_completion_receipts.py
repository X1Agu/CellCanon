from __future__ import annotations
import itertools

def completion_receipt(alternatives, sources, *, maximum_unknown_edges=10):
    unknown = sorted({i for p in alternatives for i in p['refs'] if sources[i].get('action_sign') not in [-1, 1]})
    known = sorted({p['net_sign'] for p in alternatives if p['net_sign'] in [-1, 1]})
    if len(unknown) > maximum_unknown_edges:
        return dict(exhaustive=False, known_net_signs=known, unknown_source_indices=unknown, reason='Too many unknown source signs for bounded enumeration; no certainty inferred.')
    outcomes = set()
    for signs in itertools.product([-1, 1], repeat=len(unknown)):
        assigned = dict(zip(unknown, signs))
        nets = []
        for p in alternatives:
            sign = 1
            for i in p['refs']:
                sign *= sources[i]['action_sign'] if i not in assigned else assigned[i]
            nets.append(sign)
        outcomes.add(tuple(sorted(set(nets))))
    return dict(exhaustive=True, known_net_signs=known, unknown_source_indices=unknown, hypothetical_completion_net_sign_sets=[list(v) for v in sorted(outcomes)], possible_assignments=2 ** len(unknown), new_experimental_evidence=False, scope='Hypothetical sign completions of this already sourced background chain; not measured mechanisms.')
