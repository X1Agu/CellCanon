from __future__ import annotations
from pathlib import Path
FUNCTIONAL_SOURCE_ROOT = Path('<FUNCTIONAL_SOURCE_ROOT>')
TIERS = ['direct_target', 'directed', 'shared_pathway', 'shared_pathway_max200', 'citable_connection', 'signed_network', 'signed_transcript']

def validate_witness(w, edge_by_id, seed_by_evidence, query_symbols):
    s = seed_by_evidence[w['drug'], w['seed_evidence_id']]
    assert w['nodes'][0] == s['node'] and w['seed_source_url'] == s['source_url']
    assert w['seed_tier'] == s['tier'] and w['identity_tier'] == s['identity_tier']
    edges = w['network_edge_ids']
    nodes = w['nodes']
    assert len(edges) <= 3 and len(nodes) == len(set(nodes))
    if w['tier'].startswith('shared_pathway'):
        assert not edges and len(nodes) == 3
        assert nodes[-1] == query_symbols.get(w['gene'], w['gene'])
        assert len(w['pathway_references']) == 2
        assert all((r['pathway_id'] == nodes[1] and r['source_url'] for r in w['pathway_references']))
        assert w['net_sign'] is None
        return
    assert len(nodes) == len(edges) + 1
    assert nodes[-1] == 'GENE:' + query_symbols[w['gene']]
    for a, b, eid in zip(nodes, nodes[1:], edges):
        e = edge_by_id[eid]
        assert e['source_node_id'] == a and e['target_node_id'] == b and (e['reference_count'] > 0)
    if w['tier'].startswith('signed_'):
        assert edges and s['action_sign'] in {-1, 1}
        sign = s['action_sign']
        for eid in edges:
            assert edge_by_id[eid]['sign'] in {-1, 1}
            sign *= int(edge_by_id[eid]['sign'])
        assert w['net_sign'] == sign
        if w['tier'] == 'signed_transcript':
            assert edge_by_id[edges[-1]]['interaction_type'] == 'transcriptional'
    else:
        assert w['net_sign'] is None
