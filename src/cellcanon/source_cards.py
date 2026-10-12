import json
from .model.data import CELLS
from .model.condition_pilot_knowledge import _finite
from .model.condition_llm_packets import write

def build_cards(store, rows, protocol, root):
    split_map = {d: s for (s, k) in [('support', 'train_drugs'), ('validation', 'validation_drugs'), ('test', 'test_drugs')] for d in protocol[k]}
    rows = {(d, c): rows[d + '_' + c] for d in split_map for c in CELLS}
    cards = {}
    all_targets = {}
    for drug in sorted(split_map):
        original = rows[drug, 'A549']
        seeds = [json.loads(x[0]) for x in store.conn.execute('SELECT payload FROM evidence WHERE drug=? ORDER BY evidence_id', (drug,))]
        seeds = [s for s in seeds if str(s.get('object', '')).startswith('GENE:') and s.get('predicate') in {'functional_target_action', 'measured_target_binding', 'curated_target_relation_unknown_action'}]
        grouped = {}
        for s in seeds:
            grouped.setdefault(s['object'], []).append(s)
        target_order = sorted(grouped, key=lambda t: (not any((s.get('action_sign') is not None for s in grouped[t])), not any((s.get('source_record', {}).get('supports_active_1uM') for s in grouped[t])), t))
        (targets, evidence) = ([], {})
        for target in target_order[:10]:
            records = grouped[target]
            binding = [s for s in records if s['predicate'] == 'measured_target_binding']
            action = [s for s in records if s['predicate'] != 'measured_target_binding']
            binding.sort(key=lambda s: (not bool(s['source_record'].get('single_protein_assay_format')), not bool(s['source_record'].get('supports_active_1uM')), s['source_record'].get('value_nM') or 1e+20, s['evidence_id']))
            selected = action[:1] + binding[:2]
            (refs, measurements) = ([], [])
            for s in selected:
                eid = 'E' + str(len(evidence) + 1)
                refs.append(eid)
                raw = s.get('source_record', {})
                text = raw.get('assay_description') or s.get('text') or target + ' ' + s['predicate'] + ' with declared action sign ' + str(s.get('action_sign'))
                measure = {k: _finite(raw.get(k)) for k in ('measurement_type', 'relation', 'value_nM', 'potency_lower_p', 'potency_upper_p', 'single_protein_assay_format', 'assay_cell_type', 'assay_tissue', 'target_assignment_confidence', 'supports_active_1uM')}
                evidence[eid] = {'source_id': s['evidence_id'], 'source_url': s.get('source_url'), 'target': target, 'predicate': s['predicate'], 'action_sign': s.get('action_sign'), 'text': str(text)[:380], 'measurement': measure, 'identity_tier': s.get('identity_tier')}
                if s['predicate'] == 'measured_target_binding':
                    measurements.append(measure)
            symbol = target.removeprefix('GENE:')
            docs = store.background.conn.execute("SELECT d.evidence_id,d.text FROM documents d JOIN document_symbols s ON d.evidence_id=s.evidence_id WHERE s.symbol=? AND d.kind IN ('reviewed_human_function','ncbi_human_gene_summary') ORDER BY d.kind,d.evidence_id LIMIT 1", (symbol,)).fetchone()
            description = docs[1][:750] if docs else symbol
            all_targets[target] = {'text': description, 'source_id': docs[0] if docs else None}
            context = {}
            for cell in CELLS:
                rr = store.conn.execute('SELECT payload FROM node_context WHERE node=? AND cell=?', (target, cell)).fetchone()
                cr = json.loads(rr[0]) if rr else {}
                ms = cr.get('measurements', [])
                context[cell] = {'detection': cr.get('minimum_detection'), 'mean_log1p_cp10k': ms[0].get('control_mean_log1p_cp10k') if len(ms) == 1 else None}
            targets.append({'target': target, 'evidence': refs, 'function': description[:220], 'actions': sorted({s['action_sign'] for s in records if s.get('action_sign') is not None}), 'measurements': measurements, 'control': context})
        background = [{'text': b['text'], 'source_id': original['evidence'][b['evidence']]['source_id']} for b in original['background']]
        cards[drug] = {'drug': drug, 'name': original['drug_name'], 'roles': original['native_roles'], 'pathway': original['native_pathway'], 'background': background, 'targets': targets, 'evidence': evidence, 'available_target_count': len(target_order), 'omitted_target_count': max(0, len(target_order) - 10)}
    write(root / 'cards.json', cards)
    write(root / 'target_texts.json', all_targets)
    write(root / 'protocol.json', protocol)
