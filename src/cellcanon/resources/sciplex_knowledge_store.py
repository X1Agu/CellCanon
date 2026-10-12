from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import pandas as pd
from cellcanon.resources.sciplex_knowledge import OUTPUT, ENTITY_SOURCE_ROOT, LEAF, nonempty, declared_drug_names, sha256, write_json
from cellcanon.resources.sciplex_drug_background import same_drug
from cellcanon.model.chembl_binding_annotations import threshold_evidence
from cellcanon.resources.identifiers import _gene_symbol

def evidence_id(namespace, value):
    return namespace + ':' + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]

def compile_store(output=OUTPUT):
    root = output / 'store'
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for stage in ['local_audit', 'drug_background', 'gene_background', 'network_background', 'cell_baseline', 'catalog_identity', 'catalog_annotations', 'ensembl_background']:
        path = output / stage / 'manifest.json'
        m = json.loads(path.read_text())
        for name, expected in m['output_sha256'].items():
            assert sha256(output / stage / name) == expected, (stage, name)
        receipts[stage] = {'path': str(path), 'sha256': sha256(path), 'status': m['status']}
    override_path = LEAF / 'configs/metadata/sciplex_background_source_overrides.json'
    overrides = {r['drug']: r for r in json.loads(override_path.read_text())['records']}
    receipts['supplier_extract'] = {'path': str(override_path), 'sha256': sha256(override_path)}
    upath = output / 'public_background/uniprot_manifest.json'
    um = json.loads(upath.read_text())
    assert sha256(output / 'public_background/uniprot_human_reviewed.parquet') == um['output_sha256']
    receipts['human_proteins'] = {'path': str(upath), 'sha256': sha256(upath)}
    dbpath = root / 'knowledge.sqlite'
    temp = root / 'knowledge.building.sqlite'
    if temp.exists():
        temp.unlink()
    conn = sqlite3.connect(temp)
    conn.executescript('\n    CREATE TABLE drugs(drug TEXT PRIMARY KEY, split TEXT, payload TEXT NOT NULL);\n    CREATE TABLE genes(gene TEXT PRIMARY KEY, symbol TEXT, canonical_symbol TEXT, role TEXT, payload TEXT NOT NULL);\n    CREATE TABLE aliases(alias TEXT, drug TEXT, UNIQUE(alias,drug));\n    CREATE TABLE documents(evidence_id TEXT PRIMARY KEY, entity_id TEXT, kind TEXT, source_url TEXT, text TEXT, payload TEXT);\n    CREATE TABLE document_symbols(symbol TEXT, evidence_id TEXT, UNIQUE(symbol,evidence_id));\n    CREATE VIRTUAL TABLE document_search USING fts5(evidence_id UNINDEXED,entity_id UNINDEXED,text);\n    CREATE TABLE store_metadata(key TEXT PRIMARY KEY,value TEXT);\n    CREATE INDEX document_entity ON documents(entity_id);\n    CREATE INDEX gene_symbol ON genes(symbol);\n    CREATE INDEX gene_canonical_symbol ON genes(canonical_symbol);\n    ')

    def document(entity, kind, text, url, payload):
        if not nonempty(text):
            return None
        key = evidence_id(kind, [entity, text, url])
        result = conn.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?)', (key, entity, kind, url, text, json.dumps(payload, ensure_ascii=False)))
        if result.rowcount:
            conn.execute('INSERT INTO document_search VALUES(?,?,?)', (key, entity, text))
        return key
    genes = json.loads((output / 'gene_background/gene_cards.json').read_text())
    ensembl = {r['gene']: r for r in json.loads((output / 'ensembl_background/gene_metadata.json').read_text())}
    uni = pd.read_parquet(output / 'public_background/uniprot_human_reviewed.parquet')
    uni_docs = {}
    for u in uni.to_dict('records'):
        key = document('protein:' + u['Entry'], 'reviewed_human_function', u['Function [CC]'], 'https://www.uniprot.org/uniprotkb/' + u['Entry'] + '/entry', {'taxon': 9606, 'gene_primary': u['Gene Names (primary)']})
        if key:
            uni_docs[u['Entry']] = key
            for symbol in u['Gene Names (primary)'].split():
                conn.execute('INSERT OR IGNORE INTO document_symbols VALUES(?,?)', (symbol, key))
    for g in genes:
        ids = [uni_docs[a] for a in g['uniprot_accessions'] if a in uni_docs]
        if g['ncbi_summary']:
            key = document(g['gene'], 'ncbi_human_gene_summary', g['ncbi_summary'], g['ncbi_summary_url'], {'taxon': 9606, 'gene_id': g['ncbi_gene_id']})
            ids.append(key)
        g['ensembl_metadata'] = ensembl[g['gene']]
        em = g['ensembl_metadata']
        if nonempty(em['description']):
            key = document(g['gene'], 'ensembl_identity_description', em['description'], em['source_url'], {'taxon': 9606, 'biotype': em['biotype'], 'assembly_name': em['assembly_name'], 'scope': 'identity_description_not_functional_mechanism'})
            ids.append(key)
        if nonempty(g['gene_description']):
            url = 'https://www.ncbi.nlm.nih.gov/gene/' + g['ncbi_gene_id'] if g['ncbi_gene_id'] else 'https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/' + g['hgnc_ids'][0] if g['hgnc_ids'] else em['source_url']
            ids.append(document(g['gene'], 'human_gene_identity_description', g['gene_description'], url, {'taxon': 9606, 'scope': 'identity_description_not_functional_mechanism', 'identity_status': g['identity_status']}))
        g['evidence_ids'] = ids
        g['descriptive_text_available'] = bool(g['descriptive_text_available'] or nonempty(em['description']))
        for key in ids:
            conn.execute('INSERT OR IGNORE INTO document_symbols VALUES(?,?)', (g['canonical_symbol'], key))
        conn.execute('INSERT INTO genes VALUES(?,?,?,?,?)', (g['gene'], g['gene_symbol'], g['canonical_symbol'], g['role'], json.dumps(g, ensure_ascii=False)))
    pub = {r['drug']: r for r in json.loads((output / 'drug_background/pubchem_drug_background.json').read_text())}
    catalog = {r['drug']: r for r in json.loads((output / 'catalog_identity/resolution.json').read_text()) if r['accepted']}
    catalog_pub = {r['drug']: r for r in json.loads((output / 'catalog_annotations/pubchem_drug_background.json').read_text())}
    mapping = pd.read_parquet(output / 'drug_background/chembl_identity.parquet')
    drug_molecules = mapping.groupby('drug').molecule_chembl_id.agg(list).to_dict()
    for drug, r in catalog.items():
        assert drug not in drug_molecules
        drug_molecules[drug] = r['molecule_chembl_ids']
    mechanisms = json.loads((output / 'drug_background/chembl_mechanisms.json').read_text())
    mechanisms += json.loads((output / 'catalog_annotations/mechanisms.json').read_text())
    bym = defaultdict(list)
    for m in mechanisms:
        bym[m['molecule_chembl_id']].append(m)
    targets = {t['target_chembl_id']: t for t in json.loads((output / 'drug_background/chembl_mechanism_targets.json').read_text())}
    targets.update({t['target_chembl_id']: t for t in json.loads((output / 'catalog_annotations/targets.json').read_text())})
    binding = pd.read_parquet(output / 'drug_background/binding_evidence.parquet')
    binding['identity_tier'] = 'strict_native_full_standard_parent_inchi'
    binding = pd.concat([binding, pd.read_parquet(output / 'catalog_annotations/binding_evidence.parquet')], ignore_index=True)
    assert not binding.duplicated(['drug', 'activity_id']).any()
    binding['evidence_id'] = [evidence_id('chembl_binding', [r.drug, int(r.activity_id)]) for r in binding.itertuples()]
    binding['supports_active_1uM'] = [threshold_evidence(r.value_nM, r.relation, 1000.0)[0] for r in binding.itertuples()]
    binding.to_sql('binding_evidence', conn, index=False)
    conn.execute('CREATE INDEX binding_drug ON binding_evidence(drug)')
    conn.execute('CREATE UNIQUE INDEX binding_id ON binding_evidence(evidence_id)')
    dc = pd.read_parquet(output / 'network_background/drugcentral_moa.parquet')
    dc.to_sql('drugcentral_moa', conn, index=False)
    local = pd.read_parquet(output / 'local_audit/drug_coverage.parquet')
    legacy = pd.read_parquet(ENTITY_SOURCE_ROOT / 'entity_registry.parquet').query("entity_type == 'drug'")
    cards = []
    for d in local.to_dict('records'):
        ids = []
        identity_tier = 'catalog_reconciled_not_native_full_identity' if d['drug'] in catalog else 'strict_native_full_standard_parent_inchi' if d['drug'] in drug_molecules else 'no_chembl_identity'
        pubrows = [(pub[d['drug']], 'strict_native_full_standard_parent_inchi')]
        if d['drug'] in catalog_pub:
            pubrows.append((catalog_pub[d['drug']], 'catalog_reconciled_not_native_full_identity'))
        for pr, tier in pubrows:
            for p in pr['descriptions']:
                key = document(d['drug'], 'pubchem_identity_verified_description', p.get('Description'), p.get('DescriptionURL') or pr.get('description_url'), {'cid': pr['cid'], 'source_name': p.get('DescriptionSourceName'), 'scope': 'generic_drug_background', 'identity_tier': tier, 'response_free_source_screen': 'public_compound_annotation_not_benchmark_labels'})
                if key:
                    ids.append(key)
        mechs = []
        mechanism_genes = set()
        for mid in drug_molecules.get(d['drug'], []):
            for m in bym[mid]:
                target = targets.get(m.get('target_chembl_id'), {})
                url = 'https://www.ebi.ac.uk/chembl/explore/compound/' + mid
                key = document(d['drug'], 'chembl_mechanism', m.get('mechanism_of_action'), url, {'mechanism_id': m.get('mec_id'), 'references': m.get('mechanism_refs', []), 'source_release': 'ChEMBL_37', 'condition_effect_observed': False, 'target_organism': target.get('organism'), 'identity_tier': identity_tier})
                if key:
                    ids.append(key)
                if target.get('organism') == 'Homo sapiens':
                    mechanism_genes.update((g for component in target.get('target_components', []) if (g := _gene_symbol(component)[0])))
                mechs.append({'evidence_id': key, 'target_chembl_id': m.get('target_chembl_id'), 'target_type': target.get('target_type'), 'organism': target.get('organism'), 'action_type': m.get('action_type'), 'direct_interaction': m.get('direct_interaction'), 'target_components': target.get('target_components', []), 'identity_tier': identity_tier})
        dc_ids = []
        for r in dc.loc[dc.drug.eq(d['drug'])].to_dict('records'):
            key = document(d['drug'], 'drugcentral_human_moa', f"{r['DRUG_NAME']}: {r['ACTION_TYPE']} of {r['TARGET_NAME']} ({r['GENE']}); curated drug mechanism target.", 'https://drugcentral.org/drugcard/' + str(r['STRUCT_ID']), {'taxon': 9606, 'gene_symbol': r['canonical_gene_symbol'], 'source_release': r['source_release'], 'references': [r['MOA_SOURCE_URL'], r['ACT_SOURCE_URL']], 'condition_effect_observed': False, 'identity_tier': 'strict_native_full_standard_parent_inchi'})
            if key:
                ids.append(key)
                dc_ids.append(key)
        supplier = overrides.get(d['drug'])
        supplier_ids = []
        if supplier:
            assert supplier['catalog_number'] == d['drug'] and same_drug(d['canonical_smiles'], supplier['source_smiles'])
            key = document(d['drug'], 'supplier_reported_pharmacology', supplier['statement'], supplier['source_url'], supplier)
            if key:
                ids.append(key)
                supplier_ids.append(key)
        old = legacy[legacy.canonical_id.eq(d['legacy_unique_entity'])]
        legacy_ids = []
        for v in old.to_dict('records'):
            key = document(d['drug'], 'unverified_legacy_context', v['entity_text'], str(ENTITY_SOURCE_ROOT / 'entity_registry.parquet'), {'text_source': v['text_source'], 'claim_status': 'unverified_legacy_summary_not_external_fact'})
            if key:
                legacy_ids.append(key)
        group = binding[binding.drug.eq(d['drug'])]
        active = sorted({r.target_gene_symbol for r in group.itertuples() if threshold_evidence(r.value_nM, r.relation, 1000.0)[0]})
        card = {'drug': d['drug'], 'split': d['split'], 'canonical_name': d['canonical_name'], 'canonical_smiles': d['canonical_smiles'], 'native_role_labels': list(d['target_roles']), 'native_pathway_label': d['drug_moa'], 'native_annotation_scope': 'catalog_background_not_verified_condition_mechanism', 'chembl_ids': drug_molecules.get(d['drug'], []), 'pubchem_cid': pub[d['drug']]['cid'], 'chembl_identity_tier': identity_tier, 'catalog_resolution': catalog.get(d['drug']), 'supplier_reported_targets': [supplier['reported_target_symbol']] if supplier else [], 'supplier_evidence_ids': supplier_ids, 'drugcentral_evidence_ids': dc_ids, 'evidence_ids': sorted(set(ids)), 'legacy_context_ids': sorted(set(legacy_ids)), 'mechanisms': mechs, 'measured_target_genes': sorted(set(group.target_gene_symbol)), 'mechanism_component_genes': sorted(mechanism_genes), 'active_target_genes_1uM': active, 'binding_records': len(group), 'drugcentral_moa_genes': sorted(set(dc.loc[dc.drug.eq(d['drug']), 'canonical_gene_symbol'].dropna())), 'condition_causal_mechanism_verified': False}
        cards.append(card)
        conn.execute('INSERT INTO drugs VALUES(?,?,?)', (d['drug'], d['split'], json.dumps(card, ensure_ascii=False)))
        for name in [d['drug'], *declared_drug_names(d)]:
            conn.execute('INSERT OR IGNORE INTO aliases VALUES(?,?)', (name.casefold().strip(), d['drug']))
    for filename, table in [('reactome_memberships', 'reactome_symbols'), ('reactome_ensembl_memberships', 'reactome_ensembl'), ('go_memberships', 'go_memberships'), ('regulatory_network', 'regulatory_network'), ('complex_components', 'complex_components')]:
        data = pd.read_parquet(output / 'network_background' / (filename + '.parquet'))
        data.to_sql(table, conn, index=False)
    pd.read_parquet(output / 'cell_baseline/all_human_control_baseline.parquet').to_sql('cell_baseline', conn, index=False)
    conn.executescript('\n    CREATE INDEX baseline_lookup ON cell_baseline(cell_line,gene_symbol);\n    CREATE INDEX pathway_symbol ON reactome_symbols(canonical_symbol);\n    CREATE INDEX pathway_ensembl ON reactome_ensembl(ensembl_id);\n    CREATE INDEX go_symbol ON go_memberships(canonical_symbol);\n    CREATE INDEX complex_symbol ON complex_components(canonical_symbol);\n    CREATE INDEX network_source ON regulatory_network(source_node_id);\n    CREATE INDEX network_target ON regulatory_network(target_node_id);\n    CREATE INDEX document_symbol ON document_symbols(symbol);\n    ')
    for key, value in {'dataset': 'sciplex3', 'cells': ['A549', 'K562', 'MCF7'], 'sources': receipts, 'numeric_authority': 'background_only_no_model_change', 'layer5_complete_routes': 'deferred'}.items():
        conn.execute('INSERT INTO store_metadata VALUES(?,?)', (key, json.dumps(value)))
    conn.commit()
    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    counts = {t: conn.execute('SELECT count(*) FROM ' + t).fetchone()[0] for t in ['drugs', 'genes', 'documents', 'binding_evidence', 'cell_baseline']}
    conn.close()
    temp.replace(dbpath)
    write_json(root / 'drug_cards.json', cards)
    write_json(root / 'manifest.json', {'status': 'LOCAL_KNOWLEDGE_STORE_COMPILED_PENDING_INDEPENDENT_VERIFICATION', 'counts': counts, 'inputs': receipts, 'output_sha256': {'knowledge.sqlite': sha256(dbpath), 'drug_cards.json': sha256(root / 'drug_cards.json')}, 'llm_calls': 0, 'predictor_updates': 0})
    print(json.dumps(counts), flush=True)

class LocalKnowledge:

    def __init__(self, path=OUTPUT / 'store/knowledge.sqlite'):
        self.conn = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        self.conn.close()

    def drug(self, name):
        rows = self.conn.execute('SELECT DISTINCT drug FROM aliases WHERE alias=?', (name.casefold().strip(),)).fetchall()
        if len(rows) != 1:
            raise ValueError('Drug identity is missing or ambiguous: ' + name)
        return json.loads(self.conn.execute('SELECT payload FROM drugs WHERE drug=?', (rows[0]['drug'],)).fetchone()[0])

    def evidence(self, key):
        row = self.conn.execute('SELECT * FROM documents WHERE evidence_id=?', (key,)).fetchone()
        if row is not None:
            return dict(row)
        row = self.conn.execute('SELECT * FROM binding_evidence WHERE evidence_id=?', (key,)).fetchone()
        if row is not None:
            r = dict(row)
            return {'evidence_id': key, 'entity_id': r['drug'], 'kind': 'human_single_protein_binding', 'source_url': 'https://www.ebi.ac.uk/chembl/api/data/activity/' + str(r['activity_id']) + '.json', 'text': f"{r['target_gene_symbol']}: {r['measurement_type']} {r['relation']} {r['value_nM']} nM.", 'payload': json.dumps({**r, 'taxon': 9606, 'condition_effect_observed': False})}
        if key.startswith('network:'):
            row = self.conn.execute('SELECT * FROM regulatory_network WHERE edge_id=?', (key.split(':', 1)[1],)).fetchone()
            if row is not None:
                r = dict(row)
                return {'evidence_id': key, 'entity_id': r['source_node_id'], 'kind': 'background_network_relation', 'source_url': 'https://omnipathdb.org/interactions?organisms=9606', 'text': f"{r['source_node_id']} to {r['target_node_id']}; direction and sign follow the attached consensus fields.", 'payload': json.dumps(r)}
        raise KeyError(key)

    def binding(self, drug, symbol=None, limit=5):
        d = self.drug(drug)
        sql = 'SELECT evidence_id FROM binding_evidence WHERE drug=?'
        args = [d['drug']]
        if symbol is not None:
            sql += ' AND target_gene_symbol=?'
            args.append(symbol)
        sql += ' ORDER BY supports_active_1uM DESC,value_nM,activity_id LIMIT ?'
        args.append(min(50, max(1, limit)))
        return [self.evidence(r[0]) for r in self.conn.execute(sql, args)]

    def gene_evidence(self, symbol, limit=3):
        sql = "SELECT d.evidence_id FROM document_symbols s JOIN documents d USING(evidence_id) WHERE s.symbol=? ORDER BY CASE d.kind WHEN 'reviewed_human_function' THEN 0 WHEN 'ncbi_human_gene_summary' THEN 1 ELSE 2 END,d.evidence_id LIMIT ?"
        return [self.evidence(r[0]) for r in self.conn.execute(sql, (symbol, min(20, max(1, limit))))]

    def baseline(self, symbol, cell):
        if cell not in {'A549', 'K562', 'MCF7'}:
            raise ValueError('Unsupported cell for this SciPlex baseline: ' + cell)
        sql = 'SELECT ensembl_id,gene_symbol,cell_line,control_cells,control_mean_log1p_cp10k,control_detection_fraction FROM cell_baseline WHERE cell_line=? AND (gene_symbol=? OR ensembl_id IN (SELECT gene FROM genes WHERE canonical_symbol=?)) ORDER BY ensembl_id'
        return [dict(r) for r in self.conn.execute(sql, (cell, symbol, symbol))]

    def gene(self, name):
        rows = self.conn.execute('SELECT DISTINCT gene,payload FROM genes WHERE gene=? OR symbol=? OR canonical_symbol=?', (name, name, name)).fetchall()
        if len(rows) != 1:
            raise ValueError('Gene identity is missing or ambiguous in the SciPlex output panel: ' + name)
        return json.loads(rows[0]['payload'])

    def relationships(self, symbol, limit=10):
        nodes = ['GENE:' + symbol] + [r[0] for r in self.conn.execute('SELECT DISTINCT complex_id FROM complex_components WHERE canonical_symbol=?', (symbol,))]
        placeholders = ','.join(('?' for _ in nodes))
        sql = f'SELECT edge_id,source_node_id,target_node_id,source_node_type,target_node_type,consensus_direction,sign,sign_status,references_json FROM regulatory_network WHERE typed_nodes_resolved=1 AND (source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders})) ORDER BY typed_signed_usable DESC,reference_count DESC,edge_id LIMIT ?'
        rows = []
        for row in self.conn.execute(sql, (*nodes, *nodes, min(50, max(1, limit)))):
            r = dict(row)
            r['evidence_id'] = 'network:' + r['edge_id']
            r['reference_urls'] = ['https://pubmed.ncbi.nlm.nih.gov/' + ref.rsplit(':', 1)[-1] + '/' for ref in json.loads(r.pop('references_json')) if ref.rsplit(':', 1)[-1].isdigit()][:3]
            r['semantics'] = 'background_relation_not_condition_effect; complex_nodes_remain_complexes'
            rows.append(r)
        return rows

    def search(self, query, limit=5, *, include_legacy=False):
        tokens = re.findall('[\\w-]+', query)[:12]
        if not tokens:
            return []
        text = ' OR '.join(('"' + t.replace('"', '""') + '"' for t in tokens))
        sql = 'SELECT s.evidence_id FROM document_search s JOIN documents d USING(evidence_id) WHERE document_search MATCH ? ' + ('' if include_legacy else "AND d.kind!='unverified_legacy_context' ") + 'ORDER BY bm25(document_search),s.evidence_id LIMIT ?'
        rows = self.conn.execute(sql, (text, min(30, max(1, limit)))).fetchall()
        return [self.evidence(row[0]) for row in rows]

    def condition(self, drug, cell, *, max_chars=8000):
        if cell not in {'A549', 'K562', 'MCF7'}:
            raise ValueError('Unsupported cell for this SciPlex baseline: ' + cell)
        if max_chars < 2000:
            raise ValueError('The packet budget must allow at least 2000 characters for identity and provenance.')
        d = self.drug(drug)
        packet = {'drug': d['drug'], 'name': d['canonical_name'], 'cell': cell, 'dataset': 'sciplex3', 'native_roles': d['native_role_labels'], 'native_pathway': d['native_pathway_label'], 'native_annotation_scope': d['native_annotation_scope'], 'chembl_identity_tier': d['chembl_identity_tier'], 'evidence': [], 'target_background': [], 'condition_causal_mechanism_verified': False, 'scope': 'Public drug/gene background and 24h vehicle controls; no perturbation-response labels.', 'baseline_source_manifest': json.loads(self.conn.execute("SELECT value FROM store_metadata WHERE key='sources'").fetchone()[0])['cell_baseline'], 'retrieval_truncated': False}

        def add(field, value):
            packet[field].append(value)
            if len(json.dumps(packet, ensure_ascii=False)) > max_chars:
                packet[field].pop()
                packet['retrieval_truncated'] = True
                return False
            return True

        def excerpt(e, chars=450):
            payload = json.loads(e['payload'])
            return {'evidence_id': e['evidence_id'], 'kind': e['kind'], 'source_url': e['source_url'], 'text_excerpt': e['text'][:chars], 'context': {k: payload[k] for k in ['taxon', 'target_organism', 'condition_effect_observed', 'scope', 'identity_tier', 'evidence_tier', 'human_target_binding_verified', 'assay_organism'] if k in payload}, 'excerpt_truncated': len(e['text']) > chars}
        priority = {'chembl_mechanism': 0, 'drugcentral_human_moa': 1, 'supplier_reported_pharmacology': 2, 'pubchem_identity_verified_description': 3}
        docs = sorted([self.evidence(key) for key in d['evidence_ids']], key=lambda e: (priority.get(e['kind'], 9), e['evidence_id']))
        for e in docs[:2]:
            add('evidence', excerpt(e))
        target_genes = list(dict.fromkeys(d['mechanism_component_genes'] + d['drugcentral_moa_genes'] + d['active_target_genes_1uM'] + d['supplier_reported_targets'] + d['measured_target_genes']))
        for symbol in target_genes[:6]:
            base = self.baseline(symbol, cell)
            pathways = [dict(r) for r in self.conn.execute('SELECT DISTINCT pathway_id,pathway_name FROM reactome_symbols WHERE canonical_symbol=? ORDER BY pathway_id LIMIT 3', (symbol,))]
            for r in pathways:
                r['source_url'] = 'https://reactome.org/PathwayBrowser/#/' + r['pathway_id']
            value = {'gene_symbol': symbol, 'active_binding_evidence': symbol in d['active_target_genes_1uM'], 'mechanism_component': symbol in d['mechanism_component_genes'], 'supplier_reported_only': symbol in d['supplier_reported_targets'] and symbol not in d['measured_target_genes'], 'binding_evidence': [excerpt(e, 120) for e in self.binding(d['drug'], symbol, limit=1)], 'gene_evidence': [excerpt(e) for e in self.gene_evidence(symbol, limit=1)], 'control_baseline': base, 'pathway_memberships': pathways, 'pathway_semantics': 'membership_not_activation', 'network_relations': self.relationships(symbol, limit=1)}
            if not add('target_background', value):
                value['network_relations'] = []
                value['pathway_memberships'] = []
                add('target_background', value)
        if len(docs) > len(packet['evidence']) or len(target_genes) > len(packet['target_background']):
            packet['retrieval_truncated'] = True
        assert len(json.dumps(packet, ensure_ascii=False)) <= max_chars
        return packet
