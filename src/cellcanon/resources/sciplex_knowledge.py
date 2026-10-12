from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from cellcanon.model.data import metadata, sha256, SCIPLEX_ROOT, METADATA_ROOT
LEAF = Path('<NORMALIZED_SOURCES_ROOT>')
OUTPUT = Path('<KNOWLEDGE_ROOT>')
ENTITY_SOURCE_ROOT = Path('<ENTITY_SOURCE_ROOT>')
NETWORK_SOURCE_ROOT = Path('<NETWORK_SOURCE_ROOT>')
BIO = Path('<NORMALIZED_SOURCES_ROOT>')

def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + '\n')
    temporary.replace(path)

def normalized_name(value):
    return re.sub('[^a-z0-9]', '', str(value).casefold())

def declared_drug_names(row):
    values = {str(row['canonical_name']), str(row['legacy_drug'])}
    name = str(row['canonical_name'])
    values.update(re.findall('\\(([^)]+)\\)', name))
    values.add(re.sub('\\([^)]*\\)', '', name).strip())
    return sorted((v for v in values if len(normalized_name(v)) >= 3))

def nonempty(value):
    return isinstance(value, str) and value.strip().casefold() not in {'', 'nan', 'none', 'unknown', 'na', 'n/a', '-', 'other', 'others'}

def storage_strings(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    if value.lstrip().startswith('['):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError('Expected an identifier list')
        return storage_strings(parsed)
    return [value.strip()]

def cohort():
    drugs, genes = metadata()
    qpath = BIO / 'drug_identity_isolation_v1/sciplex3_quarantine.json'
    quarantine = json.loads(qpath.read_text())
    kept = quarantine['kept_drugs']['support'] + quarantine['kept_drugs']['validation'] + drugs.loc[drugs.split.eq('test'), 'drug'].tolist()
    drugs = drugs[drugs.drug.isin(kept)].drop(columns=['morgan_fingerprint']).copy()
    assert drugs.groupby('split').size().to_dict() == {'support': 54, 'test': 113, 'validation': 19}
    assert genes.groupby('role').size().to_dict() == {'core': 2000, 'extra': 931}
    return (drugs, genes)

def verified_sources():
    records = {}

    def read(path, expected=None):
        actual = sha256(path)
        if expected is not None:
            assert actual == expected, str(path)
        records[str(path)] = actual
        return pd.read_parquet(path)
    p = json.loads((ENTITY_SOURCE_ROOT / 'prepare_manifest.json').read_text())
    x = json.loads((NETWORK_SOURCE_ROOT / 'network_manifest.json').read_text())
    tables = {}
    for key in ['entities', 'aliases']:
        tables[key] = read(ENTITY_SOURCE_ROOT / Path(p['outputs'][key]).name, p['output_sha256'][key])
    for key in ['reactome_memberships', 'signed_network']:
        item = next((v for v in x['outputs'] if v['name'] == key))
        tables[key] = read(NETWORK_SOURCE_ROOT / Path(item['relative_path']).name, item['sha256'])
    bindings, descriptors = ([], [])
    for folder, manifest, descriptor in [('sciplex_chembl_binding_annotations_v1', 'binding_manifest.json', 'drug_bindings.json'), ('sciplex_binding_query_annotations_v1', 'query_binding_manifest.json', 'query_drug_bindings.json')]:
        base = BIO / folder
        receipt = json.loads((base / manifest).read_text())
        bindings.append(read(base / 'binding_evidence.parquet', receipt['output_sha256']['binding_evidence.parquet']))
        assert sha256(base / descriptor) == receipt['output_sha256'][descriptor]
        records[str(base / descriptor)] = sha256(base / descriptor)
        records[str(base / manifest)] = sha256(base / manifest)
        descriptors.extend(json.loads((base / descriptor).read_text()))
    tables['bindings'] = pd.concat(bindings, ignore_index=True)
    tables['binding_descriptors'] = descriptors
    for path in [ENTITY_SOURCE_ROOT / 'prepare_manifest.json', NETWORK_SOURCE_ROOT / 'network_manifest.json', METADATA_ROOT / 'prepare_manifest.json', SCIPLEX_ROOT / 'build_manifest.json', BIO / 'drug_identity_isolation_v1/sciplex3_quarantine.json']:
        records[str(path)] = sha256(path)
    return (tables, records)

def local_audit(output=OUTPUT):
    root = output / 'local_audit'
    root.mkdir(parents=True, exist_ok=True)
    drugs, genes = cohort()
    tables, sources = verified_sources()
    entities = tables['entities']
    alias = tables['aliases'].query("entity_type == 'drug'")
    aliases = defaultdict(set)
    for row in alias.itertuples():
        aliases[normalized_name(row.alias)].add(str(row.canonical_id))
    drug_entities = entities.query("entity_type == 'drug'").groupby('canonical_id')
    descriptions = {r['drug']: r for r in tables['binding_descriptors']}
    rows = []
    for record in drugs.to_dict('records'):
        candidates = set().union(*(aliases.get(normalized_name(n), set()) for n in declared_drug_names(record)))
        matched = next(iter(candidates)) if len(candidates) == 1 else None
        text = drug_entities.get_group(matched).entity_text.tolist() if matched in drug_entities.groups else []
        binding = descriptions[record['drug']]
        record.update({'native_moa_nonempty': nonempty(record['drug_moa']), 'native_roles_nonempty': any((nonempty(v) for v in record['target_roles'])), 'legacy_entity_candidates': sorted(candidates), 'legacy_unique_entity': matched, 'legacy_text_available': any((nonempty(t) for t in text)), 'legacy_text_species_or_identity_independently_verified': False, 'chembl_ids': binding.get('annotation_reference_drugs', []), 'binding_measured_genes': binding.get('measured_genes', []), 'binding_active_genes_1uM': binding.get('active_genes_1uM', []), 'binding_records': binding.get('binding_records', 0)})
        rows.append(record)
    drug_table = pd.DataFrame(rows)
    eg = entities.query("entity_type == 'gene'").copy()
    original_gene_rows = len(eg)
    eg['ensembl_key'] = eg.ensembl_id.map(storage_strings)
    eg = eg.explode('ensembl_key')
    eg['ensembl_key'] = eg.ensembl_key.fillna('').str.replace('\\.\\d+$', '', regex=True)
    groups = eg.groupby('ensembl_key')
    memberships = tables['reactome_memberships']
    assert memberships.species.eq('Homo sapiens').all()
    pathway_genes = set(memberships.gene_symbol)
    network = tables['signed_network']
    network_genes = set(network.source_gene_symbol) | set(network.target_gene_symbol)
    gene_rows = []
    for row in genes.to_dict('records'):
        group = groups.get_group(row['gene']) if row['gene'] in groups.groups else eg.iloc[:0]
        gene_rows.append({**row, 'legacy_exact_ensembl_rows': len(group), 'legacy_text_available': any((nonempty(t) for t in group.entity_text)), 'legacy_unique_accessions': len({a for value in group.uniprot_accession for a in storage_strings(value)}), 'legacy_species_verified': False, 'reactome_symbol_member': row['gene_symbol'] in pathway_genes, 'network_symbol_incident': row['gene_symbol'] in network_genes})
    gene_table = pd.DataFrame(gene_rows)
    drug_table.to_parquet(root / 'drug_coverage.parquet', index=False)
    gene_table.to_parquet(root / 'gene_coverage.parquet', index=False)
    tables['bindings'].to_parquet(root / 'binding_evidence.parquet', index=False)
    drug_summary = {}
    for split, frame in drug_table.groupby('split'):
        drug_summary[split] = {'drugs': len(frame), 'native_moa': int(frame.native_moa_nonempty.sum()), 'native_roles': int(frame.native_roles_nonempty.sum()), 'legacy_unique_entity': int(frame.legacy_unique_entity.notna().sum()), 'legacy_text_available': int(frame.legacy_text_available.sum()), 'chembl_mapped': int(frame.chembl_ids.map(len).gt(0).sum()), 'binding_measured': int(frame.binding_records.gt(0).sum()), 'binding_active_1uM': int(frame.binding_active_genes_1uM.map(len).gt(0).sum())}
    gene_summary = {role: {'genes': len(frame), 'legacy_exact_ensembl': int(frame.legacy_exact_ensembl_rows.gt(0).sum()), 'legacy_text_available': int(frame.legacy_text_available.sum()), 'legacy_multiple_accessions': int(frame.legacy_unique_accessions.gt(1).sum()), 'reactome_symbol_member': int(frame.reactome_symbol_member.sum()), 'network_symbol_incident': int(frame.network_symbol_incident.sum())} for role, frame in gene_table.groupby('role')}
    summary = {'status': 'LOCAL_BACKGROUND_INSPECTED', 'created_utc': datetime.now(timezone.utc).isoformat(), 'drug_coverage': drug_summary, 'gene_coverage': gene_summary, 'legacy_gene_rows': original_gene_rows, 'legacy_gene_identifier_rows_after_multivalue_expansion': len(eg), 'legacy_unique_ensembl': int(eg.ensembl_key.nunique()), 'claim_boundary': 'Legacy text/symbol availability only; no independent species validation or condition mechanism correctness.', 'source_sha256': sources, 'output_sha256': {p.name: sha256(p) for p in root.glob('*.parquet')}, 'model_api_calls': 0, 'public_http_calls': 0, 'perturbation_response_panel_reads': 0}
    write_json(root / 'manifest.json', summary)
    print(json.dumps({'drug_coverage': drug_summary, 'gene_coverage': gene_summary}, ensure_ascii=False), flush=True)
