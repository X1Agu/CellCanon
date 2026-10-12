"""Offline converters for source exports; no network requests or model calls."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import shutil
import zipfile
import pandas as pd

EXPECTED_REACTOME_RELEASE = 'v97'
RETRIEVAL_DATE = 'provided_snapshot'
NETWORK_COLUMNS = ('source', 'target', 'source_genesymbol', 'target_genesymbol',
    'is_directed', 'is_stimulation', 'is_inhibition', 'consensus_direction',
    'consensus_stimulation', 'consensus_inhibition', 'sources', 'references', 'curation_effort')


def stable_hash(value, length=16):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:length]


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(label + ': missing columns ' + ', '.join(missing))


def assert_unique(frame, columns, label):
    require_columns(frame, columns, label)
    if frame.duplicated(columns).any():
        raise ValueError(label + ': duplicate keys ' + ', '.join(columns))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_manifest(root, name, files, sources=None, provenance=None):
    save_json(root/name, {'outputs': [{'name': key, 'relative_path': filename,
        'sha256': digest(root/filename)} for key, filename in files.items()],
        'source_sha256': {key: digest(path) for key,path in (sources or {}).items()},
        'provenance': provenance or {},
        'scope': 'Public biological background; no perturbation response labels.'})


def table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, sep='\t' if path.suffix in ('.tsv', '.txt') else ',', keep_default_na=False)


def records(path, key=None):
    data=json.loads(Path(path).read_text())
    if isinstance(data,dict):
        if key is None or key not in data:
            raise ValueError('JSON export must be a list or contain its documented record key.')
        data=data[key]
    if not isinstance(data,list) or not all(isinstance(row,dict) for row in data):
        raise ValueError('JSON records must be a list of objects.')
    return data


def parse_reactome_gmt(raw_zip: bytes, *, member: str, fixed_genes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        if archive.namelist() != [member]:
            raise ValueError('Reactome archive member changed')
        lines = archive.read(member).decode('utf-8').splitlines()
    pathway_rows = []
    member_rows = []
    for line in lines:
        fields = line.rstrip('\n').split('\t')
        if len(fields) < 3:
            raise ValueError('Reactome GMT row is malformed')
        (name, pathway_id, *genes) = fields
        unique_genes = list(dict.fromkeys(genes))
        pathway_rows.append({'pathway_id': pathway_id, 'pathway_name': name, 'member_count': len(unique_genes), 'fixed_gene_member_count': sum((gene in fixed_genes for gene in unique_genes)), 'species': 'Homo sapiens', 'source_release': EXPECTED_REACTOME_RELEASE})
        member_rows.extend(({'pathway_id': pathway_id, 'pathway_name': name, 'gene_symbol': gene, 'gene_in_fixed_universe': gene in fixed_genes, 'species': 'Homo sapiens', 'source_release': EXPECTED_REACTOME_RELEASE} for gene in unique_genes))
    pathways = pd.DataFrame(pathway_rows)
    memberships = pd.DataFrame(member_rows)
    assert_unique(pathways, ['pathway_id'], 'Reactome pathways')
    assert_unique(memberships, ['pathway_id', 'gene_symbol'], 'Reactome memberships')
    return (pathways.sort_values('pathway_id').reset_index(drop=True), memberships.sort_values(['pathway_id', 'gene_symbol']).reset_index(drop=True))

def normalize_omnipath(raw: bytes, *, dataset: str, interaction_type: str, fixed_genes: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), sep='\t', low_memory=False)
    require_columns(frame, NETWORK_COLUMNS, f'OmniPath {dataset}')
    if dataset == 'dorothea_abc':
        require_columns(frame, ['dorothea_level'], 'DoRothEA A/B/C')
        levels = frame.dorothea_level.map(lambda value: set(str(value).split(';')))
        if not levels.map(lambda value: bool(value & {'A','B','C'}) and value <= {'A','B','C','D','E'}).all():
            raise ValueError('Every DoRothEA row must contain an A, B or C confidence assignment.')
    rows = []
    for (source_row_index, row) in enumerate(frame.itertuples(index=False)):
        values = row._asdict()
        boolean = {key: str(values[key]).strip().casefold() == 'true' for key in ('is_directed', 'is_stimulation', 'is_inhibition', 'consensus_direction', 'consensus_stimulation', 'consensus_inhibition')}
        stimulation = boolean['consensus_stimulation']
        inhibition = boolean['consensus_inhibition']
        if stimulation and inhibition:
            (sign, sign_status) = (None, 'conflict')
        elif stimulation:
            (sign, sign_status) = (1, 'known_positive')
        elif inhibition:
            (sign, sign_status) = (-1, 'known_negative')
        else:
            (sign, sign_status) = (None, 'unknown')
        sources = sorted(set(str(values.get('sources') or '').split(';')) - {'', 'nan'})
        references = sorted(set(str(values.get('references') or '').split(';')) - {'', 'nan'})
        source_gene = str(values['source_genesymbol']) if pd.notna(values['source_genesymbol']) else None
        target_gene = str(values['target_genesymbol']) if pd.notna(values['target_genesymbol']) else None
        curation_effort = pd.to_numeric(pd.Series([values.get('curation_effort')]), errors='coerce').iloc[0]
        dorothea_level = values.get('dorothea_level')
        if pd.isna(dorothea_level):
            dorothea_level = None
        identity = {'dataset': dataset, 'source': values['source'], 'target': values['target'], 'sources': sources, 'references': references, 'boolean': boolean, 'dorothea_level': dorothea_level, 'source_row_index': source_row_index}
        resolved_genes = source_gene is not None and target_gene is not None
        rows.append({'edge_id': stable_hash(identity), 'source_row_index': source_row_index, 'dataset': dataset, 'interaction_type': interaction_type, 'source_uniprot': str(values['source']), 'target_uniprot': str(values['target']), 'source_gene_symbol': source_gene, 'target_gene_symbol': target_gene, **boolean, 'sign': sign, 'sign_status': sign_status, 'curation_effort': int(curation_effort) if pd.notna(curation_effort) else pd.NA, 'resource_count': len(sources), 'reference_count': len(references), 'sources_json': json.dumps(sources, ensure_ascii=False), 'references_json': json.dumps(references, ensure_ascii=False), 'dorothea_level': dorothea_level, 'source_gene_in_fixed_universe': bool(source_gene is not None and source_gene in fixed_genes), 'target_gene_in_fixed_universe': bool(target_gene is not None and target_gene in fixed_genes), 'path_eligible': boolean['consensus_direction'] and resolved_genes, 'signed_path_eligible': boolean['consensus_direction'] and resolved_genes and (sign in (-1, 1)), 'provenance_source': 'OmniPath web service', 'provenance_version': f'retrieved_{RETRIEVAL_DATE}', 'query_response_visible': False})
    result = pd.DataFrame(rows)
    result['sign'] = pd.array(result['sign'], dtype='Int8')
    result['curation_effort'] = pd.array(result['curation_effort'], dtype='Int64')
    assert_unique(result, ['edge_id'], f'OmniPath {dataset} edges')
    return result.sort_values(['source_gene_symbol', 'target_gene_symbol', 'edge_id'], kind='stable').reset_index(drop=True)


def convert_network(reactome_zip, member, omnipath, dorothea, output, source_release):
    """Preserve the source normalizers' direction/sign and stable edge identities."""
    if source_release != 'v97':
        raise ValueError('This main resource protocol supports Reactome v97 only.')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    _,members=parse_reactome_gmt(Path(reactome_zip).read_bytes(),member=member,fixed_genes=set())
    a=normalize_omnipath(Path(omnipath).read_bytes(),dataset='omnipath',interaction_type='post_translational',fixed_genes=set())
    b=normalize_omnipath(Path(dorothea).read_bytes(),dataset='dorothea_abc',interaction_type='transcriptional',fixed_genes=set())
    network=pd.concat([a,b],ignore_index=True)
    assert_unique(network,['edge_id'],'Signed network')
    members.to_parquet(output/'reactome_pathway_memberships.parquet',index=False)
    network.to_parquet(output/'signed_network.parquet',index=False)
    source_manifest(output,'network_manifest.json',
        {'reactome_memberships':'reactome_pathway_memberships.parquet','signed_network':'signed_network.parquet'},
        {'reactome_zip':reactome_zip,'omnipath_tsv':omnipath,'dorothea_tsv':dorothea},
        {'declared_reactome_release':source_release,'release_verification':'Caller declaration; GMT contents contain no independently verified release.',
         'dorothea_filter':'Every row has at least one A/B/C confidence assignment.'})


def convert_entity_tables(entities,aliases,output):
    """Import entity and alias registries from explicitly sourced tables."""
    entities=table(entities);aliases=table(aliases)
    require_columns(entities,['entity_type','canonical_id','entity_text','text_source','ensembl_id','uniprot_accession'],'Entities')
    require_columns(aliases,['entity_type','alias','canonical_id'],'Aliases')
    assert_unique(entities,['entity_type','canonical_id'],'Entities')
    for col in ['ensembl_id','uniprot_accession']:
        # The consumer accepts either a scalar or a JSON-encoded string list.
        for value in entities[col].dropna():
            if isinstance(value,str) and value.lstrip().startswith('['):
                if not isinstance(json.loads(value),list):raise ValueError('Invalid entity identifier list.')
    known=set(zip(entities.entity_type,entities.canonical_id))
    if any(key not in known for key in zip(aliases.entity_type,aliases.canonical_id)):
        raise ValueError('Alias refers to an unknown entity.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    entities.to_parquet(output/'entity_registry.parquet',index=False)
    aliases.drop_duplicates(['entity_type','alias','canonical_id']).to_parquet(output/'alias_registry.parquet',index=False)
    files={'entities':'entity_registry.parquet','aliases':'alias_registry.parquet'}
    save_json(output/'prepare_manifest.json',{'outputs':files,'output_sha256':{k:digest(output/v) for k,v in files.items()}})


def entities_from_public(hgnc,compounds,output):
    """Build identity/alias context from HGNC JSON and a public compound catalogue.

    Descriptions are copied verbatim; no mechanism summaries are generated.
    """
    data=json.loads(Path(hgnc).read_text());genes=data['response']['docs']
    drug=table(compounds)
    if 'name' not in drug and 'canonical_name' in drug:
        drug=drug.rename(columns={'canonical_name':'name'})
    require_columns(drug,['drug','name'],'Public compound catalogue')
    assert_unique(drug,['drug'],'Public compound catalogue')
    rows=[];aliases=[]
    def add_alias(kind,canonical,value):
        if isinstance(value,str) and value.strip():aliases.append({'entity_type':kind,'canonical_id':canonical,'alias':value.strip()})
    for g in genes:
        if g.get('status')!='Approved':continue
        symbol=g['symbol']
        rows.append({'entity_type':'gene','canonical_id':symbol,'entity_text':g.get('name',''),
            'text_source':'HGNC approved gene name','ensembl_id':g.get('ensembl_gene_id',''),
            'uniprot_accession':json.dumps(g.get('uniprot_ids',[]))})
        for value in [symbol,g.get('ensembl_gene_id'),*g.get('alias_symbol',[]),*g.get('prev_symbol',[])]:add_alias('gene',symbol,value)
    for d in drug.to_dict('records'):
        rows.append({'entity_type':'drug','canonical_id':d['drug'],'entity_text':d.get('description',''),
            'text_source':'Public compound catalogue supplied by the reproducer','ensembl_id':'','uniprot_accession':'[]'})
        for value in [d['drug'],d['name']]:add_alias('drug',d['drug'],value)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_parquet(output/'entity_registry.parquet',index=False)
    pd.DataFrame(aliases).drop_duplicates().to_parquet(output/'alias_registry.parquet',index=False)
    files={'entities':'entity_registry.parquet','aliases':'alias_registry.parquet'}
    save_json(output/'prepare_manifest.json',{'outputs':files,'output_sha256':{k:digest(output/v) for k,v in files.items()},
        'source_sha256':{'hgnc':digest(hgnc),'compound_catalogue':digest(compounds)},
        'text_semantics':'Verbatim public identity context, not verified condition mechanisms.'})


BINDING_COLUMNS = ['drug','split','measurement_type','relation','value_nM','reported_value','reported_units',
    'potency_lower_p','potency_upper_p','lower_p_inclusive','upper_p_inclusive','reported_pchembl',
    'activity_id','molecule_chembl_id','assay_chembl_id','target_chembl_id','target_gene_symbol',
    'gene_symbol_resolution','target_uniprot','document_chembl_id','document_year','target_assignment_confidence',
    'relationship_type','bao_format','single_protein_assay_format','assay_cell_type','assay_tissue',
    'assay_description','source_release']


def convert_bindings(mapping,activities,assays,targets,molecules,output):
    """Apply the model's existing strict human single-protein binding curator."""
    from .model.chembl_binding_annotations import curate,threshold_evidence
    from .resources.sciplex_drug_background import same_drug
    mapping=table(mapping)
    require_columns(mapping,['drug','split','molecule_chembl_id','canonical_smiles'],'Binding mapping')
    if not mapping.split.isin(['support','validation','test']).all():raise ValueError('Unknown binding split.')
    assert_unique(mapping,['drug','molecule_chembl_id'],'Binding mapping')
    assay={r['assay_chembl_id']:r for r in records(assays,'assays')}
    target={r['target_chembl_id']:r for r in records(targets,'targets')}
    molecule={r['molecule_chembl_id']:r for r in records(molecules,'molecules')}
    for r in mapping.to_dict('records'):
        if not r['molecule_chembl_id']:
            continue
        remote=molecule.get(r['molecule_chembl_id'],{}).get('molecule_structures') or {}
        if not same_drug(r['canonical_smiles'],remote.get('canonical_smiles')):
            raise ValueError('Binding mapping has missing or mismatched full standard-parent structure.')
    by_molecule={k:g.to_dict('records') for k,g in mapping.groupby('molecule_chembl_id')}
    accepted=[];excluded={}
    for activity in records(activities,'activities'):
        matched=by_molecule.get(activity.get('molecule_chembl_id'),[])
        if not matched:continue
        item,reason=curate(activity,assay.get(activity.get('assay_chembl_id')),target.get(activity.get('target_chembl_id')))
        if reason:
            excluded[reason]=excluded.get(reason,0)+1;continue
        accepted.extend({'drug':m['drug'],'split':m['split'],**item} for m in matched)
    frame=pd.DataFrame(accepted,columns=BINDING_COLUMNS).drop_duplicates(['drug','activity_id'])
    output=Path(output)
    for query,folder,mname,dname in [
        (False,'sciplex_chembl_binding_annotations_v1','binding_manifest.json','drug_bindings.json'),
        (True,'sciplex_binding_query_annotations_v1','query_binding_manifest.json','query_drug_bindings.json')]:
        root=output/folder;root.mkdir(parents=True,exist_ok=True)
        subset=frame[frame.split.eq('test') if query else frame.split.ne('test')]
        selected=mapping[mapping.split.eq('test') if query else mapping.split.ne('test')]
        descriptors=[]
        for drug,group in selected.groupby('drug',sort=True):
            ev=subset[subset.drug.eq(drug)]
            descriptors.append({'drug':drug,'annotation_reference_drugs':sorted(set(group.molecule_chembl_id)-{''}),
                'measured_genes':sorted(set(ev.target_gene_symbol)),
                'active_genes_1uM':sorted({r.target_gene_symbol for r in ev.itertuples() if threshold_evidence(r.value_nM,r.relation,1000)[0]}),
                'binding_records':len(ev)})
        subset.to_parquet(root/'binding_evidence.parquet',index=False)
        save_json(root/dname,descriptors)
        save_json(root/mname,{'output_sha256':{name:digest(root/name) for name in ['binding_evidence.parquet',dname]},
            'excluded_activity_counts':excluded,'scope':'Human binding records matched by compound identity; drugs without accepted assays retain empty evidence lists.'})


def register_functional(ligands,interactions,output,source_release):
    if source_release != '2026.2':
        raise ValueError('This main resource protocol supports Guide to Pharmacology 2026.2 only.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files={'source_ligands':'gtopdb_2026.2_ligands.csv','source_interactions':'gtopdb_2026.2_interactions.csv'}
    for key,src in [('source_ligands',ligands),('source_interactions',interactions)]:
        with Path(src).open() as stream:
            header=stream.readline()
        if not re.search(r'GtoPdb Version:\s*2026\.2(?:\s|[\"-]|$)',header):
            raise ValueError('Guide to Pharmacology version header does not match the declared release.')
        data=pd.read_csv(src,skiprows=1,keep_default_na=False,nrows=1)
        required=['Ligand ID','ChEMBL ID'] if key=='source_ligands' else ['Ligand ID','Target Species','Target Gene Symbol','Target Subunit IDs','Type','Action','Primary Target','PubMed ID','Webpage URLs']
        require_columns(data,required,'Guide to Pharmacology export')
        if Path(src).resolve()!=(output/files[key]).resolve():shutil.copyfile(src,output/files[key])
    source_manifest(output,'functional_manifest.json',files,
        provenance={'declared_release':source_release,'release_verification':'Version header checked in both supplied CSV exports.'})


def register_drugcentral(structures,interactions,output,source_release):
    if source_release != '2021-09-01':
        raise ValueError('This main resource protocol supports DrugCentral 2021-09-01 only.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files={'source_structures':'structures.tsv','source_interactions':'interactions.tsv'}
    for key,src in [('source_structures',structures),('source_interactions',interactions)]:
        data=pd.read_csv(src,sep='\t',keep_default_na=False,nrows=1)
        required=['ID','SMILES','INN'] if key=='source_structures' else ['STRUCT_ID','ORGANISM','MOA','GENE','DRUG_NAME','ACTION_TYPE','TARGET_NAME','MOA_SOURCE_URL','ACT_SOURCE_URL']
        require_columns(data,required,'DrugCentral export')
        if Path(src).resolve()!=(output/files[key]).resolve():shutil.copyfile(src,output/files[key])
    source_manifest(output,'drugcentral_manifest.json',files,
        provenance={'declared_release':source_release,'release_verification':'Caller declaration; TSV contents contain no independently verified release.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    commands={
        'network':['reactome_zip','member','omnipath','dorothea','source_release'],
        'entities':['entities','aliases'],
        'public-entities':['hgnc','compounds'],
        'bindings':['mapping','activities','assays','targets','molecules'],
        'functional':['ligands','interactions','source_release'],
        'drugcentral':['structures','interactions','source_release']}
    for name,keys in commands.items():
        p=sub.add_parser(name)
        for key in keys:p.add_argument('--'+key.replace('_','-'),required=True)
        p.add_argument('--output',required=True)
    args=vars(parser.parse_args());stage=args.pop('stage')
    if any('<' in str(v) or '>' in str(v) for v in args.values()):parser.error('Replace input placeholders before conversion.')
    try:
        {'network':convert_network,'entities':convert_entity_tables,'public-entities':entities_from_public,
         'bindings':convert_bindings,'functional':register_functional,'drugcentral':register_drugcentral}[stage](**args)
    except Exception as exc:
        # External paths and row content must not be copied into a shareable diagnostic.
        print(json.dumps({'status':'failed','stage':stage,'error_type':type(exc).__name__,
            'remedy':'Check the named input flags against docs/INPUT_SCHEMAS.md; no paths or input rows are logged.'}))
        raise SystemExit(2)
    print(json.dumps({'status':'complete','stage':stage}))


if __name__=='__main__':main()
