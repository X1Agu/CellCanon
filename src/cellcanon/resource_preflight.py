"""Offline input checks. Reports use logical resource names, never local paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

HGNC_CACHE=hashlib.sha256(b'https://rest.genenames.org/fetch/status/Approved').hexdigest()
ENTITY_COLUMNS=['entity_type','canonical_id','entity_text','text_source','ensembl_id','uniprot_accession']
NETWORK_COLUMNS=['edge_id','source_gene_symbol','target_gene_symbol','source_uniprot','target_uniprot',
    'consensus_direction','sign','reference_count','interaction_type']
BINDING_COLUMNS=['drug','activity_id','target_gene_symbol','value_nM','relation','measurement_type',
    'molecule_chembl_id','assay_chembl_id','target_chembl_id']
STAGE_DEPENDENCIES={
    'local-audit':['metadata','normalized','dataset-manifest'],
    'public-proteins':[],
    'drug-background':['local-audit'],
    'gene-background':['local-audit','public-proteins','entity-sources'],
    'network-background':['local-audit','public-proteins','network-sources','drugcentral-sources'],
    'vehicle-context':['preprocessed'],
    'catalogue-identity':['drug-background'],
    'catalogue-annotations':['catalogue-identity'],
    'ensembl-background':['gene-background'],
    'compile-store':['local-audit','public-proteins','drug-background','gene-background','network-background',
        'vehicle-context','catalogue-identity','catalogue-annotations','ensembl-background','supplier-overrides','entity-sources'],
    'functional-inputs':['compile-store','network-background','gene-background','vehicle-context','functional-sources','hgnc-cache'],
    'functional-review':['functional-inputs','functional-sources','hgnc-cache'],
    'build-routes':['compile-store','network-background','vehicle-context','functional-inputs','functional-review','hgnc-cache'],
    'prepared':['compile-store','build-routes'],
}
STAGE_DIRECTORIES={'local-audit':'local_audit','public-proteins':'public_background','drug-background':'drug_background',
    'gene-background':'gene_background','network-background':'network_background','vehicle-context':'cell_baseline',
    'catalogue-identity':'catalog_identity','catalogue-annotations':'catalog_annotations','ensembl-background':'ensembl_background',
    'compile-store':'store','functional-inputs':'functional_audit','functional-review':'functional_targets/gtopdb_review',
    'build-routes':'functional_targets/routes'}
STAGE_OUTPUTS={
    'local-audit':{'drug_coverage.parquet':['drug','split','canonical_name','legacy_drug','canonical_smiles','chembl_ids','target_roles','drug_moa'],
        'gene_coverage.parquet':['gene','gene_symbol','role'],'binding_evidence.parquet':BINDING_COLUMNS},
    'public-proteins':{'uniprot_human_reviewed.parquet':['Entry','Gene Names (primary)','GeneID','Organism (ID)','Function [CC]',
        'Gene Ontology (biological process)','Gene Ontology (molecular function)','Gene Ontology (cellular component)']},
    'drug-background':{'chembl_identity.parquet':['drug','molecule_chembl_id'],'chembl_mechanisms.json':None,
        'chembl_mechanism_targets.json':None,'chembl_molecules.json':None,'name_candidate_audit.json':None,
        'pubchem_drug_background.json':None,'binding_evidence.parquet':BINDING_COLUMNS},
    'gene-background':{'gene_cards.json':None},
    'network-background':{'reactome_memberships.parquet':['pathway_id','canonical_symbol'],
        'reactome_ensembl_memberships.parquet':['ensembl_id','pathway_id','source_url','source_snapshot'],
        'regulatory_network.parquet':NETWORK_COLUMNS+['source_node_id','target_node_id','typed_directed_usable'],
        'complex_components.parquet':['complex_id','canonical_symbol'],
        'go_memberships.parquet':['canonical_symbol','go_id','term_name','aspect'],
        'drugcentral_moa.parquet':['drug','ORGANISM','canonical_gene_symbol','ACTION_TYPE','STRUCT_ID']},
    'vehicle-context':{'all_human_control_baseline.parquet':['cell_line','ensembl_id','gene_symbol','control_detection_fraction']},
    'catalogue-identity':{'resolution.json':None,'molecule_records.json':None},
    'catalogue-annotations':{'mechanisms.json':None,'targets.json':None,'binding_evidence.parquet':BINDING_COLUMNS,
        'pubchem_drug_background.json':None},
    'ensembl-background':{'gene_metadata.json':None},
    'compile-store':{'knowledge.sqlite':None,'drug_cards.json':None},
    'functional-inputs':{'seed_evidence.json':None,'gtopdb_candidate_inventory.json':None,'route_witnesses.jsonl':None},
    'functional-review':{'accepted_seeds.json':None,'review_records.json':None},
    'build-routes':{'routes.sqlite':None,'seed_evidence.json':None},
}


def load_paths(config_file,resource_file=None):
    """Read only the path settings; no model imports, writes, downloads or API calls."""
    def load(filename):
        path=Path(filename);data=json.loads(path.read_text())
        result={}
        for key,value in data.items():
            if key.endswith('_root') or key=='compound_registry':
                if not isinstance(value,str):continue
                p=Path(value);result[key]=p if p.is_absolute() or '<' in value else path.parent/p
        return result
    paths=load(config_file)
    if resource_file:paths.update(load(resource_file))
    return paths


class Checker:
    def __init__(self,paths,verify_hashes=False):
        self.paths=paths;self.verify_hashes=verify_hashes;self.issues=[];self.checked=set()

    def issue(self,resource,code,**detail):
        self.issues.append({'resource':resource,'code':code,**detail})

    def location(self,role,relative=''):
        key=role+('/'+relative if relative else '')
        value=self.paths.get(role)
        if value is None or '<' in str(value) or '>' in str(value):
            if (role,'unconfigured') not in self.checked:
                self.issue(role,'unconfigured');self.checked.add((role,'unconfigured'))
            return key,None
        return key,Path(value)/relative

    def file(self,role,relative='',columns=None,json_keys=None,sep=',',skiprows=0):
        key,path=self.location(role,relative)
        if path is None:return None
        cache=(key,tuple(columns or []),tuple(json_keys or []))
        if cache in self.checked:return path if path.is_file() else None
        self.checked.add(cache)
        if not path.is_file():self.issue(key,'missing_file');return None
        try:
            if columns:
                if path.suffix=='.parquet':
                    import pyarrow.parquet as pq
                    available=pq.read_schema(path).names
                else:
                    import pandas as pd
                    available=pd.read_csv(path,sep=sep,nrows=0,skiprows=skiprows).columns
                missing=sorted(set(columns)-set(available))
                if missing:self.issue(key,'missing_columns',fields=missing)
            if json_keys:
                data=json.loads(path.read_text())
                missing=[k for k in json_keys if k not in data]
                if missing:self.issue(key,'missing_json_keys',fields=missing)
        except Exception as exc:
            self.issue(key,'unreadable_or_invalid',error_type=type(exc).__name__)
        return path

    def json_records(self,role,relative,fields,container=None):
        path=self.file(role,relative)
        if path is None:return
        label=role+'/'+relative
        try:
            value=json.loads(path.read_text())
            if container is not None:value=value[container]
            if not isinstance(value,list):raise TypeError()
            missing=set()
            for row in value:
                if not isinstance(row,dict):raise TypeError()
                missing.update(set(fields)-set(row))
            if missing:self.issue(label,'missing_record_fields',fields=sorted(missing))
        except (ValueError,KeyError,TypeError) as exc:self.issue(label,'invalid_json_records',error_type=type(exc).__name__)

    def manifest(self,role,relative,required=(),mapping=False):
        path=self.file(role,relative,json_keys=['outputs','output_sha256'] if mapping else ['outputs'])
        if path is None:return
        label=role+'/'+relative
        try:
            data=json.loads(path.read_text());outputs=data['outputs']
            entries={k:{'relative_path':v,'sha256':data['output_sha256'][k]} for k,v in outputs.items()} if mapping else {r['name']:r for r in outputs}
            for name in required:
                if name not in entries:self.issue(label,'missing_output_record',fields=[name])
            for name,row in entries.items():
                if required and name not in required:continue
                # Original manifests may contain provenance paths; resolve only their basename.
                filename=Path(row['relative_path']).name
                part=str(Path(relative).parent/filename)
                target=self.file(role,part)
                if target is not None and self.verify_hashes:
                    self.check_hash(target,row['sha256'],role+'/'+part)
        except (ValueError,KeyError,TypeError) as exc:self.issue(label,'invalid_manifest',error_type=type(exc).__name__)

    def check_hash(self,path,expected,label):
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected:self.issue(label,'sha256_mismatch')

    def output_stage(self,stage):
        directory=STAGE_DIRECTORIES[stage]
        mname='uniprot_manifest.json' if stage=='public-proteins' else 'manifest.json'
        mp=self.file('knowledge_root',directory+'/'+mname,json_keys=['output_sha256'])
        for filename,columns in STAGE_OUTPUTS[stage].items():self.file('knowledge_root',directory+'/'+filename,columns)
        if mp is not None and self.verify_hashes:
            try:
                data=json.loads(mp.read_text());outputs=data['output_sha256']
                if isinstance(outputs,str):outputs={'uniprot_human_reviewed.parquet':outputs}
                for filename,expected in outputs.items():
                    path=self.file('knowledge_root',directory+'/'+Path(filename).name)
                    if path:self.check_hash(path,expected,'knowledge_root/'+directory+'/'+Path(filename).name)
            except (ValueError,KeyError,TypeError) as exc:self.issue('knowledge_root/'+directory+'/'+mname,'invalid_manifest',error_type=type(exc).__name__)
        if stage=='functional-inputs':
            self.json_records('knowledge_root',directory+'/seed_evidence.json',['drug','symbol','node','tier','evidence_id','source_url','action_sign','identity_tier'])
            self.json_records('knowledge_root',directory+'/gtopdb_candidate_inventory.json',['drug','split','candidate_ligand_ids','identity_tier','exact_unique'],'mapping')
        if stage=='compile-store':self.sqlite('store/knowledge.sqlite',{'drugs':['drug','payload'],'genes':['gene','payload'],
            'documents':['evidence_id','source_url','payload'],'binding_evidence':['drug','supports_active_1uM','identity_tier'],
            'regulatory_network':NETWORK_COLUMNS+['source_node_id','target_node_id','typed_directed_usable'],
            'cell_baseline':['cell_line','gene_symbol','control_detection_fraction'],'complex_components':['complex_id','canonical_symbol'],
            'drugcentral_moa':['drug','ORGANISM','canonical_gene_symbol','ACTION_TYPE']})

    def sqlite(self,relative,tables):
        path=self.file('knowledge_root',relative)
        if not path:return
        label='knowledge_root/'+relative
        try:
            with sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True) as con:
                for name,columns in tables.items():
                    available={r[1] for r in con.execute('PRAGMA table_info("'+name+'")')}
                    missing=sorted(set(columns)-available)
                    if missing:self.issue(label,'missing_sqlite_columns',table=name,fields=missing)
        except sqlite3.Error as exc:self.issue(label,'invalid_sqlite',error_type=type(exc).__name__)

    def requirement(self,name):
        if name in STAGE_DIRECTORIES:self.output_stage(name);return
        if name=='dataset-manifest':
            self.file('data_root','build_manifest.json',json_keys=['inputs','audits','outputs'])
        elif name=='metadata':
            self.manifest('metadata_root','prepare_manifest.json',['drug_metadata','gene_order','extra_gene_order'],mapping=True)
        elif name=='preprocessed':
            self.file('data_root','build_manifest.json',json_keys=['inputs','audits','outputs'])
            self.file('data_root','metadata/valid_cells_pdata.parquet',['source_cell_index','vehicle','treatment','time_point','cell_type','replicate'])
            self.file('data_root','metadata/human_features.parquet',['human_gene_index','ensembl_id','gene_short_name'])
            for name in ['metadata/all_features.parquet','metadata/feature_to_human.i32','intermediate/library_size.u64']:
                self.file('data_root',name)
        elif name=='entity-sources':
            self.manifest('entity_source_root','prepare_manifest.json',['entities'],mapping=True)
            self.file('entity_source_root','entity_registry.parquet',ENTITY_COLUMNS)
        elif name=='normalized':
            self.manifest('entity_source_root','prepare_manifest.json',['entities','aliases'],mapping=True)
            self.file('entity_source_root','entity_registry.parquet',ENTITY_COLUMNS)
            self.file('entity_source_root','alias_registry.parquet',['entity_type','alias','canonical_id'])
            self.requirement('network-sources')
            for folder,manifest,descriptor in [
                ('sciplex_chembl_binding_annotations_v1','binding_manifest.json','drug_bindings.json'),
                ('sciplex_binding_query_annotations_v1','query_binding_manifest.json','query_drug_bindings.json')]:
                self.file('normalized_sources_root',folder+'/'+manifest,json_keys=['output_sha256'])
                self.file('normalized_sources_root',folder+'/binding_evidence.parquet',BINDING_COLUMNS)
                self.json_records('normalized_sources_root',folder+'/'+descriptor,['drug','annotation_reference_drugs','measured_genes','active_genes_1uM','binding_records'])
        elif name=='network-sources':
            self.manifest('network_source_root','network_manifest.json',['reactome_memberships','signed_network'])
            self.file('network_source_root','reactome_pathway_memberships.parquet',['pathway_id','pathway_name','gene_symbol','species'])
            self.file('network_source_root','signed_network.parquet',NETWORK_COLUMNS)
        elif name=='drugcentral-sources':
            self.manifest('drugcentral_source_root','drugcentral_manifest.json',['source_structures','source_interactions'])
        elif name=='functional-sources':
            self.manifest('functional_source_root','functional_manifest.json',['source_ligands','source_interactions'])
            self.file('functional_source_root','gtopdb_2026.2_ligands.csv',['Ligand ID','ChEMBL ID'],skiprows=1)
            self.file('functional_source_root','gtopdb_2026.2_interactions.csv',['Ligand ID','Target Species','Target Gene Symbol',
                'Target Subunit IDs','Type','Action','Primary Target','PubMed ID','Webpage URLs'],skiprows=1)
        elif name=='supplier-overrides':
            self.json_records('normalized_sources_root','configs/metadata/sciplex_background_source_overrides.json',
                ['drug','catalog_number','source_smiles','statement','source_url','reported_target_symbol'],'records')
        elif name=='hgnc-cache':
            self.file('knowledge_root','public_background/http/'+HGNC_CACHE+'.json',json_keys=['response'])
            self.file('knowledge_root','public_background/http/'+HGNC_CACHE+'.receipt.json',json_keys=['sha256'])


def check_inputs(paths,stage='prepared',verify_hashes=False):
    check=Checker(paths,verify_hashes)
    for key in ('data_root','metadata_root','run_root','knowledge_root','embedding_model_root'):
        check.location(key)
    if stage in STAGE_DIRECTORIES or stage=='all':
        for key in ('normalized_sources_root','entity_source_root','network_source_root','functional_source_root','drugcentral_source_root'):
            check.location(key)
    if stage=='all':
        for requirement in ['metadata','normalized','preprocessed','functional-sources','drugcentral-sources','supplier-overrides']:
            check.requirement(requirement)
        for requirement in STAGE_DIRECTORIES:check.requirement(requirement)
    elif stage=='metadata':
        check.file('data_root','metadata/drug_split_v1.parquet',['drug','split'])
        check.file('data_root','metadata/truth_union_genes.parquet',['ensembl_id','gene_short_name'])
        registry=paths.get('compound_registry')
        if registry is not None:
            check.file('compound_registry',columns=['drug','smiles','target','pathway'])
        else:check.issue('compound_registry','unconfigured')
    else:
        for requirement in STAGE_DEPENDENCIES[stage]:check.requirement(requirement)
        if stage=='gene-background':
            data=paths.get('data_root')
            if data is not None and '<' not in str(data):
                check.paths['reference_root']=Path(data).parent/'reference/ncbi_gene'
                check.file('reference_root','Homo_sapiens.gene_info.gz',['#tax_id','GeneID','Symbol','Synonyms','dbXrefs','description'],sep='\t')
    return {'status':'ready' if not check.issues else 'not_ready','stage':stage,
        'hashes_checked':bool(verify_hashes),'issues':check.issues,
        'scope':'Checks required local files, schemas and optional checksums.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--paths')
    parser.add_argument('--stage',choices=['all','metadata',*STAGE_DEPENDENCIES],default='prepared')
    parser.add_argument('--hashes',action='store_true')
    args=parser.parse_args()
    try:result=check_inputs(load_paths(args.config,args.paths),args.stage,args.hashes)
    except Exception as exc:
        result={'status':'not_ready','stage':args.stage,'issues':[{'resource':'configuration','code':'unreadable_or_invalid','error_type':type(exc).__name__}]}
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='ready' else 2)


if __name__=='__main__':main()
