"""Offline regressions for input contracts; all in-test records are synthetic."""
import json
import zipfile
from types import SimpleNamespace
import pandas as pd
from cellcanon.input_convert import (convert_network,entities_from_public,convert_entity_tables,
    normalize_omnipath)
from cellcanon.resource_preflight import check_inputs,Checker
from cellcanon.functional_prepare import make_seeds,unsigned_witnesses,ligand_mapping
from cellcanon.resources.sciplex_network_background import parse_go_term


def test_route_store_remains_hash_checked_after_relocation(tmp_path,monkeypatch):
    import sqlite3
    import pytest
    from cellcanon.resources import sciplex_route_knowledge as routes
    from cellcanon.model.data import sha256
    root=tmp_path/'new_location';store=root/'store';store.mkdir(parents=True)
    background=store/'knowledge.sqlite';background.write_bytes(b'synthetic background')
    route_root=root/'functional_targets/routes';route_root.mkdir(parents=True)
    route_file=route_root/'routes.sqlite'
    with sqlite3.connect(route_file) as con:
        con.execute('CREATE TABLE metadata (key TEXT, value TEXT)')
        con.execute('INSERT INTO metadata VALUES (?,?)',('seed_action_conflicts','{}'))
    manifest={'output_sha256':{'routes.sqlite':sha256(route_file)},
              'source_sha256':{'/previous/location/store/knowledge.sqlite':sha256(background)}}
    path=route_root/'manifest.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr(routes,'OUTPUT',root)
    monkeypatch.setattr(routes,'LocalKnowledge',lambda:SimpleNamespace(close=lambda:None))
    reader=routes.RouteKnowledge(root/'functional_targets');reader.close()
    background.write_bytes(b'changed background')
    with pytest.raises(ValueError,match='Background knowledge changed'):
        routes.RouteKnowledge(root/'functional_targets')
    manifest['source_sha256']['/other/location/store/knowledge.sqlite']=sha256(background)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identify one'):
        routes.RouteKnowledge(root/'functional_targets')


def test_uniprot_go_terms_are_parsed():
    # Source format used by UniProt's public TSV export.
    match=parse_go_term('protein phosphorylation [GO:0006468]')
    assert match and match[1]=='protein phosphorylation' and match[2]=='GO:0006468'
    assert parse_go_term('not a GO accession') is None
    assert parse_go_term('bad [GO:abc]') is None


def test_preflight_reports_fields_without_machine_paths(tmp_path):
    root=tmp_path/'private-user-host';root.mkdir()
    pd.DataFrame({'wrong_column':[1]}).to_parquet(root/'alias_registry.parquet')
    check=Checker({'entity_source_root':root})
    check.file('entity_source_root','alias_registry.parquet',['entity_type','alias','canonical_id'])
    check.file('entity_source_root','missing.parquet')
    result=json.dumps(check.issues)
    assert 'private-user-host' not in result and str(tmp_path) not in result
    assert check.issues[0]['fields']==['alias','canonical_id','entity_type']
    assert check.issues[1]['code']=='missing_file'
    assert check_inputs({'knowledge_root':'<KNOWLEDGE_ROOT>'},'prepared')['status']=='not_ready'


def test_manifest_hash_checks_detect_changed_input(tmp_path):
    source=tmp_path/'table.json';source.write_text('{}')
    (tmp_path/'manifest.json').write_text(json.dumps({'outputs':[{'name':'source','relative_path':'table.json','sha256':'0'*64}]}))
    check=Checker({'network_source_root':tmp_path},True)
    check.manifest('network_source_root','manifest.json',['source'])
    assert any(r['code']=='sha256_mismatch' for r in check.issues)


def network_tsv():
    return pd.DataFrame([
        dict(source='P1',target='P2',source_genesymbol='GENE1',target_genesymbol='GENE2',
             is_directed=True,is_stimulation=True,is_inhibition=False,consensus_direction=True,
             consensus_stimulation=True,consensus_inhibition=False,sources='resource',references='123',curation_effort=1,dorothea_level='A'),
        dict(source='P2',target='P3',source_genesymbol='GENE2',target_genesymbol='GENE3',
             is_directed=True,is_stimulation=True,is_inhibition=True,consensus_direction=True,
             consensus_stimulation=True,consensus_inhibition=True,sources='resource',references='456',curation_effort=1,dorothea_level='A'),
    ]).to_csv(sep='\t',index=False).encode()


def test_public_network_conversion_preserves_unknown_signs(tmp_path):
    raw=network_tsv();network=normalize_omnipath(raw,dataset='omnipath',interaction_type='post_translational',fixed_genes=set())
    assert network.iloc[0]['sign']==1
    assert pd.isna(network.iloc[1]['sign'])
    assert network.iloc[1]['sign_status']=='conflict'
    assert not network.iloc[1]['signed_path_eligible']
    archive=tmp_path/'reactome.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('pathways.gmt','pathway\tR-HSA-test\tGENE1\tGENE2\n')
    for name in ['omnipath.tsv','dorothea.tsv']:(tmp_path/name).write_bytes(raw)
    output=tmp_path/'normalized'
    convert_network(archive,'pathways.gmt',tmp_path/'omnipath.tsv',tmp_path/'dorothea.tsv',output,source_release='v97')
    check=Checker({'network_source_root':output},True);check.requirement('network-sources')
    assert check.issues==[]
    combined=pd.read_parquet(output/'signed_network.parquet')
    assert set(combined.interaction_type)=={'post_translational','transcriptional'}
    assert combined.edge_id.is_unique


def test_public_entities_keep_alias_identity_and_verbatim_text(tmp_path):
    hgnc=tmp_path/'hgnc.json';hgnc.write_text(json.dumps({'response':{'docs':[
        {'status':'Approved','symbol':'GENE1','ensembl_gene_id':'ENSG0001','name':'Example gene',
         'alias_symbol':['OLD1'],'uniprot_ids':['P1']}]}}))
    compound=tmp_path/'compounds.csv';pd.DataFrame([{'drug':'D1','name':'Example compound','description':'Supplied text.'}]).to_csv(compound,index=False)
    output=tmp_path/'entities';entities_from_public(hgnc,compound,output)
    entities=pd.read_parquet(output/'entity_registry.parquet');aliases=pd.read_parquet(output/'alias_registry.parquet')
    assert entities.loc[entities.canonical_id.eq('D1'),'entity_text'].iloc[0]=='Supplied text.'
    assert aliases.loc[aliases.alias.eq('OLD1'),'canonical_id'].iloc[0]=='GENE1'
    convert_entity_tables(output/'entity_registry.parquet',output/'alias_registry.parquet',tmp_path/'copied')
    check=Checker({'entity_source_root':output},True)
    check.manifest('entity_source_root','prepare_manifest.json',['entities','aliases'],True)
    assert check.issues==[]


def test_functional_seeds_do_not_infer_binding_direction():
    binding=pd.DataFrame([{'supports_active_1uM':1,'drug':'D1','target_gene_symbol':'G1',
        'evidence_id':'binding:1','activity_id':1,'identity_tier':'strict'}])
    dc=pd.DataFrame(columns=['ORGANISM','canonical_gene_symbol'])
    cards=[{'drug':'D1','mechanisms':[]}]
    seeds,excluded=make_seeds(cards,binding,dc,{},lambda symbol:symbol)
    assert excluded==[] and seeds[0]['action_sign'] is None
    ligand=ligand_mapping([{'drug':'D1','split':'support','chembl_ids':['CHEMBL1'],'chembl_identity_tier':'strict'}],
        pd.DataFrame([{'Ligand ID':1,'ChEMBL ID':'CHEMBL1'},{'Ligand ID':2,'ChEMBL ID':'CHEMBL1'}]))
    assert not ligand[0]['exact_unique'] and ligand[0]['candidate_ligand_ids']==[1,2]


def test_functional_unsigned_routes_preserve_evidence_and_endpoint():
    seed={'drug':'D1','symbol':'G1','node':'GENE:G1','tier':'curated_moa','evidence_id':'source:1',
        'source_url':'https://example.org/evidence','action_sign':-1,'identity_tier':'strict'}
    edge={'edge_id':'e1','source_node_id':'GENE:G1','target_node_id':'GENE:G2','interaction_type':'transcriptional'}
    ctx=SimpleNamespace(cards=[{'drug':'D1'}],queries={'G1':{'ENSG1'},'G2':{'ENSG2'}},
        adjacency={'GENE:G1':[edge]},bysymbol={},bypath={},pathsize={},symbols={'ENSG1':'G1','ENSG2':'G2'})
    rows=list(unsigned_witnesses(ctx,[seed]))
    direct=next(r for r in rows if r['tier']=='direct_target');route=next(r for r in rows if r['tier']=='directed')
    assert direct['network_edge_ids']==[] and route['network_edge_ids']==['e1']
    assert route['gene']=='ENSG2' and route['seed_evidence_id']=='source:1' and route['net_sign'] is None


def test_binding_conversion_preserves_censoring_and_rejects_nonhuman(tmp_path):
    from cellcanon.input_convert import convert_bindings
    mapping=tmp_path/'mapping.csv'
    pd.DataFrame([{'drug':'D1','split':'support','molecule_chembl_id':'CHEMBL1','canonical_smiles':'CC'},
                  {'drug':'D2','split':'test','molecule_chembl_id':'','canonical_smiles':'CCC'}]).to_csv(mapping,index=False)
    assay={'assay_chembl_id':'ASSAY1','target_chembl_id':'TARGET1','assay_type':'B','confidence_score':9,
           'relationship_type':'D','assay_organism':'Homo sapiens','bao_format':'BAO_0000357'}
    target={'target_chembl_id':'TARGET1','organism':'Homo sapiens','tax_id':9606,'target_type':'SINGLE PROTEIN',
            'target_components':[{'accession':'P1','target_component_synonyms':[{'syn_type':'GENE_SYMBOL','component_synonym':'G1'}]}]}
    activity={'activity_id':1,'molecule_chembl_id':'CHEMBL1','assay_chembl_id':'ASSAY1','target_chembl_id':'TARGET1',
        'assay_type':'B','standard_type':'Ki','standard_flag':1,'data_validity_comment':None,'potential_duplicate':0,
        'standard_relation':'<','standard_units':'uM','standard_value':'0.5','target_organism':'Homo sapiens'}
    bad={**activity,'activity_id':2,'target_organism':'Mus musculus'}
    fixtures={'activities':[activity,bad],'assays':[assay],'targets':[target],
        'molecules':[{'molecule_chembl_id':'CHEMBL1','molecule_structures':{'canonical_smiles':'CC'}}]}
    for name,value in fixtures.items():(tmp_path/(name+'.json')).write_text(json.dumps(value))
    out=tmp_path/'normalized'
    convert_bindings(mapping,*[tmp_path/(name+'.json') for name in ['activities','assays','targets','molecules']],out)
    rows=pd.read_parquet(out/'sciplex_chembl_binding_annotations_v1/binding_evidence.parquet')
    assert len(rows)==1 and rows.iloc[0].value_nM==500 and rows.iloc[0].relation=='<'
    missing=json.loads((out/'sciplex_binding_query_annotations_v1/query_drug_bindings.json').read_text())[0]
    assert missing['drug']=='D2' and missing['binding_records']==0 and missing['annotation_reference_drugs']==[]


def test_public_source_registration_checks_real_export_headers(tmp_path):
    from cellcanon.input_convert import register_functional,register_drugcentral
    ligand=tmp_path/'ligands.csv';ligand.write_text('# GtoPdb Version: 2026.2 - published: 2026-06-15\nLigand ID,ChEMBL ID\n1,CHEMBL1\n')
    interaction=tmp_path/'interactions.csv'
    interaction.write_text('# GtoPdb Version: 2026.2 - published: 2026-06-15\nLigand ID,Target Species,Target Gene Symbol,Target Subunit IDs,Type,Action,Primary Target,PubMed ID,Webpage URLs\n')
    register_functional(ligand,interaction,tmp_path/'functional',source_release='2026.2')
    checker=Checker({'functional_source_root':tmp_path/'functional'},True);checker.requirement('functional-sources')
    assert checker.issues==[]
    structure=tmp_path/'structures.tsv';structure.write_text('ID\tSMILES\tINN\n1\tCC\tExample\n')
    moa=tmp_path/'moa.tsv';moa.write_text('STRUCT_ID\tORGANISM\tMOA\tGENE\tDRUG_NAME\tACTION_TYPE\tTARGET_NAME\tMOA_SOURCE_URL\tACT_SOURCE_URL\n')
    register_drugcentral(structure,moa,tmp_path/'drugcentral',source_release='2021-09-01')
    checker=Checker({'drugcentral_source_root':tmp_path/'drugcentral'},True);checker.requirement('drugcentral-sources')
    assert checker.issues==[]


def test_source_release_contracts_and_dorothea_abc_filter(tmp_path):
    import pytest
    from cellcanon.input_convert import register_functional,register_drugcentral
    with pytest.raises(ValueError,match='v97'):
        convert_network('unused','unused','unused','unused',tmp_path/'out',source_release='v98')
    with pytest.raises(ValueError,match='2021-09-01'):
        register_drugcentral('unused','unused',tmp_path/'out',source_release='newer')
    ligand=tmp_path/'wrong-version.csv';ligand.write_text('# GtoPdb Version: 2025.1\nLigand ID,ChEMBL ID\n')
    with pytest.raises(ValueError,match='version header'):
        register_functional(ligand,ligand,tmp_path/'out',source_release='2026.2')
    raw=network_tsv()
    with pytest.raises(ValueError,match='A, B or C'):
        normalize_omnipath(raw.replace(b'\tA\n',b'\tD\n'),dataset='dorothea_abc',interaction_type='transcriptional',fixed_genes=set())
    mixed=normalize_omnipath(raw.replace(b'\tA\n',b'\tA;D\n'),dataset='dorothea_abc',interaction_type='transcriptional',fixed_genes=set())
    assert len(mixed)==2


def test_cli_invalid_inputs_do_not_leak_paths(tmp_path,monkeypatch,capsys):
    import sys
    import pytest
    from cellcanon.input_convert import main as convert_main
    from cellcanon.resource_preflight import main as check_main
    from cellcanon.resource_prepare import main as prepare_main
    bad=tmp_path/'private-user-machine-file';bad.write_text('[]')
    monkeypatch.setattr(sys,'argv',['preflight','--config',str(bad)])
    with pytest.raises(SystemExit) as exc:check_main()
    assert exc.value.code==2
    monkeypatch.setattr(sys,'argv',['prepare','public-proteins','--config',str(bad),'--paths',str(bad),'--check-only'])
    with pytest.raises(SystemExit) as exc:prepare_main()
    assert exc.value.code==2
    monkeypatch.setattr(sys,'argv',['convert','network','--reactome-zip',str(bad),'--member','x',
        '--omnipath',str(bad),'--dorothea',str(bad),'--source-release','v97','--output',str(tmp_path/'out')])
    with pytest.raises(SystemExit) as exc:convert_main()
    assert exc.value.code==2
    captured=capsys.readouterr();output=captured.out+captured.err
    assert 'private-user-machine-file' not in output and str(tmp_path) not in output and 'Traceback' not in output
    assert 'BadZipFile' in output


def test_stage_preflight_requires_configured_roots_and_gene_source():
    report=check_inputs({},'public-proteins')
    missing={r['resource'] for r in report['issues'] if r['code']=='unconfigured'}
    assert {'knowledge_root','embedding_model_root','entity_source_root','functional_source_root'}<=missing
