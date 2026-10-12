"""Prepare cited target seeds and unsigned routes from the compiled main store.

Run this stage before functional-review. It does not load
perturbation responses, predict expression, or call a model service.
"""
from collections import defaultdict,deque
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
from .resources.sciplex_knowledge import OUTPUT,sha256,write_json
from .resources.sciplex_layer5_coverage import FUNCTIONAL_SOURCE_ROOT,validate_witness
from .resources.sciplex_route_completion import Context
from .resources.identifiers import _gene_symbol

PARENT=OUTPUT/'functional_audit'
ACTION_SIGN={'INHIBITOR':-1,'ANTAGONIST':-1,'INVERSE AGONIST':-1,
    'NEGATIVE ALLOSTERIC MODULATOR':-1,'ACTIVATOR':1,'AGONIST':1,'POSITIVE ALLOSTERIC MODULATOR':1}


def shortest_witnesses(start,adjacency,*,maximum_edges=3):
    """Find unsigned, cycle-free shortest witnesses."""
    found={};queue=deque([(start,1,(start,),())]);best={(start,1):0}
    while queue:
        node,sign,nodes,edges=queue.popleft()
        if len(edges)>=maximum_edges:continue
        for e in adjacency.get(node,[]):
            other=e['target_node_id']
            if other in nodes:continue
            ns=(*nodes,other);es=(*edges,e['edge_id']);depth=len(es)
            found.setdefault((other,1,e['interaction_type']=='transcriptional'),{'nodes':ns,'edge_ids':es,'sign':None})
            state=(other,1)
            if depth<maximum_edges and depth<best.get(state,maximum_edges+1):
                best[state]=depth;queue.append((other,1,ns,es))
    return found


def make_seeds(cards,binding,drugcentral,documents,resolve):
    seeds={};excluded=[]
    def seed(drug,symbol,tier,evidence,url,action=None,identity=None):
        symbol=resolve(symbol)
        if not symbol:return
        seeds[drug,symbol,tier,evidence]={'drug':drug,'symbol':symbol,'node':'GENE:'+symbol,
            'tier':tier,'evidence_id':evidence,'source_url':url,'action':action,
            'action_sign':ACTION_SIGN.get(str(action).upper()),'identity_tier':identity}
    for r in binding.loc[binding.supports_active_1uM.eq(1)].to_dict('records'):
        seed(r['drug'],r['target_gene_symbol'],'active_assay',r['evidence_id'],
            'https://www.ebi.ac.uk/chembl/api/data/activity/'+str(r['activity_id'])+'.json',identity=r['identity_tier'])
    for d in cards:
        for m in d['mechanisms']:
            if m['organism']!='Homo sapiens' or m['direct_interaction']!=1:continue
            if m['target_type']!='SINGLE PROTEIN' or len(m['target_components'])!=1:
                excluded.append({'drug':d['drug'],'target_type':m['target_type'],'evidence_id':m['evidence_id'],
                    'reason':'non_single_protein_target_not_flattened'});continue
            symbol=_gene_symbol(m['target_components'][0])[0]
            if symbol and m['evidence_id']:
                seed(d['drug'],symbol,'curated_moa',m['evidence_id'],documents[m['evidence_id']]['source_url'],m['action_type'],m['identity_tier'])
    for i,r in enumerate(drugcentral.to_dict('records')):
        if r['ORGANISM']!='Homo sapiens' or not isinstance(r['canonical_gene_symbol'],str):continue
        seed(r['drug'],r['canonical_gene_symbol'],'curated_moa','DrugCentral:row:'+str(i),
            'https://drugcentral.org/drugcard/'+str(r['STRUCT_ID']),r['ACTION_TYPE'],'strict_native_full_standard_parent_inchi')
    return list(seeds.values()),excluded


def ligand_mapping(cards,ligands):
    ids=defaultdict(set)
    for row in ligands.to_dict('records'):
        for mid in str(row['ChEMBL ID']).replace(';','|').split('|'):
            if mid.strip():ids[mid.strip()].add(row['Ligand ID'])
    return [{'drug':card['drug'],'split':card['split'],
        'candidate_ligand_ids':sorted(set().union(*(ids[c] for c in card['chembl_ids']))),
        'identity_tier':card['chembl_identity_tier'],
        'exact_unique':len(set().union(*(ids[c] for c in card['chembl_ids'])))==1} for card in cards]


def unsigned_witnesses(ctx,seeds):
    by_drug=defaultdict(list)
    for seed in seeds:by_drug[seed['drug']].append(seed)
    paths={}
    for drug in (d['drug'] for d in ctx.cards):
        chosen={}
        for seed in sorted(by_drug[drug],key=lambda s:(s['tier']!='curated_moa',s['action_sign'] is None,s['symbol'],s['evidence_id'])):
            chosen.setdefault(seed['symbol'],seed)
        tiers={key:{} for key in ('direct_target','directed','shared_pathway','shared_pathway_max200')}
        def add(tier,gene,seed,nodes,edges=(),references=()):
            tiers[tier].setdefault(gene,{'drug':drug,'gene':gene,'tier':tier,'seed_evidence_id':seed['evidence_id'],
                'seed_tier':seed['tier'],'seed_source_url':seed['source_url'],'identity_tier':seed['identity_tier'],
                'nodes':list(nodes),'network_edge_ids':list(edges),'pathway_references':list(references),
                'net_sign':None,'condition_effect_status':'not_evaluated'})
        for symbol,seed in chosen.items():
            for gene in sorted(ctx.queries.get(symbol,[])):add('direct_target',gene,seed,[seed['node']])
            if symbol not in paths:paths[symbol]=shortest_witnesses('GENE:'+symbol,ctx.adjacency)
            for (node,_,_),path in paths[symbol].items():
                if node.startswith('GENE:'):
                    for gene in sorted(ctx.queries.get(node[5:],[])):add('directed',gene,seed,path['nodes'],path['edge_ids'])
            for pid,ref in sorted(ctx.bysymbol.get(symbol,{}).items(),key=lambda item:(len(ctx.pathsize[item[0]]),item[0])):
                for gene,other in ctx.bypath.get(pid,{}).items():
                    nodes=[seed['node'],pid,ctx.symbols.get(gene,gene)]
                    add('shared_pathway',gene,seed,nodes,references=[ref,other])
                    if len(ctx.pathsize[pid])<=200:add('shared_pathway_max200',gene,seed,nodes,references=[ref,other])
        for rows in tiers.values():yield from rows.values()


def run(resources=OUTPUT,output=PARENT):
    resources=Path(resources);output=Path(output)
    if (output/'manifest.json').exists():raise ValueError('Functional input manifest already exists; choose a fresh knowledge root.')
    inputs={}
    def verify(path,expected=None):
        actual=sha256(path)
        if expected is not None and actual!=expected:raise ValueError('Resource digest mismatch.')
        inputs[str(Path(path).resolve())]=actual
    for stage in ('store','network_background','gene_background','cell_baseline'):
        mp=resources/stage/'manifest.json';manifest=json.loads(mp.read_text());verify(mp)
        for name,expected in manifest['output_sha256'].items():verify(resources/stage/name,expected)
    mp=FUNCTIONAL_SOURCE_ROOT/'functional_manifest.json';functional=json.loads(mp.read_text());verify(mp)
    for item in functional['outputs']:
        if item['name'].startswith('source_'):verify(FUNCTIONAL_SOURCE_ROOT/Path(item['relative_path']).name,item['sha256'])
    hp=resources/'public_background/http'/(hashlib.sha256(b'https://rest.genenames.org/fetch/status/Approved').hexdigest()+'.json')
    verify(hp,json.loads(hp.with_suffix('.receipt.json').read_text())['sha256'])
    hgnc=json.loads(hp.read_text())['response']['docs'];primary={h['symbol'] for h in hgnc};aliases=defaultdict(set)
    for h in hgnc:
        for name in [h['symbol'],*h.get('alias_symbol',[]),*h.get('prev_symbol',[])]:aliases[name].add(h['symbol'])
    def resolve(symbol):
        if symbol in primary:return symbol
        values=aliases.get(symbol,set());return next(iter(values)) if len(values)==1 else None
    con=sqlite3.connect('file:'+str(resources/'store/knowledge.sqlite')+'?mode=ro',uri=True)
    cards=[json.loads(row[0]) for row in con.execute('SELECT payload FROM drugs ORDER BY drug')]
    binding=pd.read_sql_query('SELECT * FROM binding_evidence',con);dc=pd.read_sql_query('SELECT * FROM drugcentral_moa',con)
    documents={r[0]:{'source_url':r[1]} for r in con.execute('SELECT evidence_id,source_url FROM documents')};con.close()
    seeds,excluded=make_seeds(cards,binding,dc,documents,resolve)
    ligands=pd.read_csv(FUNCTIONAL_SOURCE_ROOT/'gtopdb_2026.2_ligands.csv',skiprows=1,keep_default_na=False)
    mapping=ligand_mapping(cards,ligands)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'seed_evidence.json',seeds);write_json(output/'excluded_nonprotein_targets.json',excluded)
    write_json(output/'gtopdb_candidate_inventory.json',{'mapping':mapping,
        'mapping_rule':'Unique exact ChEMBL ID; no name matching or structure fallback.'})
    ctx=Context();lookup={(s['drug'],s['evidence_id']):s for s in seeds};count=0
    with (output/'route_witnesses.jsonl').open('w') as stream:
        for item in unsigned_witnesses(ctx,seeds):
            validate_witness(item,ctx.edges,lookup,ctx.symbols)
            stream.write(json.dumps(item,ensure_ascii=False)+'\n');count+=1
    write_json(output/'manifest.json',{'status':'FUNCTIONAL_INPUTS_PREPARED','source_sha256':inputs,
        'output_sha256':{p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'},
        'seed_rows':len(seeds),'unsigned_witnesses':count,'model_calls':0,'response_label_reads':0,
        'scope':'Background connectivity and cited target actions for constructing route hypotheses.'})
    print(json.dumps({'status':'functional_inputs_prepared','seed_rows':len(seeds),'unsigned_witnesses':count}))
