"""Predict, Explain and Update workflow entry points."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from .model import data, dose_control, unified_atlas_training as training
from .model.condition_llm_packets import write, PacketBuilder
from .model.condition_pilot_features import PilotMap
from .model.isolation import subset_by_drugs


def read(path):
    return json.loads(Path(path).read_text())


def configure(filename):
    filename=Path(filename).resolve(); config=read(filename)
    for key in ('data_root','metadata_root','run_root','knowledge_root','embedding_model_root'):
        value=config[key]
        if '<' in value: raise ValueError('Replace the '+key+' placeholder in your configuration.')
        config[key]=Path(value) if Path(value).is_absolute() else filename.parent/ value
    compound=config.get('compound_registry')
    if compound and '<' not in compound:
        config['compound_registry']=Path(compound) if Path(compound).is_absolute() else filename.parent/compound
    data.SCIPLEX_ROOT=config['data_root']; data.METADATA_ROOT=config['metadata_root']
    data.metadata.__defaults__=(data.METADATA_ROOT,)
    data.load_panel.__kwdefaults__['root']=data.SCIPLEX_ROOT
    dose_control.SCIPLEX_ROOT=data.SCIPLEX_ROOT
    from .model import condition_pilot_features as features
    features.SNAPSHOT=config['embedding_model_root']
    config['run_root'].mkdir(parents=True,exist_ok=True)
    config['knowledge_root'].mkdir(parents=True,exist_ok=True)
    return config


def protocol():
    return read(Path(__file__).with_name('protocols')/'split.json')


def prepare_metadata(config):
    from .math_utils.count_calibration import build_v3_drug_metadata
    root=config['metadata_root'];root.mkdir(parents=True,exist_ok=True)
    compound=Path(config['compound_registry'])
    if '<' in str(compound):raise ValueError('Replace the compound_registry placeholder in your configuration.')
    registry=pd.read_parquet(compound) if compound.suffix=='.parquet' else pd.read_csv(compound)
    split=config['data_root']/'metadata/drug_split_v1.parquet'
    frame=pd.read_parquet(split) if split.suffix=='.parquet' else pd.read_csv(split)
    drugs,_=build_v3_drug_metadata(frame,registry,radius=2,fp_size=2048)
    genes=pd.read_parquet(config['data_root']/'metadata/truth_union_genes.parquet')
    genes=genes.rename(columns={'ensembl_id':'gene','gene_short_name':'gene_symbol'})
    genes['gene_index']=np.arange(len(genes))
    reference=read(Path(__file__).with_name('protocols')/'genes.json')
    if genes.gene.astype(str).tolist()!=reference['genes']:
        raise ValueError('Preprocessed gene order must match src/cellcanon/protocols/genes.json.')
    files={'drug_metadata':drugs,'gene_order':genes.iloc[:2000],'extra_gene_order':genes.iloc[2000:]}
    outputs={}
    for name,frame in files.items():
        file=root/(name+'.parquet');frame.to_parquet(file,index=False);outputs[name]=file.name
    write(root/'prepare_manifest.json',{'outputs':outputs,'output_sha256':{k:data.sha256(root/v) for k,v in outputs.items()}})


def configure_knowledge(config):
    """Bind resource locations before opening the immutable public-source indexes."""
    from .resources import sciplex_knowledge as knowledge, sciplex_knowledge_store as store
    from .resources import sciplex_route_knowledge as routes, sciplex_gtopdb_review as targets
    knowledge.OUTPUT=config['knowledge_root'];store.OUTPUT=config['knowledge_root'];routes.OUTPUT=config['knowledge_root']
    route_root=config['knowledge_root']/'functional_targets'
    targets.DESTINATION=route_root;routes.DESTINATION=route_root
    store.LocalKnowledge.__init__.__defaults__=(config['knowledge_root']/'store/knowledge.sqlite',)
    routes.RouteKnowledge.__init__.__defaults__=(route_root,)


def prepare_knowledge(config):
    configure_knowledge(config)
    from .source_cards import build_cards
    builder=PacketBuilder();p=protocol();root=config['run_root']/'knowledge';root.mkdir(exist_ok=True)
    try:
        packets={d+'_'+c:builder.condition(d,c) for d in p['train_drugs']+p['validation_drugs']+p['test_drugs'] for c in data.CELLS}
        write(root/'packets.json',packets)
        build_cards(builder.store,packets,p,root)
    finally:builder.close()


def client(config,role):
    from .model.decision_api import DecisionClient
    cap=float(config['api_budget'])
    if cap<=0: raise ValueError('Set a positive API budget before requesting language-model inference.')
    rates=config['api_rates_per_million']
    return DecisionClient(config['run_root']/'api'/role,
        endpoint=config['model_service_endpoint'],deployment=config['language_model'],
        credential_env=config['model_service_credential_env'],
        input_rate=float(rates['input']),output_rate=float(rates['output']),
        cap=cap,run_cap=cap,run_id=role,request_options=config.get('model_request_options',{}))


def compile_evidence(config):
    from .model.full_condition_evidence import compile_one
    root=config['run_root']/'knowledge';cards=read(root/'cards.json');api=client(config,'compiler')
    records=[]
    for drug in sorted(cards):
        for cell in data.CELLS:
            records.append(compile_one(api,cards[drug],cell,root/'compiled'))
    write(root/'compile_manifest.json',{'conditions':len(records),'valid':sum(r['valid'] for r in records),
                                      'files_sha256':{p.name:data.sha256(p) for p in (root/'compiled').glob('*.json')}})


def embed(config):
    from .model.condition_pilot_features import LocalEmbedder, normalize_record, target_weights, structure
    from .model.condition_llm_packets import wire
    from .model.condition_pilot_knowledge import packet
    root=config['run_root']/'knowledge';cards=read(root/'cards.json');targets=read(root/'target_texts.json')
    md,_=data.metadata();names={}
    for row in md.to_dict('records'):
        for field in ('drug','canonical_name','legacy_drug'):
            names.setdefault(str(row[field]).casefold(),set()).add(row['drug'])
    aliases={k:next(iter(v)) for k,v in names.items() if len(v)==1}
    inputs={};rows={}
    for drug,card in cards.items():
        bytarget={t['target']:t for t in card['targets']}
        for cell in data.CELLS:
            key=drug+'_'+cell;record=normalize_record(card,cell,read(root/'compiled'/(key+'.json')),aliases)
            selected=record['result']['selected'] if record['valid'] else []
            text=[]
            for sel in selected:
                t=bytarget[sel['target']]
                text.append({'target':sel['target'],'relevance_hypothesis':sel['relevance'],
                    'assay_scope_hypothesis':sel['assay_scope'],'function':t['function'],'control':t['control'][cell],
                    'sources':[packet(card,cell)['evidence'][ref] for ref in sel['evidence']]})
            inputs[key]=wire(text) if selected else 'No valid selected biological evidence.'
            features=structure(card,cell,selected);features['compiler_valid']=float(record['valid'])
            rows[key]={'drug':drug,'cell':cell,'compiled_structure':features,'compiler_valid':record['valid'],
                       'compiled_target_weights':target_weights(card,cell,selected)}
    inputs.update({'target:'+k:v['text'] for k,v in targets.items()})
    vectors=LocalEmbedder(root/'embedding_cache').encode(inputs);dimension=len(next(iter(vectors.values())))
    for key,row in rows.items():
        v=sum((w*vectors['target:'+t] for t,w in row['compiled_target_weights'].items()),np.zeros(dimension))
        row['compiled_target_vector']=(v/max(np.linalg.norm(v),1e-12)).tolist()
        row['compiled_vector']=vectors[key].tolist()
    write(root/'features_complete.json',rows)
    write(root/'features_complete_manifest.json',{'feature_sha256':data.sha256(root/'features_complete.json'),'conditions':len(rows)})


def prepare_model(config):
    p=protocol();root=config['run_root']/'model';out=root/'inputs'
    if (root/'protocol.json').exists():raise FileExistsError('Use a fresh run for model preparation.')
    out.mkdir(parents=True,exist_ok=True)
    train=subset_by_drugs(data.load_panel('support'),p['train_drugs']);valid=subset_by_drugs(data.load_panel('validation'),p['validation_drugs'])
    ids=p['identity_groups']
    pools=[{ids[d] for d in p[k]} for k in ('train_drugs','validation_drugs','test_drugs')]
    if any(pools[i]&pools[j] for i in range(3) for j in range(i)):raise ValueError('Canonical identity overlap.')
    order=np.random.default_rng(20260916).permutation(len(train.drugs));fold=np.empty(len(order),int);fold[order]=np.arange(len(order))%5
    means=train.continuous.mean(0);scale=max(float(np.sqrt(np.square(train.continuous-means).mean())),1.)
    memory=np.concatenate([(train.continuous-means)/scale,train.de,np.where(train.de==1,train.up,0.)],axis=-1).astype('f4')
    np.savez_compressed(out/'atlas.npz',observations=memory,continuous_mean=means,continuous_scale=scale,train_allowed=fold[:,None]!=fold[None,:])
    for name,panel in [('support',train),('validation',valid)]:
        np.savez_compressed(out/(name+'_labels.npz'),**{k:getattr(panel,k) for k in ('continuous','de','up','sampled')})
    meta,_=data.metadata();kernels=data.drug_kernels(meta);index={d:i for i,d in enumerate(meta.drug)}
    drugs=train.drugs+valid.drugs+p['test_drugs'];ti=[index[d] for d in train.drugs];qi=[index[d] for d in drugs]
    similarity=np.stack([kernels[k][np.ix_(qi,ti)] for k in ('chemical','target','moa')],axis=-1).astype('f4')
    np.savez_compressed(out/'similarities.npz',values=similarity)
    rows=read(config['run_root']/'knowledge/features_complete.json');get=lambda ds:[rows[d+'_'+c] for d in ds for c in data.CELLS]
    mapper=PilotMap('compiled_semantic',96)
    z=np.concatenate([mapper.fit(get(train.drugs),[(d,c) for d in train.drugs for c in data.CELLS]),mapper.transform(get(valid.drugs+p['test_drugs']))])
    np.savez_compressed(out/'compiled_semantic.npz',knowledge=z.reshape(len(drugs),3,-1).astype('f4'));joblib.dump(mapper,out/'compiled_semantic.joblib')
    write(out/'manifest.json',{**p,'all_drugs':drugs,'compiled_test_drugs':p['test_drugs'],'genes':train.genes,'symbols':train.symbols,
        'folds':fold.tolist(),'cells':list(data.CELLS),'files_sha256':{f.name:data.sha256(f) for f in out.iterdir() if f.is_file()}})
    # Use the configuration selected on validation data.
    write(root/'protocol.json',{'specs':[{'hidden':64,'weight_decay':.01,'seed':s} for s in (20260916,20260917,20260918)],
        'epochs':25,'checkpoints':[25],'classifier_ridges':[10.,100.],'classifier_epochs':200,
        'code_sha256':{},'views':['compiled_semantic'],'test_truth_used_for_selection':False})


def predict(config):
    training.freeze(config['run_root']/'model')


def explain(config):
    from .model.source_explain_pipeline_v2 import run_complete_source_explanation
    root=config['run_root'];cases=read(root/'explain_inputs/public.json')
    api=client(config,'explain')
    for case in cases:
        run_complete_source_explanation(api,case,root/'explanations'/str(case['id']))


def score_explain(config):
    from .model.explanation_benchmark import claim_metrics,route_metrics
    cases=read(config['run_root']/'explain_inputs/private.json');rows=[];gold=[];predicted=[];routes=[];rankings={}
    for case in cases:
        result=read(config['run_root']/'explanations'/str(case['id'])/'result.json')['value']
        rows.append({'id':case['id'],'claims':claim_metrics(case['claims'],result['claims'])[0]})
        gold.extend(case['claims']);predicted.extend(result['claims'])
        if case['route']:routes.append(case['route']);rankings[case['id']]=result.get('route_ranking',[])
    write(config['run_root']/'explanation_metrics.json',{'conditions':len(cases),'claims':claim_metrics(gold,predicted)[0],
        'routes':route_metrics(routes,rankings),'per_condition':rows})


def prepare_explain(config):
    configure_knowledge(config)
    from .model.explanation_cases import build_case
    from .model.explanation_repair import canonical_case
    from .model.observation_state import digest
    from .resources.sciplex_route_knowledge import RouteKnowledge
    p=protocol();packets=read(config['run_root']/'knowledge/packets.json');store=RouteKnowledge();public=[];private=[]
    drugs=sorted(p['test_drugs'],key=lambda d:digest(['native-explain-20260915','test',d]))[:24]
    try:
        for drug in drugs:
            for cell in data.CELLS:
                case,gold=build_case(packets[drug+'_'+cell],store,'test')
                # Detach nested mappings so route annotation cannot mutate claims.
                detached=json.loads(json.dumps(case,ensure_ascii=False,allow_nan=False))
                public.append(canonical_case(detached));private.append(gold)
    finally:store.close()
    write(config['run_root']/'explain_inputs/public.json',public)
    write(config['run_root']/'explain_inputs/private.json',private)
