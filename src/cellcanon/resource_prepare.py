"""Run source preparation modules against explicitly supplied public snapshots."""
import argparse
import importlib
import inspect
from pathlib import Path
from .workflow import configure,configure_knowledge,protocol,write

STAGES={
    'local-audit':('sciplex_knowledge','local_audit'),
    'public-proteins':('sciplex_public_background','uniprot'),
    'drug-background':('sciplex_drug_background','run'),
    'gene-background':('sciplex_gene_background','run'),
    'network-background':('sciplex_network_background','run'),
    'vehicle-context':('vehicle_context','run'),
    'catalogue-identity':('sciplex_catalog_identity','run'),
    'catalogue-annotations':('sciplex_catalog_annotations','run'),
    'ensembl-background':('sciplex_ensembl_background','run'),
    'compile-store':('sciplex_knowledge_store','compile_store'),
    'functional-inputs':('cellcanon.functional_prepare','run'),
    'functional-review':('sciplex_gtopdb_review','run'),
    'build-routes':('sciplex_route_completion','build'),
}


def bind(config,paths):
    root=config['knowledge_root'];inputs=Path(paths['normalized_sources_root'])
    roles={'OUTPUT':root,'SCIPLEX_ROOT':config['data_root'],'METADATA_ROOT':config['metadata_root'],
        'ENTITY_SOURCE_ROOT':Path(paths['entity_source_root']),'NETWORK_SOURCE_ROOT':Path(paths['network_source_root']),
        'FUNCTIONAL_SOURCE_ROOT':Path(paths['functional_source_root']),'DRUGCENTRAL_ROOT':Path(paths['drugcentral_source_root']),
        'BIO':inputs,'LEAF':inputs,
        'PARENT':root/'functional_audit','DESTINATION':root/'functional_targets'}
    if any('<' in str(p) for p in roles.values()):raise ValueError('Replace every resource path placeholder.')
    modules=[importlib.import_module('cellcanon.resources.'+p.stem) for p in Path(__file__).with_name('resources').glob('*.py') if p.stem!='__init__']
    modules.append(importlib.import_module('cellcanon.functional_prepare'))
    for module in modules:
        replacements={}
        for name,value in roles.items():
            if hasattr(module,name):
                old=getattr(module,name)
                if isinstance(old,Path):replacements[old]=value
                setattr(module,name,value)
        for value in list(vars(module).values()):
            if inspect.isfunction(value) and value.__module__==module.__name__ and value.__defaults__:
                value.__defaults__=tuple(replacements.get(d,d) if isinstance(d,Path) else d for d in value.__defaults__)
    configure_knowledge(config)
    write(inputs/'drug_identity_isolation_v1/sciplex3_quarantine.json',{'kept_drugs':{'support':protocol()['train_drugs'],'validation':protocol()['validation_drugs']}})
    return roles


def main():
    import json
    from .resource_preflight import load_paths,check_inputs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=list(STAGES));parser.add_argument('--config',required=True);parser.add_argument('--paths',required=True)
    parser.add_argument('--check-only',action='store_true',help='Inspect required inputs offline without creating files.')
    parser.add_argument('--hashes',action='store_true',help='Also check available manifest output digests.')
    args=parser.parse_args()
    try:
        paths=load_paths(args.config,args.paths)
        report=check_inputs(paths,args.stage,args.hashes)
        if args.check_only or report['status']!='ready':
            print(json.dumps(report,indent=2))
            raise SystemExit(0 if report['status']=='ready' else 2)
        config=configure(args.config)
        bind(config,paths)
        module,name=STAGES[args.stage]
        qualified=module if module.startswith('cellcanon.') else 'cellcanon.resources.'+module
        getattr(importlib.import_module(qualified),name)()
    except Exception as exc:
        print(json.dumps({'status':'failed','stage':args.stage,'error_type':type(exc).__name__,
            'remedy':'Check source schemas, stage order and hashes in docs/INPUT_SCHEMAS.md.'}))
        raise SystemExit(2)


if __name__=='__main__':main()
