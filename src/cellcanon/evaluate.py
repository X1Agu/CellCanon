"""Evaluate saved predictions against test responses."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,auc,f1_score
from .workflow import read,write,protocol
from .model.data import load_panel,sha256,CELLS
from .model.isolation import subset_by_drugs
from .model.full_cell_math import continuous_rows,binary_scores
from .model.dose_control_math import lfc_to_mean


def classification(truth,up,sampled,prob,hard,*,direction_score,direction_hard):
    """Keep the DIR ranking score separate from its caller-defined hard decision.

    Generated expression uses signed logFC and logFC > 0. Condition heads use
    the up probability and p_up >= 0.5. Joint classes use only the hard decision.
    """
    direction_score=np.asarray(direction_score)
    direction_hard=np.asarray(direction_hard)
    if direction_score.shape!=np.shape(truth) or direction_hard.shape!=np.shape(truth):
        raise ValueError('DIR scores and hard decisions must align with truth.')
    if direction_hard.dtype!=np.bool_:
        raise ValueError('DIR hard decisions must be explicit boolean labels.')
    result=[]
    for role,sl in [('core2000',slice(0,2000)),('extra931',slice(2000,2931)),('all2931',slice(None))]:
        for scope in ('official_sampled','dense'):
            use=sampled[...,sl] if scope=='official_sampled' else np.ones(truth[...,sl].shape,bool)
            for task in ('DE','DIR_given_true_DE'):
                mask=use if task=='DE' else use&(truth[...,sl]==1)
                y=(truth if task=='DE' else up)[...,sl][mask]
                s=(prob if task=='DE' else direction_score)[...,sl][mask]
                h=hard[...,sl][mask] if task=='DE' else direction_hard[...,sl][mask]
                score=binary_scores(y,s,h)
                if len(y):
                    precision,recall,_=precision_recall_curve(y,s);score['auprc']=float(auc(recall,precision))
                else:score['auprc']=None
                result.append({'role':role,'scope':scope,'task':task,**score})
            true=np.where(truth[...,sl],np.where(up[...,sl]==1,1,2),0)
            predicted=np.where(hard[...,sl],np.where(direction_hard[...,sl],1,2),0)
            result.append({'role':role,'scope':scope,'task':'joint_three_state','n':int(use.sum()),
                           'macro_f1':float(f1_score(true[use],predicted[use],labels=[0,1,2],average='macro',zero_division=0))})
    return result


def cells(config):
    root=config['run_root'];out=root/'cells_test';frozen=read(out/'manifest.json')
    for file,h in frozen['files_sha256'].items():
        if sha256(out/file)!=h:raise ValueError('Population output checksum does not match cells_test/manifest.json.')
    panel=subset_by_drugs(load_panel('test',allow_test=True),protocol()['test_drugs']);pred=dict(np.load(out/'population_readouts.npz'))
    classes=classification(panel.de,panel.up,panel.sampled,pred['de_score'],pred['de'],
        direction_score=pred['lfc'],direction_hard=pred['lfc']>0)
    scores={k:float(v.mean()) for k,v in continuous_rows(panel.continuous,pred['lfc'][...,:2000]).items()}
    details=[]
    for ci,cell in enumerate(CELLS):
        ctx=np.load(root/'cell_model'/(cell+'.npz'));control=ctx['mean'][:2000]
        ref=lfc_to_mean(panel.continuous[:,ci],control);generated=pred['mean'][:,ci,:2000]
        stats=pd.read_parquet(config['data_root']/'splits/drug_zero_shot_v1/locked_test/truth_full_stats'/(cell+'_HVG2000_plus_ExtraLM_stats.parquet'))
        rank=stats.pivot(index='pert',columns='ensembl_id',values='scores').reindex(index=panel.drugs,columns=panel.genes[:2000]).to_numpy()
        for di,drug in enumerate(panel.drugs):
            for size in (20,50,100,2000):
                ix=np.argsort(-abs(rank[di]),kind='stable')[:size]
                y=ref[di,ix];v=generated[di,ix];den=np.square(y-y.mean()).sum()
                delta=continuous_rows(y-control[ix],v-control[ix])
                details.append({'drug':drug,'cell':cell,'genes':size,'mean_expression_r2':float(1-np.square(v-y).sum()/den) if den>0 else None,
                    'delta_pearson':float(delta['pearson']),'delta_spearman':float(delta['spearman'])})
    write(root/'predict_metrics.json',{'conditions':len(panel.drugs)*3,'continuous_genes':2000,'continuous':scores,'classification':classes,'expression_panels':details})


def update(config):
    root=config['run_root'];frozen=read(root/'update/freeze_manifest.json')
    for run in frozen['runs']:
        for item in run['files']:
            if sha256(root/item['file'])!=item['sha256']:raise ValueError('Update output checksum does not match update/freeze_manifest.json.')
    panel=subset_by_drugs(load_panel('test',allow_test=True),protocol()['test_drugs']);rows=[]
    for run in frozen['runs']:
        hidden=run['hidden'];truth={k:getattr(panel,k).reshape(-1,getattr(panel,k).shape[-1])[hidden] for k in ('continuous','de','up','sampled')}
        for item in run['files']:
            pred=dict(np.load(root/item['file']));score={k:float(v.mean()) for k,v in continuous_rows(truth['continuous'],pred['continuous']).items()}
            rows.append({'seed':run['seed'],'schedule':run['schedule'],'budget':item['budget'],'conditions':len(hidden),'continuous_genes':2000,
                'continuous':score,'classification':classification(truth['de'],truth['up'],truth['sampled'],pred['de'],pred['hard_de'],
                    direction_score=pred['up'],direction_hard=pred['up']>=.5)})
    write(root/'update_metrics.json',rows)
