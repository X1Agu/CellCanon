"""Fit the selected dose/state/detection auxiliaries and generate populations."""
import numpy as np
import h5py
import joblib
from scipy import sparse
from scipy.special import expit,logit
from sklearn.decomposition import PCA
from .workflow import read,write,protocol
from .model import dose_control as source
from .model.data import CELLS,sha256
from .model.condition_llm_features import FeatureMap,row_features
from .model.dose_control_math import (DOSES,group_statistics,dose_features,flat,unflat,ridge_fit,ridge_predict,
                                     covariance,sym_function,covariance_decode,gaussian_transport,lfc_to_mean)
from .model.condition_led_cells import class_detection_experts,prepare_condition_map,predict_condition_cells
from .model.probability_guided_cells import response_experts,decision_weight,guided_response,guided_map
from .model.full_cell_math import numpy_sufficient,finish_statistics


def fit(config):
    root=config['run_root'];out=root/'cell_model'
    if (out/'manifest.json').exists():raise FileExistsError('A fitted cell model already exists. Set run_root to a new directory to fit another model.')
    out.mkdir(exist_ok=True);p=protocol();obs,genes=source.obs_and_genes();drugs=p['train_drugs'];index={d:i for i,d in enumerate(drugs)}
    packets=read(root/'knowledge/packets.json');features={}
    mapper=FeatureMap('static_text',32)
    for split,key in [('train','train_drugs'),('validation','validation_drugs'),('test','test_drugs')]:
        rows=[row_features(packets[d+'_'+c]) for d in p[key] for c in CELLS]
        z=mapper.fit(rows,[(d,c) for d in drugs for c in CELLS]) if split=='train' else mapper.transform(rows)
        features[split]=z.reshape(len(p[key]),3,-1)
    joblib.dump(mapper,out/'auxiliary_mapper.joblib');np.savez_compressed(out/'features.npz',**features)
    labels=[];detect=[];raw_detect=[];q0s=[];q0raw=[];logcov=[]
    for ci,cell in enumerate(CELLS):
        cr=np.flatnonzero(obs.control & obs.cell_type.astype(str).eq(cell))
        controls=source.selected_expression(cr).toarray();cm=controls.mean(0,dtype='f8')
        q0=((controls>0).sum(0)+.5)/(len(controls)+1);q0s.append(q0);q0raw.append((controls>0).mean(0))
        pca=PCA(n_components=16,svd_solver='randomized',random_state=20260911).fit(controls[:,:2000])
        z=pca.transform(controls[:,:2000]);cc=covariance(z,.1)
        np.savez_compressed(out/(cell+'.npz'),expression=controls,rows=cr,replicate=obs.iloc[cr].replicate.to_numpy(str),
            mean=cm,detection=(controls>0).mean(0),smoothed_detection=q0,genes=genes.ensembl_id.to_numpy(str),
            components=pca.components_,center=pca.mean_,control_cov=cc)
        sr=np.flatnonzero(obs.cell_type.astype(str).eq(cell)&obs.treatment.isin(index)&~obs.control)
        sub=obs.iloc[sr];x=source.selected_expression(sr)
        codes=np.array([index[str(d)] for d in sub.treatment])*4+np.searchsorted(DOSES,sub.dose_numeric)
        ng=len(drugs)*4;counts=np.bincount(codes,minlength=ng)
        combined=sparse.vstack([source.selected_expression(cr),x]);control=np.r_[np.ones(len(cr),bool),np.zeros(len(sr),bool)]
        stats=group_statistics(combined,np.r_[np.full(len(cr),-1),codes],ng,control)
        labels.append({k:stats[k].reshape(len(drugs),4,-1) for k in ('lfc','de','up')})
        nz=x.copy();nz.eliminate_zeros();nz.data=np.ones(len(nz.data),float)
        group=sparse.csr_matrix((np.ones(len(sr)),(codes,np.arange(len(sr)))),shape=(ng,len(sr)))
        detected=(group@nz).toarray();q=(detected+.5)/(counts[:,None]+1)
        detect.append((logit(q)-logit(q0)).reshape(len(drugs),4,-1))
        raw_detect.append((detected/counts[:,None]).reshape(len(drugs),4,-1))
        zx=np.asarray(x[:,:2000]@pca.components_.T)-pca.mean_@pca.components_.T
        logcov.append(np.stack([sym_function(covariance(zx[codes==j],.1),np.log).ravel() for j in range(ng)]).reshape(len(drugs),4,-1))
    truth={k:np.stack([a[k] for a in labels],axis=1) for k in ('lfc','de','up')}
    x=dose_features(features['train'])
    for name,y,penalty,rank in [('dose',truth['lfc'][...,:2000],1.,8),('extra',truth['lfc'][...,2000:],1.,8),
        ('detection',np.stack(detect,axis=1),10.,None),('state',np.stack(logcov,axis=1),1.,None)]:
        np.savez_compressed(out/(name+'.npz'),**ridge_fit(x,flat(y),penalty,rank))
    experts=class_detection_experts(np.stack(raw_detect,axis=1),np.stack(q0raw),truth['de'],truth['up'])
    np.savez_compressed(out/'detection_experts.npz',**experts)
    support=np.load(root/'model/inputs/support_labels.npz')
    np.savez_compressed(out/'response_experts.npz',**response_experts(support['continuous'],support['de'][...,:2000],support['up'][...,:2000]))
    write(out/'manifest.json',{'fit_drugs':drugs,'fit_treated_scope':'support only','state_reference':'DMSO only',
        'files_sha256':{f.name:sha256(f) for f in out.iterdir() if f.is_file()},
        'selected_parameters':{'mean_strength':1.,'cap':1.,'detection_strength':1.,'shape_strength':0.,'state_strength':.1,'amplitude_retention':.25}})


def generate(config,split='test'):
    root=config['run_root'];fitted=root/'cell_model';out=root/('cells_'+split)
    if out.exists():raise FileExistsError('Use a new output directory for generation.')
    frozen=read(fitted/'manifest.json')
    for name,h in frozen['files_sha256'].items():
        if sha256(fitted/name)!=h:raise ValueError('Cell model hash mismatch.')
    out.mkdir();p=protocol();ds=p['test_drugs' if split=='test' else 'validation_drugs'];obs,genes=source.obs_and_genes();names=genes.ensembl_id.to_numpy(str)
    path=root/'model/freeze/compiled_semantic.npz' if split=='test' else root/'model/learning/compiled_semantic_validation.npz'
    parent=dict(np.load(path));z=np.load(fitted/'features.npz')[split];x=dose_features(z)
    pred={k:unflat(ridge_predict(x,dict(np.load(fitted/(k+'.npz'))))) for k in ('dose','extra','detection')}
    cov=unflat(covariance_decode(ridge_predict(x,dict(np.load(fitted/'state.npz'))),16))
    response=np.load(fitted/'response_experts.npz')['mean'];detection=np.load(fitted/'detection_experts.npz')['probability']
    rowmap=obs.groupby(['cell_type','treatment','dose_numeric'],observed=True,sort=False).indices
    sums=np.zeros((len(ds),3,len(names)));us=sums.copy();counts=np.zeros((len(ds),3),int);statistics=[];groups=[]
    for ci,cell in enumerate(CELLS):
        ctx=dict(np.load(fitted/(cell+'.npz')));cm=ctx['mean'];controls=ctx['expression']
        raw=lfc_to_mean(np.concatenate([pred['dose'][:,ci],pred['extra'][:,ci]],axis=-1),cm)
        q=expit(np.clip(pred['detection'][:,ci]+logit(ctx['smoothed_detection'])[None,None,:],-12,12))
        with h5py.File(out/(cell+'.h5'),'w') as h:
            h.create_dataset('genes',data=names.astype('S'));h.create_dataset('control_mean',data=cm)
            for di,drug in enumerate(ds):
                allocation=[np.sort(rowmap[cell,drug,dose]) for dose in DOSES];ns=np.array([len(rows) for rows in allocation])
                pooled=np.average(raw[di,:,:2000],axis=0,weights=ns)
                hard=parent['hard_de'][di,ci].astype(bool);up=parent['up'][di,ci]
                w=decision_weight(parent['de'][di,ci],hard)
                goal=lfc_to_mean(guided_response(parent['continuous'][di,ci],w[:2000],up[:2000],response[:,ci],strength=1.,cap=1.),cm[:2000])
                gain=np.divide(goal,pooled,out=np.zeros_like(goal),where=pooled>0)
                for ki,(dose,rows) in enumerate(zip(DOSES,allocation)):
                    group_id=ci*len(ds)*4+di*4+ki;rng=np.random.default_rng(20260912+group_id)
                    reps=obs.iloc[rows].replicate.astype(str).to_numpy();choice=np.empty(len(rows),int)
                    for rep in sorted(set(reps)):
                        ix=np.flatnonzero(reps==rep);pool=np.flatnonzero(ctx['replicate']==rep)
                        choice[ix]=rng.choice(pool,len(ix),replace=len(ix)>len(pool))
                    control=controls[choice];target=raw[di,ki].copy();target[:2000]=np.where(pooled>0,target[:2000]*gain,goal)
                    target=np.clip(target,0,np.log1p(10000))
                    params=prepare_condition_map(control,names,rows,control_mean=cm,control_detection=ctx['detection'],
                        components=ctx['components'],state_center=ctx['center'],transform=gaussian_transport(ctx['control_cov'],cov[di,ci,ki]),
                        target_mean=target,learned_detection=q[di,ki],class_detection=detection[:,ci],hard_de=hard,up_probability=up,
                        expert_strength=0.,amplitude_retention=.25,state_strength=.1)
                    params=guided_map(params,control,rows,target,w,up,detection[:,ci],detection_strength=1.,shape_strength=0.)
                    y=predict_condition_cells(control,names,rows,params);ss,uu=numpy_sufficient(y,controls)
                    sums[di,ci]+=ss;us[di,ci]+=uu;counts[di,ci]+=len(y)
                    g=h.create_group(drug+'/'+str(int(dose)));g.create_dataset('expression',data=y,compression='lzf')
                    g.create_dataset('control_source_row',data=ctx['rows'][choice]);g.create_dataset('target_allocation_row',data=rows)
                    maps=out/'maps';maps.mkdir(exist_ok=True);np.savez_compressed(maps/f'{group_id:04d}.npz',**params)
                    groups.append({'drug':drug,'cell':cell,'dose_nM':int(dose),'n':len(y),'map':f'maps/{group_id:04d}.npz'})
        statistics.append(finish_statistics(sums[:,ci],us[:,ci],counts[:,ci],cm,len(controls)))
    np.savez_compressed(out/'population_readouts.npz',**{k:np.stack([s[k] for s in statistics],axis=1) for k in statistics[0]})
    write(out/'manifest.json',{'split':split,'drugs':ds,'cells':list(CELLS),'groups':groups,'cells_generated':int(counts.sum()),
        'readout':'Wilcoxon no-tie approximation, BH over 2931 genes, q<0.05 and abs(log2FC)>0.25',
        'files_sha256':{str(f.relative_to(out)):sha256(f) for f in out.rglob('*') if f.is_file()},'treated_query_response_reads':0})
