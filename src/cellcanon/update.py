"""Fit update models and apply them as new measurements become available."""
from pathlib import Path
from itertools import combinations
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .workflow import read,write,protocol
from .model.data import metadata,drug_kernels,load_panel,sha256
from .model.isolation import subset_by_drugs
from .model.response_feature_head import ResponseFeatureBuilder
from .model.response_count_head import count_features
from .model.observation_updaters import ResidualUpdater
from .model.condition_feedback_benchmark import condition_partition
from .model.unified_mlp_update import UnifiedMLPUpdate


def patterns():
    for target in range(3):
        sources=[c for c in range(3) if c!=target]
        for n in (1,2):
            for cells in combinations(sources,n):yield target,cells


def fit_oof(config):
    from . import fit_update_oof as routine
    routine.PARENT=config['run_root']/'model';routine.ROOT=config['run_root']/'update'
    routine.FEATURES=config['run_root']/'knowledge/features_complete.json'
    routine.fit()


def fit(config):
    root=config['run_root']/'update';out=root/'fit'
    if (root/'selection.json').exists():raise FileExistsError('A fitted update model already exists. Set run_root to a new directory to fit another model.')
    out.mkdir(parents=True,exist_ok=True)
    tr=dict(np.load(config['run_root']/'model/inputs/support_labels.npz'));oof=dict(np.load(root/'oof/predictions.npz'))
    choice=read(Path(__file__).with_name('protocols')/'update.json');np.savez_compressed(out/'support.npz',**tr)
    residual={'de_probability':tr['de']-oof['de'],'up_probability':np.where(tr['de']==1,np.nan_to_num(tr['up'])-oof['up'],0.)}
    for key,r in residual.items():ResidualUpdater(r,rank=8,noise_ratio=10.).save(out/(key+'.npz'))
    builder=ResponseFeatureBuilder(tr['de'],tr['up']);rng=np.random.default_rng(20260916);xs=[];ys=[];us=[];cx=[];cy=[]
    for target,cells in patterns():
        for i in range(len(tr['de'])):
            observed={c:{k:np.nan_to_num(tr[k][i,c]) for k in ('continuous','de','up')} for c in cells}
            heads={k:oof[k][i] for k in ('de','up')};x=builder.features(oof['continuous'][i],heads,observed,target,exclude=i)
            keep=(tr['de'][i,target]==1)|(rng.random(2931)<.05)
            xs.append(x[keep]);ys.append(tr['de'][i,target,keep]);us.append(tr['up'][i,target,keep])
            cx.append(count_features(observed,target));cy.append(np.log1p([tr['de'][i,target,:2000].sum(),tr['de'][i,target,2000:].sum()]))
    x=np.concatenate(xs);y=np.concatenate(ys);u=np.concatenate(us)
    for task in ('de','up'):
        tree=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=7,l2_regularization=1.,early_stopping=False,random_state=20260916)
        tree.fit(x if task=='de' else x[y==1],y if task=='de' else u[y==1]);joblib.dump(tree,out/choice['features'][task]['model'])
    count=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(np.stack(cx),np.stack(cy));joblib.dump(count,out/choice['counts']['model'])
    choice.update(files_sha256={str(f.relative_to(root)):sha256(f) for f in [*out.iterdir(),root/'oof/predictions.npz'] if f.is_file()},source_sha256={})
    write(root/'selection.json',choice)


def kernels(drugs):
    meta,_=metadata();order=list(meta.drug);k=drug_kernels(meta)['chemical_biology'];p=protocol()
    norm=np.sqrt(np.maximum(np.diag(k),1e-8));k=k/(norm[:,None]*norm[None,:])
    ti=[order.index(d) for d in p['train_drugs']];qi=[order.index(d) for d in drugs]
    return k[np.ix_(ti,ti)],k[np.ix_(qi,ti)],k[np.ix_(qi,qi)]


def run(config):
    root=config['run_root'];p=protocol();parent=dict(np.load(root/'model/freeze/compiled_semantic.npz'))
    schedules=read(Path(__file__).with_name('protocols')/'update_schedule.json')
    if (root/'update/freeze_manifest.json').exists():raise FileExistsError('Update predictions already exist. Use evaluate-update to score them.')
    # The simulator holds the benchmark; the updater receives only explicitly
    # admitted source rows. Hidden responses are scored in a separate command.
    panel=subset_by_drugs(load_panel('test',allow_test=True),p['test_drugs']);ks=kernels(p['test_drugs']);runs=[]
    for schedule in schedules:
        partition=condition_partition(p['test_drugs'],p['cells'],schedule['seed'],identities=p['identity_groups'])
        engine=UnifiedMLPUpdate.from_artifacts(root/'update',parent['continuous'],parent,partition,len(schedule['actions']),ks)
        out=root/'update/predictions'/str(schedule['seed'])/schedule['schedule'];out.mkdir(parents=True)
        receipts=[];files=[]
        for step,j in enumerate(schedule['actions'],1):
            di,ci=divmod(j,3);receipts.append(engine.observe(int(j),panel.continuous[di,ci],panel.de[di,ci],panel.up[di,ci]))
            if step in schedule['budgets']:
                pred=engine.predict();hidden=partition['hidden'];file=out/f'{step}.npz'
                np.savez_compressed(file,**{k:a.reshape(-1,a.shape[-1])[hidden] for k,a in pred.items()})
                files.append({'file':str(file.relative_to(root)),'sha256':sha256(file),'budget':step})
        write(out/'receipts.json',receipts)
        runs.append({'seed':schedule['seed'],'schedule':schedule['schedule'],'hidden':partition['hidden'],'files':files})
    write(root/'update/freeze_manifest.json',{'runs':runs,'model_parameters_updated':False,'endpoint':'condition-level readouts before cell generation'})
