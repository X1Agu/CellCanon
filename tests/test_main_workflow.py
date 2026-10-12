"""Synthetic preparation -> train -> frozen query outputs, without downloads."""
import numpy as np
import pandas as pd
from cellcanon import workflow
from cellcanon.model.data import Panel,CELLS


def test_main_training_and_freeze_do_not_open_test_responses(tmp_path,monkeypatch):
    rng=np.random.default_rng(31)
    p={'train_drugs':[f'd{i}' for i in range(10)],'validation_drugs':[f'v{i}' for i in range(3)],
       'test_drugs':['q0','q1'],'cells':list(CELLS)}
    all_drugs=p['train_drugs']+p['validation_drugs']+p['test_drugs'];p['identity_groups']={d:d for d in all_drugs}
    genes=[f'g{i}' for i in range(2931)];reads=[]
    panels={}
    for split,key in [('support','train_drugs'),('validation','validation_drugs')]:
        shape=(len(p[key]),3,2931);de=rng.integers(0,2,shape);up=np.where(de,rng.integers(0,2,shape),np.nan)
        panels[split]=Panel(p[key],genes,genes,rng.normal(size=shape[:2]+(2000,)),de,up,np.ones(shape,bool),[],split)
    def load(split,**kwargs):
        reads.append(split)
        assert split in panels,'Test responses were opened by training/inference.'
        return panels[split]
    monkeypatch.setattr(workflow,'protocol',lambda:p)
    monkeypatch.setattr(workflow.data,'load_panel',load)
    monkeypatch.setattr(workflow.training,'load_panel',load)
    monkeypatch.setattr(workflow.data,'metadata',lambda:(pd.DataFrame({'drug':all_drugs}),None))
    similarities={k:.9*np.eye(len(all_drugs))+.1 for k in ('chemical','target','moa')}
    monkeypatch.setattr(workflow.data,'drug_kernels',lambda _:similarities)
    k=tmp_path/'knowledge';k.mkdir()
    features={d+'_'+c:{'compiled_structure':{'constant':1.,'cell='+c:1.},'compiled_vector':rng.normal(size=12).tolist(),
        'compiled_target_vector':rng.normal(size=12).tolist(),'compiler_valid':True} for d in all_drugs for c in CELLS}
    workflow.write(k/'features_complete.json',features)
    config={'run_root':tmp_path};workflow.prepare_model(config)
    protocol=workflow.read(tmp_path/'model/protocol.json')
    protocol.update(epochs=1,checkpoints=[1],classifier_epochs=2)
    workflow.write(tmp_path/'model/protocol.json',protocol)
    workflow.training.fit(tmp_path/'model');workflow.predict(config)
    prediction=np.load(tmp_path/'model/freeze/compiled_semantic.npz')
    assert prediction['continuous'].shape==(2,3,2000)
    assert prediction['de'].shape==(2,3,2931)
    assert set(reads)=={'support','validation'}
    assert np.isfinite(prediction['continuous']).all()
