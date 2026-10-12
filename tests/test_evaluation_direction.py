"""Regression tests for the actual frozen-cell and condition-head evaluators."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from cellcanon import evaluate
from cellcanon.model.data import CELLS, Panel, sha256


def fixture(monkeypatch, directions):
    shape=(1,3,2931)
    de=np.zeros(shape,dtype=bool);up=np.full(shape,np.nan);sampled=np.zeros(shape,dtype=bool)
    scores=np.zeros(shape)
    for start in (0,2000):
        de[...,start:start+4]=True
        up[...,start:start+4]=[0,0,1,1]
        sampled[...,start:start+5]=True
        scores[...,start:start+4]=directions
    continuous=np.broadcast_to(np.linspace(-.5,.5,2000),(1,3,2000)).copy()
    genes=[f'g{i}' for i in range(2931)]
    panel=Panel(['d0'],genes,genes,continuous,de,up,sampled,[],'test')
    monkeypatch.setattr(evaluate,'load_panel',lambda split,allow_test:panel)
    monkeypatch.setattr(evaluate,'protocol',lambda:{'test_drugs':['d0']})
    return panel,scores


def dir_rows(rows):
    return [r for r in rows if r['task']=='DIR_given_true_DE']


@pytest.mark.parametrize('lfc,expected_positive_f1,expected_joint',[
    ([-.2,.1,.2,.3],.8,(1+2/3+.8)/3),
    ([-.1,0.,.1,.5],1.,1.),
])
def test_generated_cell_entrypoint_preserves_logfc_ranking_and_zero_threshold(tmp_path,monkeypatch,lfc,expected_positive_f1,expected_joint):
    panel,scores=fixture(monkeypatch,lfc)
    out=tmp_path/'cells_test';out.mkdir()
    file=out/'population_readouts.npz'
    np.savez_compressed(file,lfc=scores,de=panel.de,de_score=np.where(panel.de,2.,0.),
                        mean=evaluate.lfc_to_mean(scores,np.ones(2931)))
    evaluate.write(out/'manifest.json',{'files_sha256':{file.name:sha256(file)}})
    context=tmp_path/'cell_model';context.mkdir()
    for cell in CELLS:np.savez_compressed(context/(cell+'.npz'),mean=np.ones(2931))
    frame=pd.DataFrame({'pert':['d0']*2000,'ensembl_id':panel.genes[:2000],
                        'scores':np.linspace(-1.,1.,2000)})
    monkeypatch.setattr(evaluate.pd,'read_parquet',lambda path:frame)
    evaluate.cells({'run_root':tmp_path,'data_root':tmp_path/'external_data'})
    rows=evaluate.read(tmp_path/'predict_metrics.json')['classification']
    assert len(dir_rows(rows))==6
    for row in dir_rows(rows):
        assert row['n']==(24 if row['role']=='all2931' else 12)
        assert row['positives']==row['n']//2
        assert row['auroc']==pytest.approx(1.)
        assert row['ap']==pytest.approx(1.)
        assert row['auprc']==pytest.approx(1.)
        assert row['positive_f1']==pytest.approx(expected_positive_f1)
    for row in rows:
        if row['task']=='joint_three_state':assert row['macro_f1']==pytest.approx(expected_joint)
        if row['task']=='DE':assert row['positive_f1']==pytest.approx(1.)
    if expected_positive_f1<1:
        assert roc_auc_score([0,0,1,1],np.asarray(lfc)>0)==pytest.approx(.75)


@pytest.mark.parametrize('probability,expected_positive_f1,expected_joint',[
    ([.2,.49,.5,.6],1.,1.),
    ([.2,.6,.7,.8],.8,(1+2/3+.8)/3),
])
def test_condition_head_entrypoint_preserves_probability_ranking_and_half_threshold(tmp_path,monkeypatch,probability,expected_positive_f1,expected_joint):
    panel,scores=fixture(monkeypatch,probability)
    out=tmp_path/'update';out.mkdir()
    file=out/'predictions.npz'
    np.savez_compressed(file,continuous=panel.continuous[:,0],de=np.where(panel.de[:,0],.8,.1),
                        hard_de=panel.de[:,0],up=scores[:,0])
    evaluate.write(out/'freeze_manifest.json',{'runs':[{'seed':1,'schedule':'limited','hidden':[0],
        'files':[{'file':'update/predictions.npz','sha256':sha256(file),'budget':1}]}]})
    evaluate.update({'run_root':tmp_path})
    rows=evaluate.read(tmp_path/'update_metrics.json')[0]['classification']
    assert len(dir_rows(rows))==6
    for row in dir_rows(rows):
        assert row['n']==(8 if row['role']=='all2931' else 4)
        assert row['positives']==row['n']//2
        assert row['auroc']==pytest.approx(1.)
        assert row['ap']==pytest.approx(1.)
        assert row['auprc']==pytest.approx(1.)
        assert row['positive_f1']==pytest.approx(expected_positive_f1)
    for row in rows:
        if row['task']=='joint_three_state':assert row['macro_f1']==pytest.approx(expected_joint)
