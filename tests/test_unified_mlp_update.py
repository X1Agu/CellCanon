"""Scientific boundaries of Update with the selected parent and admitted observations."""
import copy
import json
from pathlib import Path
from cellcanon.model.observation_updaters import ResidualUpdater
import numpy as np
import pytest
from cellcanon.model.unified_mlp_update import UnifiedMLPUpdate


def fixture(*, aliases=False, budget=1, shift=0.):
    rng=np.random.default_rng(2);n=3
    parent=rng.normal(size=(n,3,2000))+shift
    heads={'de':np.full((n,3,2931),.1),'up':np.full((n,3,2931),.6),'hard_de':np.zeros((n,3,2931),np.int8)}
    support={'continuous':rng.normal(size=(4,3,2000)), 'de':rng.integers(0,2,(4,3,2931)), 'up':rng.integers(0,2,(4,3,2931))}
    oof={'continuous':np.zeros((4,3,2000))}
    part={'hidden':[0,3,6], 'observation':[1,2,4,5,7,8], 'identities':['a','a' if aliases else 'b','c']}
    cfg=json.loads((Path(__file__).parents[1]/'src/cellcanon/protocols/update.json').read_text())
    residual={k:ResidualUpdater(rng.normal(size=(4,3,2931)),rank=3,noise_ratio=10.) for k in cfg['residual_strengths']}
    class Tree:
        def predict_proba(self,x):
            p=1/(1+np.exp(-x[:,0]+x[:,3]));return np.stack([1-p,p],axis=1)
    class Count:
        def predict(self,x):return np.tile(np.log1p([80.,40.]),(len(x),1))
    engine=UnifiedMLPUpdate(parent,heads,part,budget,support,oof,(np.eye(4),np.ones((n,4)),np.eye(n)),cfg,residual,{'de':Tree(),'up':Tree()},Count())
    return engine,parent,heads


def assay():return np.ones(2000),np.ones(2931),np.ones(2931)


def test_empty_state_exactly_preserves_each_latest_parent_readout():
    e,p,h=fixture();r=e.predict()
    np.testing.assert_array_equal(r['continuous'],p)
    for k in h:np.testing.assert_array_equal(r[k],h[k])


@pytest.mark.parametrize('candidate',[0,3,6,-1,10,1.0])
def test_hidden_invalid_rejection_is_atomic(candidate):
    e,_,_=fixture();state=e.state_id
    with pytest.raises(ValueError):e.observe(candidate,*assay())
    assert e.state_id==state and not e.records and not e.session.observations


def test_alias_assay_cannot_be_acquired_twice():
    e,_,_=fixture(aliases=True,budget=2);e.observe(1,*assay());state=e.state_id
    assert 4 not in e.legal()
    with pytest.raises(ValueError):e.observe(4,*assay())
    assert e.state_id==state


def test_budget_and_direction_checks_leave_state_unchanged():
    e,_,_=fixture();c,d,u=assay();u[0]=.2;state=e.state_id
    with pytest.raises(ValueError):e.observe(1,c,d,u)
    assert e.state_id==state
    e.observe(1,*assay());state=e.state_id
    with pytest.raises(ValueError):e.observe(2,*assay())
    assert e.state_id==state


def test_state_binds_parent_and_predictions_do_not_mutate_receipts():
    e,_,_=fixture();changed,_,_=fixture(shift=.01)
    assert e.state_id!=changed.state_id
    c,d,u=assay();e.observe(1,c,d,u);state=e.state_id;c[:]=20
    assert np.all(e.records[1]['continuous']==1)
    e.predict()
    assert state==e.state_id


def test_hidden_alias_is_rejected_at_construction():
    e,p,h=fixture(aliases=True)
    part=copy.deepcopy(e.session.partition);part['hidden']=[0,6];part['observation'].append(3)
    with pytest.raises(ValueError,match='Hidden alias'):
        UnifiedMLPUpdate(p,h,part,1,{'de':np.zeros((4,3,2931)),'up':np.zeros((4,3,2931))}, {}, (),e.selection,{}, {},None)
