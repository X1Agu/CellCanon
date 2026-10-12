from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from cellcanon.model.observation_state import NumericalWorld, Observation, observations_from_table
from cellcanon.model.observation_updaters import ResidualUpdater, ResidualObservation


def fixture(output=None):
    rng=np.random.default_rng(3)
    latent=rng.normal(size=(20,1,4))
    model=ResidualUpdater(np.concatenate([latent,2*latent],axis=1),rank=4,noise_ratio=.1)
    world=NumericalWorld(drug='D',conditions=['C0','C1'],genes={'lfc':['g0','g1','g2','g3']},
        prior={'lfc':np.zeros((2,4))},variances={'lfc':model.second_moment},
        units={'lfc':'log2_fold_change'},model_id='S0',budget=2,output=output)
    world.register_updater('lfc',model,updater_id='support-v1')
    observation=Observation('o0','D','C0','lfc',('g0','g1'),(1.,-1.),(.1,.1),
        'log2_fold_change','source','a'*64,1.)
    return world,model,observation


def update(world):
    p=world.propose_update(field='lfc',targets=['C1'],mode='gaussian',strength=.5)
    world.commit(p,reason='Measured source residual transferred within supported cells')


def test_real_update_idempotence_retraction_and_replay(tmp_path):
    world,model,o=fixture(tmp_path/'world'); initial=world.state_id
    world.observe(o); observed=world.state_id
    assert observed!=initial and world.spent==1
    np.testing.assert_array_equal(world.values['lfc'],0)
    assert world.observe(o)==observed and world.spent==1
    world.add_claim('old',statement='Old prediction',prediction_fields=['lfc'])
    update(world)
    assert world.claims['old']['status']=='invalidated'
    assert world.values['lfc'][1,0]>0 and world.values['lfc'][1,1]<0
    np.testing.assert_array_equal(world.values['lfc'][0],0)
    updated=world.state_id
    world.add_claim('new',statement='Observed source changed target',observation_ids=['o0'])
    recovered=NumericalWorld.replay(tmp_path/'world')
    assert recovered.state_id==world.state_id
    np.testing.assert_array_equal(recovered.values['lfc'],world.values['lfc'])
    recovered.register_updater('lfc',model,updater_id='support-v1')
    recovered.retract('o0',reason='Source calibration failed')
    np.testing.assert_array_equal(recovered.values['lfc'],0)
    assert recovered.spent==1 and recovered.claims['new']['status']=='invalidated'
    old=NumericalWorld.replay(tmp_path/'world',state_id=updated)
    assert old.values['lfc'][1,0]>0


@pytest.mark.parametrize('changes',[{'drug':'other'},{'units':'counts'},{'genes':('unknown','g1')},
    {'genes':('g0','g0')},{'variances':(-1.,0.)},{'values':(np.nan,0.)},{'cost':3.},
    {'source_sha256':'invalid'},{'condition':'C2'}])
def test_invalid_observations_are_atomic(changes):
    world,_,o=fixture();before=world.state_id
    with pytest.raises(ValueError): world.observe(replace(o,**changes))
    assert world.state_id==before and world.spent==0 and not world.observations


def test_stale_and_modified_proposals_and_failed_persistence(monkeypatch):
    world,_,o=fixture();world.observe(o)
    p=world.propose_update(field='lfc',targets=['C1'],mode='gaussian',strength=.5)
    p['values'][0,0]=np.nan
    with pytest.raises(ValueError):world.commit(p,reason='modified')
    p=world.propose_update(field='lfc',targets=['C1'],mode='gaussian',strength=.5)
    world.add_claim('x',statement='New state')
    with pytest.raises(ValueError):world.commit(p,reason='stale')
    before=world.state_id; values=world.values['lfc'].copy()
    def fail():raise OSError('disk failure')
    monkeypatch.setattr(world,'_persist',fail)
    with pytest.raises(OSError):update(world)
    assert world.state_id==before and not world.write_rules
    np.testing.assert_array_equal(values,world.values['lfc'])


def test_retraction_failure_and_updater_version_are_atomic():
    world,model,o=fixture();world.observe(o);update(world)
    with pytest.raises(ValueError):world.register_updater('lfc',model,updater_id='other-version')
    world.observe(replace(o,observation_id='o1'))
    update(world)
    before=world.state_id;values=world.values['lfc'].copy();world.updaters.clear()
    with pytest.raises(KeyError):world.retract('o0',reason='requires unavailable updater')
    assert world.state_id==before and not world.retracted
    np.testing.assert_array_equal(world.values['lfc'],values)


def test_partial_noise_and_duplicate_ids(tmp_path):
    world,model,o=fixture();world.observe(o)
    with pytest.raises(ValueError):world.observe(replace(o,values=(2.,-1.)))
    r=world.residual_observations('lfc')
    ix,_,v=model._observed(r+r)
    np.testing.assert_array_equal(ix,[0,1]);np.testing.assert_allclose(v,.05)
    with pytest.raises(ValueError):model.infer([ResidualObservation(0,np.array([.5]),np.ones(1),np.ones(1))])
    csv=tmp_path/'obs.csv'
    pd.DataFrame([dict(observation_id='o2',drug='D',condition='C0',field='lfc',gene='g2',value=0.,variance=.5,units='log2_fold_change')]).to_csv(csv,index=False)
    imported=observations_from_table(csv,source_id='file')[0]
    assert imported.genes==('g2',) and imported.values==(0.,)
    world.observe(imported);assert world.spent==2
    with pytest.raises(ValueError):world.observe(replace(imported,observation_id='o3'))


def test_degenerate_updater_roundtrip(tmp_path):
    model=ResidualUpdater(np.zeros((4,2,3)))
    model.save(tmp_path/'model.npz');other=ResidualUpdater.load(tmp_path/'model.npz')
    obs=ResidualObservation(0,np.array([0]),np.ones(1),np.ones(1))
    np.testing.assert_array_equal(other.infer([obs])[0],0)
