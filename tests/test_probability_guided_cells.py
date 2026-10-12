import numpy as np
import pytest
from cellcanon.model.probability_guided_cells import (
    decision_weight, guided_response, response_experts, guided_map)
from cellcanon.model.condition_led_cells import prepare_condition_map, predict_condition_cells


def test_boundary_weight_is_monotone_and_not_raw_detection_probability():
    p=np.array([.01,.1,.2,.6]); h=p>=.2
    w=decision_weight(p,h)
    assert (np.diff(w)>0).all() and w[1]<.5<w[2]
    assert not np.array_equal(w,p)


def test_response_neutrality_and_bounded_correction():
    r=np.array([3.,-.1,.2]);e=np.array([[-2.]*3,[2.]*3]);w=np.array([1.,1.,0.])
    u=np.array([0.,.5,1.])
    np.testing.assert_array_equal(guided_response(r,w,u,e,strength=0),r)
    y=guided_response(r,w,u,e,strength=1,cap=.6)
    assert y[0]==pytest.approx(2.4)
    np.testing.assert_array_equal(y[1:],r[1:])
    with pytest.raises(ValueError):guided_response(r,w,u+1,e)


def test_support_response_experts_ignore_non_de_labels_and_backoff():
    y=np.array([[[-1.,99.]],[[2.,99.]]]);de=np.array([[[1,0]],[[1,0]]],bool)
    up=np.array([[[0.,np.nan]],[[1.,np.nan]]])
    e=response_experts(y,de,up)
    np.testing.assert_allclose(e['mean'],[[[-1.,-1.]],[[2.,2.]]])
    assert e['counts'].sum()==2


def test_guided_map_replays_cells_and_preserves_requested_mean_without_projection():
    rng=np.random.default_rng(9);x=rng.random((40,5));x[x<.3]=0
    genes=np.array(list('abcde'));ids=np.arange(40);cm=x.mean(0);q0=(x>0).mean(0)
    p=prepare_condition_map(x,genes,ids,control_mean=cm,control_detection=q0,
        components=np.eye(5)[:2],state_center=cm,transform=np.eye(2),target_mean=cm,
        learned_detection=q0,class_detection=np.stack([q0,q0]),hard_de=np.ones(5,bool),
        up_probability=np.ones(5),expert_strength=0)
    target=cm*1.1
    g=guided_map(p,x,ids,target,np.full(5,.8),np.full(5,.9),np.stack([q0*.8,np.ones(5)]),
                 detection_strength=1.,shape_strength=0.)
    y=predict_condition_cells(x,genes,ids,g,project=False)
    replay=np.concatenate([predict_condition_cells(x[:13],genes,ids[:13],g,project=False),
                           predict_condition_cells(x[13:],genes,ids[13:],g,project=False)])
    np.testing.assert_array_equal(y,replay)
    np.testing.assert_allclose(y.mean(0,dtype='f8'),target,atol=1e-7)
    assert np.all(np.var(y,axis=0)>0)
    assert np.any(g['requested_detection']!=p['requested_detection'])
