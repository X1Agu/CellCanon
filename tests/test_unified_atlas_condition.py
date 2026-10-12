"""Readout preservation and admitted-observation flow, without external data."""
import pytest
import torch

from cellcanon.model.unified_atlas_condition import (
    UnifiedAtlasCondition, AtlasBcMLP, identity_mask)


def fixture():
    torch.manual_seed(7)
    net = UnifiedAtlasCondition(6, 9, cells=2, hidden=8, output_dimensions=4, dropout=0)
    args = [torch.randn(2, 2, 6), torch.randn(3, 2, 6),
            torch.randn(3, 2, 9), torch.rand(2, 3, 3),
            identity_mask(['a', 'query'], ['a', 'b', 'c'])]
    return net, args


def test_alias_identity_and_fold_exclusion_are_both_applied():
    mask = identity_mask(['same', 'novel'], ['same', 'b', 'c'],
                         excluded=torch.tensor([[False, True, False], [False, False, True]]))
    assert mask.tolist() == [[False, False, True], [True, True, False]]
    with pytest.raises(ValueError, match='unique canonical'):
        identity_mask(['q'], ['alias_group', 'alias_group'])
    with pytest.raises(ValueError, match='independent Atlas donor'):
        identity_mask(['a'], ['a'])


def test_excluded_observation_cannot_change_query_representation():
    net, args = fixture(); net.eval()
    expected, detail = net(*args, return_retrieval=True)
    changed = [x.clone() for x in args]
    changed[2][0] += 1000
    actual = net(*changed)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert torch.count_nonzero(detail['weights'][0, :, 0]) == 0
    assert not torch.allclose(actual[1], expected[1])


def test_biological_query_and_admitted_atlas_both_affect_one_latent():
    net, args = fixture(); net.eval()
    expected = net(*args)
    assert expected.shape == (2, 2, 4)
    for index in (0, 2, 3):
        changed = [x.clone() for x in args]
        changed[index] = torch.zeros_like(changed[index])
        assert not torch.allclose(net(*changed), expected)
    expected.square().sum().backward()
    assert net.query_map.weight.grad.abs().sum() > 0
    assert net.value_map[0].weight.grad.abs().sum() > 0
    assert net.biological_prior_weights.grad.abs().sum() > 0


def test_continuous_bc_is_nonlinear_and_frontend_learns_without_parent_outputs():
    _, args = fixture()
    model = AtlasBcMLP(6, 9, cells=2, hidden=8, latent=4, genes=5, dropout=0)
    before = model.frontend.query_map.weight.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    target = torch.randn(2, 2, 5)
    for _ in range(3):
        loss = (model(*args)-target).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert model(*args).shape == (2, 2, 5)
    assert not torch.equal(before, model.frontend.query_map.weight)
    assert model.frontend.value_map[0].weight.grad.abs().sum() > 0
    assert not hasattr(model, 'parent')


def test_classification_features_are_separate_from_continuous_bc_readout():
    _, args = fixture()
    model = AtlasBcMLP(6, 9, cells=2, hidden=8, latent=4, genes=5, dropout=0).eval()
    z = model.encode(*args)
    with torch.no_grad():
        model.bc_mlp[0][-1].bias.add_(10)
    torch.testing.assert_close(z, model.encode(*args), rtol=0, atol=0)


def test_no_legal_donor_and_invalid_priors_fail_before_head_execution():
    net, args = fixture()
    bad = [x.clone() for x in args]; bad[-1][0] = False
    with pytest.raises(ValueError, match='All donors excluded'):
        net(*bad)
    bad = [x.clone() for x in args]; bad[3][0, 0, 0] = 2
    with pytest.raises(ValueError, match='Similarity priors'):
        net(*bad)
