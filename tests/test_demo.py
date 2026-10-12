"""Cross-stage offline smoke coverage with actual numeric/proof/state kernels."""
import json
import urllib.request

import numpy as np
import pytest

from cellcanon.demo import run_demo


def test_offline_demo_integrates_prediction_population_evaluation_and_withdrawal(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('The offline demo attempted a network request.')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    output = tmp_path / 'demo'
    report = run_demo(output)
    assert report['synthetic'] and report['offline'] and not report['paper_reproduction']
    assert report['predict']['query_responses_used_for_fitting'] is False
    assert report['predict']['final_training_mse'] < report['predict']['initial_training_mse']
    frozen = np.load(output / 'predictions.npz')
    population = np.load(output / 'population_readouts.npz')
    generated = np.load(output / 'generated_cells.npz')['generated']
    assert generated.shape == (3, 48, 2931)
    np.testing.assert_allclose(population['mean'][0], generated.mean(1, dtype='f8'))
    np.testing.assert_array_equal(population['up'], population['lfc'] > 0)
    assert np.expm1(generated.astype('f8')).sum(-1).max() <= 10000.01
    assert report['generation']['metrics']['DIR_given_true_DE']['n'] > 0
    updated = np.load(output / 'update_readouts.npz')
    np.testing.assert_array_equal(updated['prior'], frozen['continuous'][0])
    np.testing.assert_array_equal(updated['after_observation'][0], updated['prior'][0])
    assert np.any(updated['after_observation'][1:] != updated['prior'][1:])
    np.testing.assert_array_equal(updated['after_withdrawal'], updated['prior'])
    explain = json.loads((output / 'explanation.json').read_text())
    assert explain['proposal_producer'] == 'scripted'
    assert explain['audit_before_withdrawal'] == []
    assert explain['audit_after_withdrawal'][0]['id'] == 'background'
    assert report == json.loads((output / 'summary.json').read_text())


def test_demo_rejects_existing_outputs_before_writing(tmp_path):
    existing = tmp_path / 'keep.txt'
    existing.write_text('preserve this')
    with pytest.raises(ValueError, match='new or empty'):
        run_demo(tmp_path)
    assert existing.read_text() == 'preserve this'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['keep.txt']
