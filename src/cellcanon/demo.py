"""Small offline integration demo; all inputs are synthetic and made in memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from .evaluate import classification
from .model.data import Panel
from .model.decoding import apply_matched_decoder, fit_matched_decoder
from .model.condition_llm_learning import fit_classification, predict_classification
from .model.unified_atlas_condition import AtlasBcMLP, identity_mask
from .model.condition_led_cells import prepare_condition_map, predict_condition_cells
from .model.probability_guided_cells import decision_weight, guided_map, guided_response, response_experts
from .model.dose_control_math import lfc_to_mean, ridge_fit, ridge_predict
from .model.full_cell_math import continuous_rows, finish_statistics, numpy_sufficient
from .model.observation_state import NumericalWorld, Observation, digest
from .model.observation_updaters import ResidualUpdater
from .model.explanation_verifier import path_proofs
from .model.explanation_repair import audit_prediction
from .model.answerable_explanation import validate_and_bind
from .model.path_completion_receipts import completion_receipt


GENES = 2931
CORE = 2000
CONTEXTS = ('context_0', 'context_1', 'context_2')


def _require(condition, message):
    if not condition:
        raise RuntimeError('Synthetic demo invariant failed: ' + message)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _response(features, coefficients):
    # A fictional shared latent mechanism creates correlated cell responses.
    return np.clip(np.einsum('ncf,fg->ncg', features, coefficients), -2., 2.)


def _panel(names, genes, response, split):
    de = np.abs(response) > .5
    up = np.where(de, response > 0, np.nan)
    return Panel(names, genes, genes, response[..., :CORE], de, up,
                 np.ones(de.shape, dtype=bool), [], split)


def _predict(rng, seed, output):
    support_n, validation_n = 10, 4
    support_ids = [f'synthetic_support_{i}' for i in range(support_n)]
    validation_ids = [f'synthetic_validation_{i}' for i in range(validation_n)]
    query_ids = ['synthetic_query']
    genes = [f'synthetic_gene_{i}' for i in range(GENES)]
    latent = rng.normal(size=(support_n + validation_n + 1, 1, 8))
    features = (latent + rng.normal(0, .15, (len(latent), 3, 8))).astype('f4')
    coefficients = rng.normal(0, .22, (8, GENES))
    support_y = _response(features[:support_n], coefficients)
    validation_y = _response(features[support_n:-1], coefficients)
    support = _panel(support_ids, genes, support_y, 'synthetic_support')
    validation = _panel(validation_ids, genes, validation_y, 'synthetic_validation')
    support_values = np.concatenate([support.continuous, support.de,
                                     np.nan_to_num(support.up)], axis=-1)
    # Only support observations fit this small Atlas projection.
    values = PCA(n_components=8, svd_solver='randomized', random_state=seed).fit_transform(
        support_values.reshape(support_n * 3, -1)).reshape(support_n, 3, 8)
    values /= max(float(values.std()), 1e-6)
    distances = np.square(latent[:, 0, None] - latent[None, :, 0]).mean(-1)
    similarity = np.exp(-distances)
    similarity = np.repeat(similarity[..., None], 3, axis=-1).astype('f4')
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
    bank = tensor(features[:support_n])
    observations = tensor(values)
    train_args = (bank, bank, observations, tensor(similarity[:support_n, :support_n]),
                  identity_mask(support_ids, support_ids))
    model = AtlasBcMLP(8, 8, cells=3, hidden=12, latent=8, genes=CORE, dropout=0)
    model.response_mean.copy_(tensor(support.continuous.mean(0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    target = tensor(support.continuous)
    losses = []
    for _ in range(8):
        loss = (model(*train_args) - target).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    query_args = (tensor(features[-1:]), bank, observations,
                  tensor(similarity[-1:, :support_n]), identity_mask(query_ids, support_ids))
    validation_args = (tensor(features[support_n:-1]), bank, observations,
                      tensor(similarity[support_n:-1, :support_n]),
                      identity_mask(validation_ids, support_ids))
    with torch.no_grad():
        z_train = model.encode(*train_args).numpy()
        z_validation = model.encode(*validation_args).numpy()
        z_query = model.encode(*query_args).numpy()
        parent, retrieval = model(*query_args, return_retrieval=True)
        parent = parent.numpy()
        fitted_support = model(*train_args).numpy()
    models = {task: fit_classification(z_train, support, ridge, epochs=4)
              for task, ridge in [('de', 10.), ('up', 100.)]}
    heads = {task: predict_classification(z_query, models[task])[task] for task in models}
    val_de = predict_classification(z_validation, models['de'])['de']
    decoder = fit_matched_decoder(validation, val_de)
    heads['hard_de'] = apply_matched_decoder(query_ids, genes, heads['de'], decoder)
    extra_model = ridge_fit(features[:support_n], support_y[..., CORE:], penalty=1., rank=8)
    predicted = np.concatenate([parent, ridge_predict(features[-1:], extra_model)], axis=-1)
    np.savez_compressed(output / 'predictions.npz', continuous=parent, **heads,
                        auxiliary_extra_response=predicted[..., CORE:])
    _require(np.isfinite(predicted).all(), 'finite predictions')
    _require(losses[-1] < losses[0], 'training must reduce its synthetic loss')
    _require(np.allclose(retrieval['weights'].numpy().sum(-1), 1.), 'normalized Atlas weights')
    return dict(features=features, coefficients=coefficients, support=support,
                support_y=support_y, fitted_support=fitted_support, parent=parent,
                predicted=predicted, heads=heads, genes=genes,
                report={'support_drugs': support_n, 'validation_drugs': validation_n,
                        'query_drugs': 1, 'contexts': 3, 'continuous_genes': CORE,
                        'classification_genes': GENES, 'training_epochs': 8,
                        'head_epochs': 4, 'initial_training_mse': losses[0],
                        'final_training_mse': losses[-1],
                        'count_decoder_validation_conditions': validation_n * 3,
                        'query_responses_used_for_fitting': False})


def _populations(rng, state, output):
    n = 48
    controls = rng.gamma(8., .08, size=(3, n, GENES)).astype('f4')
    controls[rng.random(controls.shape) < .05] = 0
    experts = response_experts(state['support_y'], state['support'].de, state['support'].up)
    generated, predictions, references = [], [], []
    # Query response labels are created only after frozen numerical prediction.
    query_truth = _response(state['features'][-1:], state['coefficients'])[0]
    ids = np.arange(n)
    for c in range(3):
        x = controls[c]
        cm, q0 = x.mean(0, dtype='f8'), (x > 0).mean(0)
        weight = decision_weight(state['heads']['de'][0, c], state['heads']['hard_de'][0, c])
        direction = state['heads']['up'][0, c]
        guided = guided_response(state['predicted'][0, c], weight, direction,
                                 experts['mean'][:, c], strength=1., cap=1.)
        target = lfc_to_mean(guided, cm)
        components = PCA(n_components=2).fit(x[:, :CORE]).components_
        detection = np.stack([q0 * .8, np.minimum(q0 * 1.1, 1.)])
        params = prepare_condition_map(x, state['genes'], ids, control_mean=cm,
            control_detection=q0, components=components, state_center=cm[:CORE],
            transform=np.eye(2), target_mean=target, learned_detection=q0,
            class_detection=detection, hard_de=state['heads']['hard_de'][0, c],
            up_probability=direction, expert_strength=0)
        params = guided_map(params, x, ids, target, weight, direction, detection)
        cells = predict_condition_cells(x, state['genes'], ids, params)
        replay = np.concatenate([predict_condition_cells(x[:11], state['genes'], ids[:11], params),
                                 predict_condition_cells(x[11:], state['genes'], ids[11:], params)])
        _require(np.array_equal(cells, replay), 'generation must be chunk-independent')
        _require(np.expm1(cells.astype('f8')).sum(1).max() <= 10000.01, 'CP10K mass bound')
        generated.append(cells)
        synthetic_treated = np.log1p(np.expm1(x.astype('f8')) * np.exp2(query_truth[c]))
        for population, destination in [(cells, predictions), (synthetic_treated, references)]:
            sums, ranks = numpy_sufficient(population, x)
            destination.append(finish_statistics(sums, ranks, np.asarray(n), cm, n))
    pred = {k: np.stack([p[k] for p in predictions])[None] for k in predictions[0]}
    ref = {k: np.stack([p[k] for p in references])[None] for k in references[0]}
    np.savez_compressed(output / 'generated_cells.npz', generated=np.stack(generated), controls=controls)
    np.savez_compressed(output / 'population_readouts.npz', **pred)
    sampled = np.ones_like(ref['de'], dtype=bool)
    metrics = classification(ref['de'], ref['up'], sampled, pred['de_score'], pred['de'],
                             direction_score=pred['lfc'], direction_hard=pred['lfc'] > 0)
    continuous = {k: float(np.mean(v)) for k, v in
                  continuous_rows(ref['lfc'][..., :CORE], pred['lfc'][..., :CORE]).items()}
    _require(np.isfinite(pred['lfc']).all(), 'finite population logFC')
    _require(int(ref['de'].sum()) > 0, 'synthetic reference must contain DE queries')
    _write(output / 'evaluation.json', {'synthetic': True, 'classification': metrics,
                                      'continuous_core': continuous})
    summary_metrics = {r['task']: r for r in metrics
                       if r['scope'] == 'dense' and r['role'] == 'all2931'}
    _require(summary_metrics['DIR_given_true_DE']['n'] == int(ref['de'].sum()),
             'DIR denominator must use reference DE genes')
    return dict(cells_per_context=n, generated_cells=3 * n, generated_genes=GENES,
                statistics='Rank-sum without tie correction; BH over 2931 genes',
                direction_score='signed generated logFC', hard_direction='logFC > 0',
                chunk_replay_exact=True, metrics=summary_metrics,
                continuous_core=continuous), ref


def _update(state, reference, output):
    residuals = state['support'].continuous - state['fitted_support']
    updater = ResidualUpdater(residuals, rank=4, noise_ratio=1.)
    prior = state['parent'][0].copy()
    world = NumericalWorld(drug='synthetic_query', conditions=CONTEXTS,
        genes={'lfc': state['genes'][:CORE]}, prior={'lfc': prior},
        variances={'lfc': updater.second_moment}, units={'lfc': 'log2_fold_change'},
        model_id='synthetic-atlas', budget=1, output=output / 'world')
    world.register_updater('lfc', updater, updater_id='synthetic-support-residuals')
    initial_id = world.state_id
    world.add_claim('prior-response', statement='Synthetic target response prediction.',
                    prediction_fields=['lfc'])
    source = reference['lfc'][0, 0, :CORE]
    observation = Observation('synthetic-assay-0', 'synthetic_query', CONTEXTS[0], 'lfc',
        tuple(state['genes'][:CORE]), tuple(source), tuple(np.full(CORE, .1)),
        'log2_fold_change', 'synthetic-source', digest(source.tolist()), 1.)
    world.observe(observation)
    proposal = world.propose_update(field='lfc', targets=CONTEXTS[1:], mode='gaussian', strength=.5)
    world.commit(proposal, reason='Offline synthetic source residual transfer.')
    revised = world.values['lfc'].copy()
    changed = int(np.count_nonzero(revised[1:] != prior[1:]))
    _require(changed > 0, 'admitted observation must revise hidden target values')
    _require(np.array_equal(revised[0], prior[0]), 'source prediction must remain separate from observation')
    _require(world.claims['prior-response']['status'] == 'invalidated', 'stale prediction claim')
    world.add_claim('source-dependent', statement='Synthetic source informs target prediction.',
                    observation_ids=['synthetic-assay-0'])
    replay = NumericalWorld.replay(output / 'world')
    _require(replay.state_id == world.state_id and np.array_equal(replay.values['lfc'], revised),
             'persisted state must replay exactly')
    updated_id = world.state_id
    world.retract('synthetic-assay-0', reason='Demonstrate withdrawal of a synthetic observation.')
    _require(np.array_equal(world.values['lfc'], prior), 'withdrawal must restore prior')
    _require(world.claims['source-dependent']['status'] == 'invalidated', 'withdrawn-source claim')
    np.savez_compressed(output / 'update_readouts.npz', prior=prior, after_observation=revised,
                        after_withdrawal=world.values['lfc'])
    result = dict(subsystem='ResidualUpdater + NumericalWorld; not the full selected Update fusion',
        source_context=CONTEXTS[0], target_contexts=list(CONTEXTS[1:]), observed_assays=1,
        changed_target_coordinates=changed, replay_exact=True, withdrawal_restores_prior=True,
        dependent_claim_invalidated=True, states={'initial': initial_id, 'updated': updated_id,
                                                 'withdrawn': world.state_id})
    _write(output / 'update.json', result)
    return result


def _explain(output):
    sources = [
        {'id': 'synthetic-e0', 'record': {'subject': 'synthetic_query', 'object': 'synthetic_target',
          'predicate': 'functional_target_action', 'action_sign': -1}},
        {'id': 'synthetic-e1', 'record': {'subject': 'synthetic_target', 'object': 'synthetic_gene',
          'predicate': 'regulates', 'action_sign': 1}}]
    route = {'id': 'synthetic-route', 'nodes': ['synthetic_target', 'synthetic_gene'],
             'tier': 'signed_transcript', 'sign': -1}
    case = {'drug': 'synthetic_query', 'sources': sources, 'route_candidates': [route],
        'claims': [{'id': 'background', 'kind': 'path_statement', 'arguments': route},
                   {'id': 'causal', 'kind': 'causal_path', 'arguments': route}]}
    # This is an explicitly scripted proposal, not language-model inference.
    raw = {'answers': [
        {'q': 0, 'label': 'supported', 'p': [1, 0, 0], 'refs': [0, 1],
         'reason': 'The two supplied synthetic edge signs compose to minus one.'},
        {'q': 1, 'label': 'insufficient', 'p': [0, 0, 1], 'refs': [],
         'reason': 'Synthetic background relations do not demonstrate an observed causal mechanism.'}],
        'routes': [0]}
    bound = validate_and_bind(case, raw, [s['id'] for s in sources])
    proofs = path_proofs(route, sources, case['drug'])
    before = audit_prediction(case, bound)
    after_case = {**case, 'sources': sources[:1]}
    after = audit_prediction(after_case, bound)
    _require(proofs and proofs[0]['sign'] == -1 and not before, 'complete signed proof')
    _require(not path_proofs(route, after_case['sources'], case['drug']) and
             any(r['id'] == 'background' for r in after), 'withdrawn edge must break the proof')
    completion = completion_receipt([{'refs': [0, 1], 'net_sign': None}],
                                    [{'action_sign': -1}, {'action_sign': None}])
    _require(completion['possible_assignments'] == 2, 'unknown sign enumeration')
    result = dict(subsystem='Evidence binding, proof checks and source withdrawal',
        synthetic=True, proposal_producer='scripted', sources=sources, bound_proposal=bound,
        complete_proofs=proofs, audit_before_withdrawal=before, audit_after_withdrawal=after,
        unknown_sign_completions=completion, withdrawn_source='synthetic-e1',
        background_proof_invalidated=True, causal_evidence_claimed=False)
    _write(output / 'explanation.json', result)
    return result


def run_demo(output, *, seed=29):
    """Run real kernels on generated inputs, writing only to an empty directory."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a new or empty output directory for the synthetic demo.')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, old_threads))
    torch.manual_seed(seed)
    try:
        state = _predict(rng, seed, output)
        generation, reference = _populations(rng, state, output)
        updating = _update(state, reference, output)
        explanation = _explain(output)
        result = dict(synthetic=True, offline=True, paper_reproduction=False, seed=seed,
            predict=state['report'], generation=generation, update=updating,
            explain={'subsystem': explanation['subsystem'], 'proposal_producer': 'scripted',
                     'proofs_checked': len(explanation['complete_proofs']),
                     'background_proof_invalidated': explanation['background_proof_invalidated']},
            limitations=[
                'Inputs are synthetic, with eight-dimensional features and a short training run.',
                'Generation uses one synthetic condition per context and synthetic detection experts.',
                'Update uses in-sample support residuals to demonstrate prediction and state revision.',
                'Explain checks a scripted answer against its sources and tests evidence withdrawal.',
                'Metrics describe this example run. Use the README workflow for the paper experiments.'],
            artifacts=['predictions.npz', 'generated_cells.npz', 'population_readouts.npz',
                       'evaluation.json', 'explanation.json', 'update.json',
                       'update_readouts.npz', 'world/'])
        _write(output / 'summary.json', result)
        return result
    finally:
        torch.set_num_threads(old_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New or empty directory outside the source tree.')
    parser.add_argument('--seed', type=int, default=29)
    args = parser.parse_args()
    result = run_demo(args.output, seed=args.seed)
    print('Synthetic offline demo completed.')
    print(f"Generated {result['generation']['generated_cells']} cells; "
          f"revised {result['update']['changed_target_coordinates']} target coordinates; "
          'verified replay and source withdrawal.')
    print('Read summary.json in the selected output directory for results and output filenames.')


if __name__ == '__main__':
    main()
