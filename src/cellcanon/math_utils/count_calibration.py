from __future__ import annotations
import re
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from cellcanon.contracts import SchemaError, require_columns
from cellcanon.math_utils.compound_descriptors import build_sciplex_retrieval_metadata
CELLS = ('A549', 'K562', 'MCF7')
SUPPORT_DRUGS = 56
VALIDATION_DRUGS = 19
TEST_DRUGS = 113

def _normal(value: object) -> str:
    return re.sub('[^a-z0-9]', '', str(value).casefold())

def _metadata_match_score(new_name: str, old_name: str) -> float:
    from difflib import SequenceMatcher
    left, right = (_normal(new_name), _normal(old_name))
    score = SequenceMatcher(None, left, right).ratio()
    if left.startswith(right) or right.startswith(left):
        score = max(score, 0.9 + 0.09 * min(len(left), len(right)) / max(len(left), len(right)))
    elif right in left or left in right:
        score = max(score, 0.85 + 0.1 * min(len(left), len(right)) / max(len(left), len(right)))
    return float(score)

def build_v3_drug_metadata(drug_split: pd.DataFrame, legacy_registry: pd.DataFrame, *, radius: int, fp_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(drug_split, ['drug_id', 'canonical_name', 'split', 'is_control'], 'Drug split')
    require_columns(legacy_registry, ['drug', 'smiles', 'target', 'pathway'], 'SciPlex compound registry')
    drugs = drug_split.loc[~drug_split['is_control'].astype(bool)].copy()
    matches: list[dict[str, Any]] = []
    for row in drugs.itertuples(index=False):
        candidates = [(_metadata_match_score(str(row.canonical_name), str(old.drug)), old) for old in legacy_registry.itertuples(index=False)]
        score, selected = max(candidates, key=lambda item: (item[0], -len(str(item[1].drug))))
        if score < 0.9:
            raise SchemaError(f'Response-blind metadata match is ambiguous for {row.canonical_name}: {score}')
        matches.append({'drug_id': str(row.drug_id), 'canonical_name': str(row.canonical_name), 'split': str(row.split), 'legacy_drug': str(selected.drug), 'metadata_match_score': score, 'smiles': str(selected.smiles), 'target': str(selected.target), 'pathway': str(selected.pathway)})
    match_frame = pd.DataFrame(matches).sort_values('drug_id', kind='stable').reset_index(drop=True)
    if len(match_frame) != SUPPORT_DRUGS + VALIDATION_DRUGS + TEST_DRUGS:
        raise SchemaError('Metadata mapping did not map all 188 non-control drugs')
    adapted = match_frame.rename(columns={'drug_id': 'drug'})
    metadata = build_sciplex_retrieval_metadata(adapted, radius=radius, fp_size=fp_size)
    metadata = metadata.merge(match_frame[['drug_id', 'canonical_name', 'split', 'legacy_drug', 'metadata_match_score']], left_on='drug', right_on='drug_id', how='left', validate='one_to_one').drop(columns='drug_id')
    if not metadata[['chemical_available', 'target_available', 'moa_available']].all().all():
        raise SchemaError('Response-blind metadata is incomplete')
    return (metadata, match_frame)

def _condition_budget_features(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for condition, block in frame.groupby('condition_id', sort=False):
        values = block[score].to_numpy(dtype=float)
        rows.append({'condition_id': str(condition), 'drug': str(block['drug'].iloc[0]), 'cell_line': str(block['cell_line'].iloc[0]), 'score_mean': float(values.mean()), 'score_std': float(values.std()), 'score_q90': float(np.quantile(values, 0.9)), 'score_q95': float(np.quantile(values, 0.95)), 'score_q99': float(np.quantile(values, 0.99)), 'score_gt_half': float(np.mean(values >= 0.5)), **{f'cell_{cell}': float(str(block['cell_line'].iloc[0]) == cell) for cell in CELLS}})
    return pd.DataFrame(rows)

def _fit_budget_decoder(validation: pd.DataFrame, *, score: str, alpha_grid: Sequence[float]) -> dict[str, Any]:
    features = ['score_mean', 'score_std', 'score_q90', 'score_q95', 'score_q99', 'score_gt_half', *[f'cell_{cell}' for cell in CELLS]]
    condition = _condition_budget_features(validation, score)
    truth = validation.groupby('condition_id', sort=False)['is_de'].mean().rename('truth_fraction')
    condition = condition.merge(truth, left_on='condition_id', right_index=True, validate='one_to_one')
    groups = condition['drug'].astype(str).to_numpy()
    x = condition[features].to_numpy(dtype=float)
    y = condition['truth_fraction'].to_numpy(dtype=float)
    splitter = GroupKFold(n_splits=min(5, condition['drug'].nunique()))
    candidates = []
    for alpha in map(float, alpha_grid):
        oof = np.full(len(condition), np.nan)
        for fit_index, hold_index in splitter.split(x, y, groups):
            model = Ridge(alpha=alpha).fit(x[fit_index], y[fit_index])
            oof[hold_index] = model.predict(x[hold_index])
        candidates.append({'alpha': alpha, 'mae': float(np.mean(np.abs(oof - y))), 'correlation': float(np.corrcoef(oof, y)[0, 1])})
    selected = sorted(candidates, key=lambda row: (row['mae'], -row['correlation'], row['alpha']))[0]
    model = Ridge(alpha=float(selected['alpha'])).fit(x, y)
    return {'features': features, 'selected_alpha': float(selected['alpha']), 'coefficient': model.coef_.tolist(), 'intercept': float(model.intercept_), 'candidate_scores': candidates, 'validation_rows': int(len(condition)), 'validation_mae': float(selected['mae'])}

def _apply_budget_decoder(frame: pd.DataFrame, spec: Mapping[str, Any], *, score: str) -> pd.DataFrame:
    condition = _condition_budget_features(frame, score)
    x = condition[list(spec['features'])].to_numpy(dtype=float)
    fraction = x @ np.asarray(spec['coefficient'], dtype=float) + float(spec['intercept'])
    condition['predicted_de_fraction'] = np.clip(fraction, 1.0 / frame.groupby('condition_id').size().max(), 0.8)
    condition['predicted_de_count'] = 0
    hard = np.zeros(len(frame), dtype='int8')
    position = pd.Series(np.arange(len(frame), dtype=int), index=frame.index)
    for row in condition.itertuples(index=False):
        index = frame.index[frame['condition_id'].astype(str).eq(str(row.condition_id))]
        k = int(np.clip(round(float(row.predicted_de_fraction) * len(index)), 1, len(index)))
        selected = frame.loc[index, score].nlargest(k).index
        hard[position.loc[selected].to_numpy(dtype=int)] = 1
        condition.loc[condition['condition_id'].eq(row.condition_id), 'predicted_de_count'] = k
    return (condition, hard)
