from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any
import anndata as ad
import numpy as np
import pandas as pd
SPLIT_VERSION = 'sciplex3_drug_ood_v1'
CONTROL_TREATMENT = 'S0000'
EXPECTED_DRUGS = 188
EXPECTED_COUNTS = {'support': 56, 'validation': 19, 'test': 113}
EXPECTED_CELLS = 581777
EXPECTED_CONTROL_CELLS = 13004
EXPECTED_CELL_TYPES = ('A549', 'MCF7', 'K562')
LABEL_NAMES = ('DE', 'DIR', 'ExtraLM_DE', 'ExtraLM_DIR')
TRUTH_SUFFIXES = ('HVG2000_plus_ExtraLM_stats', 'ExtraLM_only_stats')

def sha256_file(path: Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (block := stream.read(chunk_size)):
            digest.update(block)
    return digest.hexdigest()

def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)

def split_key(drug_id: str, split_version: str) -> str:
    return hashlib.sha256(f'{split_version}\t{drug_id}'.encode('utf-8')).hexdigest()

def normalize_name(value: object) -> str:
    return ' '.join(str(value).split())

def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))

def build_registry(pdata: pd.DataFrame, split_version: str) -> tuple[pd.DataFrame, dict[str, str]]:
    vehicle = pdata['vehicle'].astype(str).str.upper()
    if set(vehicle.unique()) != {'TRUE', 'FALSE'}:
        raise ValueError(f'Unexpected vehicle values: {sorted(vehicle.unique())}')
    controls = pdata.loc[vehicle.eq('TRUE')].copy()
    treated = pdata.loc[vehicle.eq('FALSE')].copy()
    if len(pdata) != EXPECTED_CELLS or len(controls) != EXPECTED_CONTROL_CELLS:
        raise ValueError(f'Unexpected cell counts: total={len(pdata)}, control={len(controls)}')
    if set(controls['treatment'].astype(str)) != {CONTROL_TREATMENT}:
        raise ValueError('Vehicle cells do not map uniquely to S0000')
    if set(pdata['time_point'].astype(str)) != {'24'}:
        raise ValueError('Drug split must be constructed from the 24 h-only v3 build')
    if set(pdata['cell_type'].astype(str)) != set(EXPECTED_CELL_TYPES):
        raise ValueError('Unexpected cell-line coverage')
    drug_columns = ['treatment', 'product_name', 'catalog_number']
    drugs = treated[drug_columns].drop_duplicates().copy()
    if drugs['treatment'].nunique() != EXPECTED_DRUGS or len(drugs) != EXPECTED_DRUGS:
        raise ValueError(f'Expected {EXPECTED_DRUGS} one-to-one drug rows, observed {len(drugs)}')
    if treated.groupby('treatment', observed=True)['product_name'].nunique().max() != 1:
        raise ValueError('A drug ID maps to multiple product names')
    if treated.groupby('product_name', observed=True)['treatment'].nunique().max() != 1:
        raise ValueError('A product name maps to multiple drug IDs')
    if treated.groupby('treatment', observed=True)['catalog_number'].nunique().max() != 1:
        raise ValueError('A drug ID maps to multiple catalog numbers')
    drugs = drugs.rename(columns={'treatment': 'drug_id', 'product_name': 'drug_raw'})
    drugs['drug_id'] = drugs['drug_id'].astype(str)
    drugs['drug_raw'] = drugs['drug_raw'].astype(str)
    drugs['canonical_name'] = drugs['drug_raw'].map(normalize_name)
    if drugs['canonical_name'].nunique() != EXPECTED_DRUGS:
        raise ValueError('Whitespace-normalized canonical names are not one-to-one')
    drugs['inchi_key'] = ''
    drugs['split_hash'] = drugs['drug_id'].map(lambda value: split_key(value, split_version))
    drugs = drugs.sort_values(['split_hash', 'drug_id'], kind='mergesort').reset_index(drop=True)
    if drugs['split_hash'].duplicated().any():
        raise ValueError('SHA-256 split-key collision')
    drugs['split_rank'] = np.arange(EXPECTED_DRUGS, dtype=np.int16)
    drugs['split'] = np.select([drugs['split_rank'].lt(56), drugs['split_rank'].lt(75)], ['support', 'validation'], default='test')
    drugs['is_control'] = False
    drugs['split_version'] = split_version
    control = pd.DataFrame([{'drug_id': CONTROL_TREATMENT, 'drug_raw': 'Vehicle', 'catalog_number': CONTROL_TREATMENT, 'canonical_name': 'Vehicle', 'inchi_key': '', 'split_hash': '', 'split_rank': pd.NA, 'split': 'control', 'is_control': True, 'split_version': split_version}])
    registry = pd.concat([drugs, control], ignore_index=True)
    registry['split_rank'] = registry['split_rank'].astype('Int16')
    registry = registry[['drug_id', 'drug_raw', 'canonical_name', 'catalog_number', 'inchi_key', 'split_hash', 'split_rank', 'split', 'is_control', 'split_version']]
    mapping = dict(zip(drugs['drug_id'], drugs['split'], strict=True))
    return (registry, mapping)

def build_cell_registry(pdata: pd.DataFrame, registry: pd.DataFrame, mapping: dict[str, str], split_version: str) -> pd.DataFrame:
    name_map = registry.set_index('drug_id')['canonical_name'].astype(str).to_dict()
    vehicle = pdata['vehicle'].astype(str).str.upper().eq('TRUE')
    cell_split = pdata['treatment'].astype(str).map(mapping)
    cell_split = cell_split.mask(vehicle, 'control')
    if cell_split.isna().any():
        raise ValueError('At least one cell lacks a split assignment')
    access_map = {'support': 'training_allowed', 'validation': 'validation_only', 'test': 'locked_test_only', 'control': 'shared_transductive_control'}
    result = pd.DataFrame({'cell': pdata['cell'].astype(str), 'source_cell_index': pdata['source_cell_index'].astype(np.int32), 'valid_cell_index': pdata['valid_cell_index'].astype(np.int32), 'cell_type': pdata['cell_type'].astype(str), 'drug_id': pdata['treatment'].astype(str), 'canonical_name': pdata['treatment'].astype(str).map(name_map), 'dose_nM': pdata['dose'].astype(str), 'replicate': pdata['replicate'].astype(str), 'time_point_hours': pdata['time_point'].astype(str), 'is_control': vehicle.to_numpy(), 'split': cell_split.astype(str), 'access_policy': cell_split.astype(str).map(access_map), 'split_version': split_version})
    if result['cell'].duplicated().any():
        raise ValueError('Duplicate cells in split registry')
    return result

def condition_summary(cell_registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = cell_registry.groupby(['split', 'cell_type', 'drug_id', 'canonical_name', 'is_control'], sort=True)
    for (split, cell, drug_id, name, is_control), group in grouped:
        doses = sorted(group['dose_nM'].astype(str).unique(), key=lambda value: float(value))
        replicates = sorted(group['replicate'].astype(str).unique())
        time_points = sorted(group['time_point_hours'].astype(str).unique(), key=float)
        rows.append({'split': split, 'cell_type': cell, 'drug_id': drug_id, 'canonical_name': name, 'is_control': bool(is_control), 'n_cells': len(group), 'n_doses': len(doses), 'doses_nM': ';'.join(doses), 'n_replicates': len(replicates), 'replicates': ';'.join(replicates), 'time_points_hours': ';'.join(time_points)})
    return pd.DataFrame(rows).sort_values(['split', 'cell_type', 'drug_id'], kind='mergesort')

def output_directory(split_root: Path, split: str) -> Path:
    return split_root / ('locked_test' if split == 'test' else split)

def relative_file_record(dataset_root: Path, path: Path) -> dict[str, Any]:
    return {'path': str(path.relative_to(dataset_root)), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--split-version', default=SPLIT_VERSION)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    metadata_root = dataset_root / 'metadata'
    split_root = dataset_root / 'splits' / 'drug_zero_shot_v1'
    registry_csv = metadata_root / 'drug_split_v1.csv'
    registry_parquet = metadata_root / 'drug_split_v1.parquet'
    cell_parquet = metadata_root / 'cell_split_v1.parquet'
    condition_csv = metadata_root / 'drug_cell_condition_split_v1.csv'
    planned = [split_root, registry_csv, registry_parquet, cell_parquet, condition_csv]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite existing split artifacts:\n' + '\n'.join(existing))
    source_h5ad = dataset_root / 'sciplex3_core_hvg2000_input.h5ad'
    pdata_path = metadata_root / 'valid_cells_pdata.parquet'
    label_paths = [dataset_root / 'labels' / f'{cell}_{name}.csv' for cell in EXPECTED_CELL_TYPES for name in LABEL_NAMES]
    truth_paths = [dataset_root / 'truth_full_stats' / f'{cell}_{suffix}.parquet' for cell in EXPECTED_CELL_TYPES for suffix in TRUTH_SUFFIXES]
    require_files([source_h5ad, pdata_path, *label_paths, *truth_paths])
    pdata = pd.read_parquet(pdata_path)
    registry, mapping = build_registry(pdata, args.split_version)
    cell_registry = build_cell_registry(pdata, registry, mapping, args.split_version)
    conditions = condition_summary(cell_registry)
    treated_registry = registry.loc[~registry['is_control']].copy()
    counts = treated_registry['split'].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Unexpected drug counts: {counts}')
    drug_sets = {split: set(treated_registry.loc[treated_registry['split'].eq(split), 'drug_id'].astype(str)) for split in EXPECTED_COUNTS}
    if set.union(*drug_sets.values()) != set(treated_registry['drug_id'].astype(str)):
        raise ValueError('Drug split union is incomplete')
    for left in EXPECTED_COUNTS:
        for right in EXPECTED_COUNTS:
            if left < right and drug_sets[left] & drug_sets[right]:
                raise ValueError(f'Drug leakage between {left} and {right}')
    treated_conditions = conditions.loc[~conditions['is_control']]
    expected_condition_counts = {split: count * 3 for split, count in EXPECTED_COUNTS.items()}
    observed_condition_counts = treated_conditions['split'].value_counts().to_dict()
    if observed_condition_counts != expected_condition_counts:
        raise ValueError(f'Unexpected drug-cell condition counts: {observed_condition_counts}')
    if not treated_conditions['n_doses'].eq(4).all() or not treated_conditions['n_replicates'].eq(2).all():
        raise ValueError('A treated drug-cell condition lacks four doses or two replicates')
    if set(treated_conditions['time_points_hours']) != {'24'}:
        raise ValueError('A treated condition contains a non-24 h time point')
    metadata_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=False)
    registry.to_csv(registry_csv, index=False, lineterminator='\n')
    registry.to_parquet(registry_parquet, index=False)
    cell_registry.to_parquet(cell_parquet, index=False)
    conditions.to_csv(condition_csv, index=False, lineterminator='\n')
    source = ad.read_h5ad(source_h5ad)
    if list(source.obs_names.astype(str)) != cell_registry['cell'].tolist():
        raise ValueError('H5AD obs order differs from the cell split registry')
    if source.shape != (EXPECTED_CELLS, 2000):
        raise ValueError(f'Unexpected source H5AD shape: {source.shape}')
    partition_audit: dict[str, Any] = {}
    output_paths: list[Path] = [registry_csv, registry_parquet, cell_parquet, condition_csv]
    control_cells = set(cell_registry.loc[cell_registry['is_control'], 'cell'])
    source_obs_set = set(source.obs_names.astype(str))
    materialized_treated_cells: set[str] = set()
    for split in ('support', 'validation', 'test'):
        destination = output_directory(split_root, split)
        (destination / 'labels').mkdir(parents=True, exist_ok=True)
        (destination / 'truth_full_stats').mkdir(parents=True, exist_ok=True)
        cell_mask = cell_registry['is_control'] | cell_registry['split'].eq(split)
        expected_obs = cell_registry.loc[cell_mask, 'cell'].tolist()
        subset = source[cell_mask.to_numpy()].copy()
        subset.obs['drug_split'] = cell_registry.loc[cell_mask, 'split'].to_numpy()
        subset.obs['dataset_partition'] = split
        subset.obs['is_shared_control'] = cell_registry.loc[cell_mask, 'is_control'].to_numpy()
        subset.obs['split_version'] = args.split_version
        subset.uns['drug_zero_shot_split'] = {'split_version': args.split_version, 'partition': split, 'dmso_policy': 'shared_transductive_control', 'treated_drug_count': EXPECTED_COUNTS[split], 'response_access': {'support': 'training_allowed', 'validation': 'validation_only', 'test': 'locked_test_only'}[split]}
        h5ad_path = destination / f'sciplex3_core_hvg2000_{split}_input.h5ad'
        subset.write_h5ad(h5ad_path, compression='gzip')
        output_paths.append(h5ad_path)
        if list(subset.obs_names.astype(str)) != expected_obs:
            raise ValueError(f'{split} H5AD obs order mismatch')
        subset_control = set(subset.obs_names[subset.obs['is_shared_control'].astype(bool)].astype(str))
        if subset_control != control_cells:
            raise ValueError(f'{split} does not contain the exact shared DMSO set')
        subset_treated = set(subset.obs_names.astype(str)) - subset_control
        if materialized_treated_cells & subset_treated:
            raise ValueError(f'Treated-cell leakage into {split}')
        materialized_treated_cells.update(subset_treated)
        label_rows: dict[str, int] = {}
        for cell in EXPECTED_CELL_TYPES:
            for label_name in LABEL_NAMES:
                source_path = dataset_root / 'labels' / f'{cell}_{label_name}.csv'
                frame = pd.read_csv(source_path)
                part = frame.loc[frame['pert'].astype(str).isin(drug_sets[split])].copy()
                if set(part['pert'].astype(str)) - drug_sets[split]:
                    raise ValueError(f'Label leakage in {source_path.name}/{split}')
                output_path = destination / 'labels' / source_path.name
                part.to_csv(output_path, index=False, lineterminator='\n')
                output_paths.append(output_path)
                label_rows[source_path.name] = len(part)
        truth_rows: dict[str, int] = {}
        for cell in EXPECTED_CELL_TYPES:
            for suffix in TRUTH_SUFFIXES:
                source_path = dataset_root / 'truth_full_stats' / f'{cell}_{suffix}.parquet'
                frame = pd.read_parquet(source_path)
                part = frame.loc[frame['pert'].astype(str).isin(drug_sets[split])].copy()
                if set(part['pert'].astype(str)) != drug_sets[split]:
                    raise ValueError(f'Truth partition does not cover every {split} drug: {source_path.name}')
                output_path = destination / 'truth_full_stats' / source_path.name
                part.to_parquet(output_path, index=False)
                output_paths.append(output_path)
                truth_rows[source_path.name] = len(part)
        partition_audit[split] = {'drugs': EXPECTED_COUNTS[split], 'drug_cell_conditions': expected_condition_counts[split], 'treated_cells': len(subset_treated), 'shared_control_cells': len(subset_control), 'materialized_h5ad_cells': subset.n_obs, 'h5ad_shape': list(subset.shape), 'h5ad_nnz': int(subset.X.nnz), 'label_rows': label_rows, 'truth_rows': truth_rows}
        del subset
        gc.collect()
    if materialized_treated_cells | control_cells != source_obs_set:
        raise ValueError('Materialized split cells do not cover the source H5AD')
    if len(materialized_treated_cells) != EXPECTED_CELLS - EXPECTED_CONTROL_CELLS:
        raise ValueError('Materialized treated-cell count mismatch')
    for source_path in label_paths:
        source_frame = pd.read_csv(source_path)
        observed = sum((partition_audit[split]['label_rows'][source_path.name] for split in EXPECTED_COUNTS))
        if observed != len(source_frame):
            raise ValueError(f'Partitioned label rows do not sum to source: {source_path.name}')
    for source_path in truth_paths:
        source_frame = pd.read_parquet(source_path, columns=['pert'])
        observed = sum((partition_audit[split]['truth_rows'][source_path.name] for split in EXPECTED_COUNTS))
        if observed != len(source_frame):
            raise ValueError(f'Partitioned truth rows do not sum to source: {source_path.name}')
    assignment_records = treated_registry[['drug_id', 'split_hash', 'split_rank', 'split']].astype({'split_rank': int}).to_dict('records')
    assignment_sha = sha256_json(assignment_records)
    audit = {'method_version': args.split_version, 'split_axis': 'drug_id_only', 'response_blind': True, 'standard_time_point_hours': 24, 'drug_id_source': 'GSM4150378 treatment/catalog code', 'split_key_formula': 'sha256(split_version + TAB + drug_id)', 'split_rank_base': 0, 'drug_counts': EXPECTED_COUNTS, 'drug_cell_condition_counts': expected_condition_counts, 'control_drug_id': CONTROL_TREATMENT, 'control_cells': EXPECTED_CONTROL_CELLS, 'dmso_policy': 'shared_transductive_control', 'drug_set_pairwise_overlap': {'support_validation': len(drug_sets['support'] & drug_sets['validation']), 'support_test': len(drug_sets['support'] & drug_sets['test']), 'validation_test': len(drug_sets['validation'] & drug_sets['test'])}, 'all_three_cell_lines_per_drug': bool(treated_conditions.groupby('drug_id')['cell_type'].nunique().eq(3).all()), 'all_four_doses_per_drug_cell': bool(treated_conditions['n_doses'].eq(4).all()), 'all_two_replicates_per_drug_cell': bool(treated_conditions['n_replicates'].eq(2).all()), 'all_24h': bool(treated_conditions['time_points_hours'].eq('24').all()), 'treated_cells_partitioned_once': True, 'shared_control_set_exact_in_all_partitions': True, 'split_assignment_sha256': assignment_sha, 'partition_audit': partition_audit, 'train_access_rule': 'support treated response plus shared DMSO only', 'validation_access_rule': 'validation treated response is forbidden during fitting and used only for tuning/calibration', 'test_access_rule': 'test treated response is locked until final evaluation', 'gene_split_created': False, 'cell_line_split_created': False, 'scaffold_split_created': False}
    audit_path = split_root / 'split_audit.json'
    write_json(audit_path, audit)
    output_paths.append(audit_path)
    source_paths = [source_h5ad, pdata_path, *label_paths, *truth_paths]
    manifest = {'method_version': args.split_version, 'created_from_dataset': str(dataset_root), 'split_assignment_sha256': assignment_sha, 'inputs': {str(path.relative_to(dataset_root)): relative_file_record(dataset_root, path) for path in source_paths}, 'outputs': {str(path.relative_to(dataset_root)): relative_file_record(dataset_root, path) for path in output_paths}, 'leakage_boundary': {'partition_unit': 'canonical treatment/catalog code', 'same_drug_all_cells_doses_replicates_genes_one_split': True, 'response_statistics_used_for_assignment': False, 'input_row_order_used_for_assignment': False, 'shared_control_policy': 'shared_transductive_control'}}
    manifest_path = split_root / 'split_manifest.json'
    write_json(manifest_path, manifest)
    print(json.dumps({'split_root': str(split_root), 'drug_registry': str(registry_csv), 'cell_registry': str(cell_parquet), 'split_assignment_sha256': assignment_sha, 'drug_counts': EXPECTED_COUNTS, 'partition_audit': partition_audit, 'manifest_sha256': sha256_file(manifest_path)}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
