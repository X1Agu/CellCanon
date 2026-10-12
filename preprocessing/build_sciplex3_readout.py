from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr
from skmisc.loess import loess
RAW_NAMES = {'matrix': 'GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix.gz', 'cells': 'GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt.gz', 'genes': 'GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz', 'pdata': 'GSM4150378_sciPlex3_pData.txt.gz', 'hash_metadata': 'GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz'}
VALID_CELL_TYPES = ('A549', 'MCF7', 'K562')
CONTROL_TREATMENT = 'S0000'
N_HVG = 2000
N_LINCS = 978
SEED = 42
METHOD_VERSION = 'sciplex3-count-readout-v1'
OFFICIAL_SCIPLEX_COMMIT = '079639c50811dd43a206a779ab2f0199a147c98f'
OFFICIAL_VALID_CELLS_SHA256 = 'fd3735ab48d976d09f35bb510023e1ca9cd391e6f5dfa2bd04e078e66b3c1e9c'
EXPECTED_SOURCE_CELLS = 799317
EXPECTED_METADATA_COMPLETE_CELLS = 762795
EXPECTED_OFFICIAL_HASH_PASS_CELLS = 649340
EXPECTED_PRE_TIMEPOINT_TASK_VALID_CELLS = 649220
EXPECTED_TASK_VALID_CELLS = 581777
EXPECTED_EXCLUDED_72H_CELLS = 67443
EXPECTED_EXCLUDED_72H_TREATED_CELLS = 65810
EXPECTED_EXCLUDED_72H_CONTROL_CELLS = 1633
EXPECTED_EXTRA_TIMEPOINT_DRUG_CELL_COMBINATIONS = 47
STANDARD_TIME_POINT = '24'
PUBLIC_PDATA_BOUNDARY_CELL = 'E04_F10_RT_BC_110_Lig_BC_97'

def sha256_file(path: Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (block := stream.read(chunk_size)):
            digest.update(block)
    return digest.hexdigest()

def sha256_lines(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)

def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))

def read_metadata(raw_root: Path, official_valid_cells_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = pd.read_csv(raw_root / RAW_NAMES['genes'], sep='\\s+', engine='c', dtype=str)
    if list(genes.columns) != ['id', 'gene_short_name']:
        raise ValueError('Unexpected GSM4150378 gene annotation schema')
    genes.insert(0, 'source_feature_index', np.arange(len(genes), dtype=np.int32))
    genes['ensembl_id'] = genes['id'].str.replace('\\.\\d+$', '', regex=True)
    genes['species'] = np.select([genes['ensembl_id'].str.startswith('ENSG'), genes['ensembl_id'].str.startswith('ENSMUSG')], ['human', 'mouse'], default='other')
    human = genes.loc[genes['species'].eq('human')].copy().reset_index(drop=True)
    human.insert(0, 'human_gene_index', np.arange(len(human), dtype=np.int32))
    if human['ensembl_id'].duplicated().any():
        duplicated = human.loc[human['ensembl_id'].duplicated(False), 'ensembl_id'].tolist()
        raise ValueError(f'Duplicate version-stripped human Ensembl IDs: {duplicated[:10]}')
    pdata = pd.read_csv(raw_root / RAW_NAMES['pdata'], sep='\\s+', engine='c', index_col=0, quotechar='"', dtype=str, keep_default_na=False, na_values=['NA'])
    cells = pd.read_csv(raw_root / RAW_NAMES['cells'], sep='\t', header=None, names=['cell_annotation_id', 'cell_annotation_sample'], dtype=str)
    if len(pdata) != len(cells):
        raise ValueError(f'pData/cell annotation length mismatch: {len(pdata)} != {len(cells)}')
    if not np.array_equal(pdata.index.astype(str).to_numpy(), cells['cell_annotation_id'].to_numpy()):
        raise ValueError('pData row order does not match cell annotation order')
    if not np.array_equal(pdata['cell'].astype(str).to_numpy(), cells['cell_annotation_id'].to_numpy()):
        raise ValueError('pData cell column does not match cell annotation order')
    if not np.array_equal(pdata['sample'].astype(str).to_numpy(), cells['cell_annotation_sample'].to_numpy()):
        raise ValueError('pData sample column does not match cell annotation sample')
    pdata.index.name = 'cell_id'
    pdata.insert(0, 'source_cell_index', np.arange(len(pdata), dtype=np.int32))
    metadata_complete = pdata['cell_type'].isin(VALID_CELL_TYPES) & pdata['treatment'].notna() & pdata['vehicle'].isin(['TRUE', 'FALSE'])
    if sha256_file(official_valid_cells_path) != OFFICIAL_VALID_CELLS_SHA256:
        raise ValueError('Official SciPlex3 valid-cell list SHA-256 mismatch')
    official_ids = pd.read_csv(official_valid_cells_path, sep='\t', header=None, names=['cell_id'], dtype=str, keep_default_na=False)['cell_id']
    if len(official_ids) != EXPECTED_OFFICIAL_HASH_PASS_CELLS or official_ids.duplicated().any():
        raise ValueError('Official SciPlex3 valid-cell list is not 649,340 unique IDs')
    missing_official_ids = set(official_ids).difference(pdata.index.astype(str))
    if missing_official_ids:
        raise ValueError(f'Official valid-cell IDs are absent from public pData: {sorted(missing_official_ids)[:10]}')
    official_hash_pass = pdata.index.astype(str).isin(set(official_ids))
    pre_timepoint_valid = metadata_complete.to_numpy(bool) & official_hash_pass
    standard_time_point = pdata['time_point'].astype('string').eq(STANDARD_TIME_POINT).fillna(False).to_numpy(bool)
    valid = pre_timepoint_valid & standard_time_point
    if len(pdata) != EXPECTED_SOURCE_CELLS:
        raise ValueError(f'Expected {EXPECTED_SOURCE_CELLS} source cells, found {len(pdata)}')
    if int(metadata_complete.sum()) != EXPECTED_METADATA_COMPLETE_CELLS:
        raise ValueError(f'Expected {EXPECTED_METADATA_COMPLETE_CELLS} metadata-complete cells, found {int(metadata_complete.sum())}')
    if int(official_hash_pass.sum()) != EXPECTED_OFFICIAL_HASH_PASS_CELLS:
        raise ValueError('Official hash-pass membership count changed')
    if int(pre_timepoint_valid.sum()) != EXPECTED_PRE_TIMEPOINT_TASK_VALID_CELLS:
        raise ValueError(f'Expected {EXPECTED_PRE_TIMEPOINT_TASK_VALID_CELLS} pre-timepoint task-valid cells, found {int(pre_timepoint_valid.sum())}')
    excluded_timepoint = pre_timepoint_valid & ~standard_time_point
    if int(excluded_timepoint.sum()) != EXPECTED_EXCLUDED_72H_CELLS:
        raise ValueError('Unexpected number of non-24h cells after official hash filtering')
    if set(pdata.loc[excluded_timepoint, 'time_point'].astype(str)) != {'72'}:
        raise ValueError('The non-standard time-point cells are not exclusively 72h')
    excluded_treated = excluded_timepoint & pdata['vehicle'].eq('FALSE').to_numpy(bool)
    excluded_control = excluded_timepoint & pdata['vehicle'].eq('TRUE').to_numpy(bool)
    if int(excluded_treated.sum()) != EXPECTED_EXCLUDED_72H_TREATED_CELLS:
        raise ValueError('Unexpected number of excluded 72h treated cells')
    if int(excluded_control.sum()) != EXPECTED_EXCLUDED_72H_CONTROL_CELLS:
        raise ValueError('Unexpected number of excluded 72h control cells')
    extra_combinations = pdata.loc[excluded_treated, ['treatment', 'cell_type']].drop_duplicates()
    if len(extra_combinations) != EXPECTED_EXTRA_TIMEPOINT_DRUG_CELL_COMBINATIONS:
        raise ValueError('Unexpected number of extra-timepoint drug-cell combinations')
    if set(extra_combinations['cell_type'].astype(str)) != {'A549'}:
        raise ValueError('Extra 72h drug-cell combinations are not exclusively A549')
    if int(valid.sum()) != EXPECTED_TASK_VALID_CELLS:
        raise ValueError(f'Expected {EXPECTED_TASK_VALID_CELLS} task-valid cells, found {int(valid.sum())}')
    numeric = {column: pd.to_numeric(pdata[column], errors='coerce') for column in ('hash_umis_P', 'hash_umis_W', 'top_to_second_best_ratio_P', 'top_to_second_best_ratio_W', 'rt_well')}
    public_code_mask = (pdata['top_oligo_P'].notna() & pdata['top_oligo_W'].notna() & numeric['hash_umis_P'].ge(5) & numeric['hash_umis_W'].ge(5) & numeric['top_to_second_best_ratio_P'].ge(5) & numeric['top_to_second_best_ratio_W'].ge(5) & numeric['rt_well'].notna() & numeric['rt_well'].ne(1)).to_numpy(bool)
    code_only_ids = set(pdata.index[public_code_mask]).difference(official_ids)
    official_only_ids = set(official_ids).difference(pdata.index[public_code_mask])
    if code_only_ids != {PUBLIC_PDATA_BOUNDARY_CELL} or official_only_ids:
        raise ValueError('Public pData predicates no longer differ from the official list only at the known ratio boundary')
    pdata['metadata_complete_cell'] = metadata_complete.to_numpy(bool)
    pdata['official_hash_pass_cell'] = official_hash_pass
    pdata['public_pdata_hash_predicate_pass_cell'] = public_code_mask
    pdata['task_valid_before_timepoint_filter'] = pre_timepoint_valid
    pdata['standard_24h_cell'] = standard_time_point
    pdata['valid_labeled_cell'] = valid
    pdata['valid_cell_index'] = -1
    pdata.loc[valid, 'valid_cell_index'] = np.arange(int(valid.sum()), dtype=np.int32)
    return (genes, pdata)

def prepare_metadata(raw_root: Path, out_root: Path, official_valid_cells_path: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    metadata_root.mkdir(parents=True, exist_ok=True)
    (genes, pdata) = read_metadata(raw_root, official_valid_cells_path)
    human = genes.loc[genes['species'].eq('human')].copy().reset_index(drop=True)
    human.insert(0, 'human_gene_index', np.arange(len(human), dtype=np.int32))
    valid = pdata['valid_labeled_cell'].to_numpy(bool)
    valid_pdata = pdata.loc[valid].copy()
    feature_map = np.full(len(genes), -1, dtype='<i4')
    feature_map[human['source_feature_index'].to_numpy(np.int64)] = human['human_gene_index'].to_numpy(np.int32)
    feature_map.tofile(metadata_root / 'feature_to_human.i32')
    valid.astype('u1').tofile(metadata_root / 'cell_valid.u8')
    genes.to_parquet(metadata_root / 'all_features.parquet', index=False)
    human.to_parquet(metadata_root / 'human_features.parquet', index=False)
    pdata.to_parquet(metadata_root / 'all_cells_pdata.parquet', index=True)
    valid_pdata.to_parquet(metadata_root / 'valid_cells_pdata.parquet', index=True)
    treated = valid_pdata.loc[valid_pdata['vehicle'].eq('FALSE')]
    controls = valid_pdata.loc[valid_pdata['vehicle'].eq('TRUE')]
    if set(controls['treatment'].astype(str)) != {CONTROL_TREATMENT}:
        raise ValueError('Vehicle controls are not exactly treatment S0000')
    treatment_counts = valid_pdata.groupby(['cell_type', 'treatment', 'vehicle'], observed=True).size().rename('cells').reset_index()
    treatment_counts.to_csv(metadata_root / 'cell_treatment_counts.csv', index=False)
    audit = {'method_version': METHOD_VERSION, 'source_features': int(len(genes)), 'human_features': int(genes['species'].eq('human').sum()), 'mouse_features': int(genes['species'].eq('mouse').sum()), 'other_features': int(genes['species'].eq('other').sum()), 'duplicate_version_stripped_human_ensembl': int(human['ensembl_id'].duplicated().sum()), 'source_cells': int(len(pdata)), 'metadata_complete_cells': int(pdata['metadata_complete_cell'].sum()), 'official_hash_pass_cells': int(pdata['official_hash_pass_cell'].sum()), 'metadata_complete_intersect_official_hash_pass': int(pdata['task_valid_before_timepoint_filter'].sum()), 'pre_timepoint_task_valid_cells': int(pdata['task_valid_before_timepoint_filter'].sum()), 'task_valid_24h_cells': int(valid.sum()), 'standard_time_point_hours': int(STANDARD_TIME_POINT), 'excluded_72h_cells': int((pdata['task_valid_before_timepoint_filter'] & ~pdata['standard_24h_cell']).sum()), 'excluded_72h_treated_cells': int((pdata['task_valid_before_timepoint_filter'] & ~pdata['standard_24h_cell'] & pdata['vehicle'].eq('FALSE')).sum()), 'excluded_72h_control_cells': int((pdata['task_valid_before_timepoint_filter'] & ~pdata['standard_24h_cell'] & pdata['vehicle'].eq('TRUE')).sum()), 'excluded_72h_drug_cell_combinations': int(pdata.loc[pdata['task_valid_before_timepoint_filter'] & ~pdata['standard_24h_cell'] & pdata['vehicle'].eq('FALSE'), ['treatment', 'cell_type']].drop_duplicates().shape[0]), 'metadata_complete_minus_official_hash_pass': int((pdata['metadata_complete_cell'] & ~pdata['official_hash_pass_cell']).sum()), 'official_hash_pass_minus_metadata_complete': int((pdata['official_hash_pass_cell'] & ~pdata['metadata_complete_cell']).sum()), 'neither_metadata_complete_nor_official_hash_pass': int((~pdata['official_hash_pass_cell'] & ~pdata['metadata_complete_cell']).sum()), 'official_valid_cells_path': str(official_valid_cells_path), 'official_valid_cells_commit': OFFICIAL_SCIPLEX_COMMIT, 'official_valid_cells_sha256': OFFICIAL_VALID_CELLS_SHA256, 'public_pdata_code_predicate_pass_cells': int(pdata['public_pdata_hash_predicate_pass_cell'].sum()), 'public_pdata_boundary_cell_excluded_by_official_list': PUBLIC_PDATA_BOUNDARY_CELL, 'criterion_failures_within_metadata_complete': {'top_oligo_P_is_na': int((pdata['metadata_complete_cell'] & pdata['top_oligo_P'].isna()).sum()), 'top_oligo_W_is_na': int((pdata['metadata_complete_cell'] & pdata['top_oligo_W'].isna()).sum()), 'hash_umis_P_lt_5_or_na': int((pdata['metadata_complete_cell'] & ~pd.to_numeric(pdata['hash_umis_P'], errors='coerce').ge(5)).sum()), 'hash_umis_W_lt_5_or_na': int((pdata['metadata_complete_cell'] & ~pd.to_numeric(pdata['hash_umis_W'], errors='coerce').ge(5)).sum()), 'top_to_second_best_ratio_P_lt_5_or_na': int((pdata['metadata_complete_cell'] & ~pd.to_numeric(pdata['top_to_second_best_ratio_P'], errors='coerce').ge(5)).sum()), 'top_to_second_best_ratio_W_lt_5_or_na': int((pdata['metadata_complete_cell'] & ~pd.to_numeric(pdata['top_to_second_best_ratio_W'], errors='coerce').ge(5)).sum()), 'rt_well_eq_1_or_na': int((pdata['metadata_complete_cell'] & ~(pd.to_numeric(pdata['rt_well'], errors='coerce').notna() & pd.to_numeric(pdata['rt_well'], errors='coerce').ne(1))).sum())}, 'valid_labeled_cells': int(valid.sum()), 'excluded_cells_total': int((~valid).sum()), 'valid_cells_by_type': {str(k): int(v) for (k, v) in valid_pdata['cell_type'].value_counts().sort_index().items()}, 'vehicle_cells_by_type': {str(k): int(v) for (k, v) in controls['cell_type'].value_counts().sort_index().items()}, 'treated_cells_by_type': {str(k): int(v) for (k, v) in treated['cell_type'].value_counts().sort_index().items()}, 'noncontrol_treatments': int(treated['treatment'].nunique()), 'treatments_per_cell': {str(cell): int(group.loc[group['vehicle'].eq('FALSE'), 'treatment'].nunique()) for (cell, group) in valid_pdata.groupby('cell_type', observed=True)}, 'doses': sorted(treated['dose'].dropna().astype(str).unique().tolist(), key=float), 'time_points': sorted(treated['time_point'].dropna().astype(str).unique().tolist(), key=float), 'cell_order_sha256': sha256_lines(pdata.index.astype(str)), 'human_ensembl_order_sha256': sha256_lines(human['ensembl_id'])}
    if audit['noncontrol_treatments'] != 188:
        raise ValueError(f"Expected 188 SciPlex3 drugs, found {audit['noncontrol_treatments']}")
    if audit['time_points'] != [STANDARD_TIME_POINT]:
        raise ValueError(f'Expected only the standard {STANDARD_TIME_POINT}h time point')
    if set(audit['treatments_per_cell'].values()) != {188}:
        raise ValueError('Every cell line must retain all 188 drugs after the 24h filter')
    treated_coverage = valid_pdata.loc[valid_pdata['vehicle'].eq('FALSE')]
    if not treated_coverage.groupby(['treatment', 'cell_type'], observed=True)['dose'].nunique().eq(4).all():
        raise ValueError('Every 24h drug-cell combination must retain four doses')
    if not treated_coverage.groupby(['treatment', 'cell_type'], observed=True)['replicate'].nunique().eq(2).all():
        raise ValueError('Every 24h drug-cell combination must retain two replicates')
    write_json(metadata_root / 'metadata_audit.json', audit)
    return audit

def compile_streamer(source: Path, binary: Path) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = ['g++', '-O3', '-std=c++17', str(source), '-lz', '-o', str(binary)]
    subprocess.run(command, check=True)

def run_streamer(*, binary: Path, mode: str, raw_root: Path, out_root: Path, extra: list[str] | None=None) -> None:
    metadata_root = out_root / 'metadata'
    genes = pd.read_parquet(metadata_root / 'all_features.parquet', columns=['species'])
    command = [str(binary), '--mode', mode, '--matrix', str(raw_root / RAW_NAMES['matrix']), '--feature-map', str(metadata_root / 'feature_to_human.i32'), '--cell-valid', str(metadata_root / 'cell_valid.u8'), '--out-dir', str(out_root / 'intermediate'), '--n-cells', str(len(pd.read_parquet(metadata_root / 'all_cells_pdata.parquet', columns=['source_cell_index']))), '--n-features', str(len(genes)), '--n-human', str(int(genes['species'].eq('human').sum()))]
    command.extend(extra or [])
    environment = dict(os.environ)
    environment.update({'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4'})
    subprocess.run(['nice', '-n', '10', 'ionice', '-c2', '-n7', *command], check=True, env=environment)

def compute_clip_values(out_root: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    intermediate = out_root / 'intermediate'
    cells = pd.read_parquet(metadata_root / 'all_cells_pdata.parquet', columns=['valid_labeled_cell'])
    n_obs = int(cells['valid_labeled_cell'].sum())
    human = pd.read_parquet(metadata_root / 'human_features.parquet')
    n_genes = len(human)
    sums = np.fromfile(intermediate / 'raw_sum.f64', dtype='<f8', count=n_genes)
    sums_sq = np.fromfile(intermediate / 'raw_sumsq.f64', dtype='<f8', count=n_genes)
    if len(sums) != n_genes or len(sums_sq) != n_genes:
        raise ValueError('Incomplete pass1 gene statistics')
    means = sums / n_obs
    variances = (sums_sq - sums * sums / n_obs) / (n_obs - 1)
    variances = np.maximum(variances, 0.0)
    nonconstant = variances > 0
    if int(nonconstant.sum()) < N_HVG:
        raise ValueError('Too few nonconstant genes for 2,000 HVGs')
    model = loess(np.log10(means[nonconstant]), np.log10(variances[nonconstant]), span=0.3, degree=2)
    model.fit()
    fitted = model.outputs.fitted_values
    regularized_std = np.ones(n_genes, dtype='<f8')
    regularized_std[nonconstant] = np.sqrt(10 ** fitted)
    clip = regularized_std * np.sqrt(n_obs) + means
    clip.astype('<f8').tofile(intermediate / 'clip_value.f64')
    table = human[['human_gene_index', 'ensembl_id', 'gene_short_name', 'source_feature_index']].copy()
    table['raw_mean'] = means
    table['raw_variance'] = variances
    table['seurat_v3_regularized_std'] = regularized_std
    table['seurat_v3_clip_value'] = clip
    table.to_parquet(metadata_root / 'human_gene_pass1_stats.parquet', index=False)
    audit = {'n_obs': n_obs, 'n_human_genes': n_genes, 'nonconstant_raw_genes': int(nonconstant.sum()), 'hvg_flavor': 'seurat_v3', 'hvg_selection_layer': 'human-only raw counts', 'loess_span': 0.3, 'loess_degree': 2, 'output_expression': 'human-library-size CP10K then log1p'}
    write_json(intermediate / 'clip_audit.json', audit)
    return audit

def select_hvgs(out_root: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    intermediate = out_root / 'intermediate'
    cells = pd.read_parquet(metadata_root / 'all_cells_pdata.parquet', columns=['valid_labeled_cell'])
    n_obs = int(cells['valid_labeled_cell'].sum())
    human = pd.read_parquet(metadata_root / 'human_gene_pass1_stats.parquet')
    n_genes = len(human)
    raw_mean = human['raw_mean'].to_numpy(float)
    regularized_std = human['seurat_v3_regularized_std'].to_numpy(float)
    clip_sum = np.fromfile(intermediate / 'clip_sum.f64', dtype='<f8', count=n_genes)
    clip_sumsq = np.fromfile(intermediate / 'clip_sumsq.f64', dtype='<f8', count=n_genes)
    log_sum = np.fromfile(intermediate / 'log_sum.f64', dtype='<f8', count=n_genes)
    log_sumsq = np.fromfile(intermediate / 'log_sumsq.f64', dtype='<f8', count=n_genes)
    if any((len(x) != n_genes for x in (clip_sum, clip_sumsq, log_sum, log_sumsq))):
        raise ValueError('Incomplete pass2 gene statistics')
    normalized_variance = (n_obs * raw_mean ** 2 + clip_sumsq - 2 * clip_sum * raw_mean) / ((n_obs - 1) * regularized_std ** 2)
    normalized_variance = np.nan_to_num(normalized_variance, nan=-np.inf, neginf=-np.inf)
    normalized_ranks = np.argsort(np.argsort(-normalized_variance))
    primary = np.argsort(normalized_ranks)[:N_HVG]
    log_mean = log_sum / n_obs
    log_variance = (log_sumsq - log_sum * log_sum / n_obs) / (n_obs - 1)
    log_variance = np.maximum(log_variance, 0.0)
    table = human.copy()
    table['seurat_v3_normalized_variance'] = normalized_variance
    table['log_cp10k_mean'] = log_mean
    table['log_cp10k_variance'] = log_variance
    table['hvg_rank'] = pd.array([pd.NA] * n_genes, dtype='Int32')
    table.loc[primary, 'hvg_rank'] = np.arange(1, N_HVG + 1, dtype=np.int32)
    table['is_core_hvg'] = table['hvg_rank'].notna()
    table.to_parquet(metadata_root / 'human_gene_hvg_stats.parquet', index=False)
    core = table.loc[primary].copy()
    core.to_csv(metadata_root / 'core_hvg2000.csv', index=False)
    audit = {'primary_method': 'scanpy-compatible Seurat v3 variance stabilization on human-only raw counts', 'primary_gene_count': N_HVG, 'primary_ensembl_order_sha256': sha256_lines(core['ensembl_id']), 'output_normalization': 'log1p(raw_count / human_only_library_size * 10000)'}
    write_json(metadata_root / 'hvg_audit.json', audit)
    return audit

def build_lincs_mapping(out_root: Path, lincs_gene_info: Path, ncbi_gene_info: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    human = pd.read_parquet(metadata_root / 'human_gene_hvg_stats.parquet')
    lincs = pd.read_csv(lincs_gene_info, sep='\t', compression='gzip', dtype=str)
    landmark = lincs.loc[lincs['pr_is_lm'].astype(str).eq('1')].copy().reset_index(drop=True)
    if len(landmark) != N_LINCS or landmark['pr_gene_id'].duplicated().any():
        raise ValueError(f'Official LINCS landmark set is not 978 unique Entrez genes: {len(landmark)}')
    ncbi = pd.read_csv(ncbi_gene_info, sep='\t', compression='gzip', dtype=str, keep_default_na=False)
    if 'GeneID' not in ncbi or 'dbXrefs' not in ncbi:
        raise ValueError('Unexpected NCBI Homo_sapiens.gene_info schema')
    entrez_to_ensembl: dict[str, list[str]] = {}
    entrez_to_symbol: dict[str, str] = {}
    for row in ncbi.itertuples(index=False):
        gene_id = str(getattr(row, 'GeneID'))
        dbxrefs = str(getattr(row, 'dbXrefs'))
        ensembl = sorted({token.split(':', 1)[1] for token in dbxrefs.split('|') if token.startswith('Ensembl:ENSG')})
        entrez_to_ensembl[gene_id] = ensembl
        entrez_to_symbol[gene_id] = str(getattr(row, 'Symbol'))
    measured = set(human['ensembl_id'].astype(str))
    symbol_to_ensembl = human.groupby('gene_short_name', dropna=False)['ensembl_id'].agg(lambda values: sorted(set(map(str, values)))).to_dict()
    rows: list[dict[str, Any]] = []
    for (index, row) in landmark.iterrows():
        entrez = str(row['pr_gene_id'])
        official_symbol = str(row['pr_gene_symbol'])
        exact = [value for value in entrez_to_ensembl.get(entrez, []) if value in measured]
        if len(exact) == 1:
            mapped = exact[0]
            method = 'exact_entrez_to_ncbi_ensembl'
        elif len(exact) > 1:
            raise ValueError(f'Ambiguous measured Ensembl mappings for LINCS Entrez {entrez}: {exact}')
        else:
            candidates = symbol_to_ensembl.get(official_symbol, [])
            if len(candidates) != 1:
                raise ValueError(f'Unresolved LINCS landmark {entrez}/{official_symbol}: {candidates}')
            mapped = candidates[0]
            method = 'fallback_exact_lincs_symbol'
        measured_row = human.loc[human['ensembl_id'].eq(mapped)].iloc[0]
        rows.append({'lincs_landmark_order': index + 1, 'lincs_entrez_id': entrez, 'lincs_symbol_2017': official_symbol, 'ncbi_current_symbol': entrez_to_symbol.get(entrez, ''), 'ensembl_id': mapped, 'sciplex_symbol': measured_row['gene_short_name'], 'human_gene_index': int(measured_row['human_gene_index']), 'source_feature_index': int(measured_row['source_feature_index']), 'mapping_method': method, 'is_core_hvg': bool(measured_row['is_core_hvg'])})
    mapping = pd.DataFrame(rows)
    if len(mapping) != N_LINCS or mapping['ensembl_id'].duplicated().any():
        raise ValueError('LINCS-to-SciPlex mapping is not a one-to-one 978-gene registry')
    mapping['gene_role'] = np.where(mapping['is_core_hvg'], 'core_hvg_lincs', 'extra_lm_not_hvg')
    mapping.to_csv(metadata_root / 'lincs978_to_sciplex_human.csv', index=False)
    extra = mapping.loc[~mapping['is_core_hvg']].copy().reset_index(drop=True)
    extra.insert(0, 'extra_lm_index', np.arange(len(extra), dtype=np.int16))
    extra.to_csv(metadata_root / 'extra_lm_not_hvg.csv', index=False)
    audit = {'official_lincs_landmarks': N_LINCS, 'mapped_to_measured_sciplex_human': int(len(mapping)), 'mapping_method_counts': {str(k): int(v) for (k, v) in mapping['mapping_method'].value_counts().items()}, 'landmarks_in_core_hvg': int(mapping['is_core_hvg'].sum()), 'extra_lm_not_hvg_count': int((~mapping['is_core_hvg']).sum()), 'extra_lm_ensembl_order_sha256': sha256_lines(extra['ensembl_id']), 'response_blind_selection': True, 'forced_to_304': False, 'lincs_source_sha256': sha256_file(lincs_gene_info), 'ncbi_gene_info_sha256': sha256_file(ncbi_gene_info)}
    write_json(metadata_root / 'lincs_mapping_audit.json', audit)
    return audit

def prepare_selection(out_root: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    genes = pd.read_parquet(metadata_root / 'all_features.parquet')
    core = pd.read_csv(metadata_root / 'core_hvg2000.csv', dtype={'ensembl_id': str})
    extra = pd.read_csv(metadata_root / 'extra_lm_not_hvg.csv', dtype={'ensembl_id': str})
    if set(core['ensembl_id']).intersection(extra['ensembl_id']):
        raise ValueError('Core HVG and extra LINCS sets overlap')
    union = pd.concat([core.assign(gene_role='core_hvg').sort_values('hvg_rank', kind='stable'), extra.rename(columns={'sciplex_symbol': 'gene_short_name'}).assign(gene_role='extra_lm_not_hvg').sort_values('lincs_landmark_order', kind='stable')], ignore_index=True, sort=False)
    union.insert(0, 'union_gene_index', np.arange(len(union), dtype=np.int16))
    union.to_parquet(metadata_root / 'truth_union_genes.parquet', index=False)
    union.to_csv(metadata_root / 'truth_union_genes.csv', index=False)
    by_ensembl = dict(zip(union['ensembl_id'].astype(str), union['union_gene_index'].astype(int)))
    selection = np.full(len(genes), -1, dtype='<i4')
    for row in genes.itertuples(index=False):
        if str(row.ensembl_id) in by_ensembl:
            selection[int(row.source_feature_index)] = by_ensembl[str(row.ensembl_id)]
    if int((selection >= 0).sum()) != len(union):
        raise ValueError('Feature selection map does not contain each union gene exactly once')
    selection.tofile(metadata_root / 'feature_to_truth_union.i32')
    audit = {'core_hvg_count': int(len(core)), 'extra_lm_not_hvg_count': int(len(extra)), 'truth_union_count': int(len(union)), 'core_extra_overlap': 0, 'truth_union_ensembl_order_sha256': sha256_lines(union['ensembl_id'])}
    write_json(metadata_root / 'selection_audit.json', audit)
    return audit

def load_selected_csr(out_root: Path, *, counts: bool=False) -> sparse.csr_matrix:
    intermediate = out_root / 'intermediate'
    summary = json.loads((intermediate / 'extract_summary.json').read_text(encoding='utf-8'))
    nnz = int(summary['selected_nnz'])
    n_obs = int(summary['valid_cells'])
    n_vars = int(summary['selected_genes'])
    indptr = np.fromfile(intermediate / 'selected_indptr.u64', dtype='<u8', count=n_obs + 1)
    indices = np.fromfile(intermediate / 'selected_indices.u16', dtype='<u2', count=nnz).astype(np.int32)
    if counts:
        data = np.fromfile(intermediate / 'selected_counts.u32', dtype='<u4', count=nnz)
    else:
        data = np.fromfile(intermediate / 'selected_log_data.f32', dtype='<f4', count=nnz)
    if len(indptr) != n_obs + 1 or len(indices) != nnz or len(data) != nnz:
        raise ValueError('Incomplete selected CSR binary arrays')
    result = sparse.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
    result.sort_indices()
    return result

def assemble_h5ad(out_root: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    valid = pd.read_parquet(metadata_root / 'valid_cells_pdata.parquet')
    union = pd.read_parquet(metadata_root / 'truth_union_genes.parquet')
    matrix = load_selected_csr(out_root, counts=False)
    if matrix.shape != (len(valid), len(union)):
        raise ValueError(f'Selected matrix shape mismatch: {matrix.shape}')
    if list(union.iloc[:N_HVG]['gene_role'].unique()) != ['core_hvg']:
        raise ValueError('First 2,000 truth-union genes are not exclusively core HVGs')
    core_matrix = matrix[:, :N_HVG].copy()
    library_all = np.fromfile(out_root / 'intermediate' / 'library_size.u64', dtype='<u8')
    source_indices = valid['source_cell_index'].to_numpy(np.int64)
    obs_columns = ['source_cell_index', 'sample', 'n.umi', 'cell_type', 'replicate', 'time_point', 'drug_dose', 'catalog_number', 'vehicle', 'dose', 'treatment', 'pathway_level_1', 'pathway_level_2', 'product_name', 'target', 'pathway', 'culture_plate']
    obs = valid[obs_columns].copy()
    obs['human_library_size'] = library_all[source_indices]
    obs['drug'] = obs['product_name'].where(obs['vehicle'].eq('FALSE'), 'vehicle')
    obs['perturbation_id'] = obs['treatment']
    for column in ['n.umi', 'time_point', 'dose']:
        obs[column] = pd.to_numeric(obs[column], errors='coerce')
    for column in obs.columns:
        if obs[column].dtype == object and obs[column].nunique(dropna=False) < 10000:
            obs[column] = obs[column].astype('category')
    obs.index = valid.index.astype(str)
    obs.index.name = 'cell_id'
    var = union.iloc[:N_HVG][['ensembl_id', 'gene_short_name', 'source_feature_index', 'human_gene_index', 'hvg_rank']].copy()
    var = var.rename(columns={'gene_short_name': 'gene_symbol'})
    landmark_ensembl = set(pd.read_csv(metadata_root / 'lincs978_to_sciplex_human.csv')['ensembl_id'].astype(str))
    var['highly_variable'] = True
    var['is_lincs_landmark'] = var['ensembl_id'].astype(str).isin(landmark_ensembl)
    var.index = var['ensembl_id'].astype(str)
    var.index.name = 'ensembl_id_index'
    adata = ad.AnnData(X=core_matrix, obs=obs, var=var)
    adata.uns['build'] = {'method_version': METHOD_VERSION, 'source_accession': 'GSE139944/GSM4150378', 'species_filter': 'raw feature id starts with ENSG; version suffix stripped', 'normalization': 'log1p(raw_count / human_only_library_size * 10000)', 'hvg_method': 'Seurat v3 on human-only raw counts, 2000 genes', 'cell_filter': 'official valid_cells.tsv intersect task-complete metadata and time_point == 24h', 'input_gene_scope': 'core_hvg_only', 'extra_truth_genes_present_in_X': False, 'extra_truth_registry': 'metadata/extra_lm_not_hvg.csv'}
    h5ad_path = out_root / 'sciplex3_core_hvg2000_input.h5ad'
    adata.write_h5ad(h5ad_path, compression='gzip', compression_opts=4)
    del adata, core_matrix, matrix
    reopened = ad.read_h5ad(h5ad_path, backed='r')
    try:
        shape = [int(reopened.n_obs), int(reopened.n_vars)]
        extra_in_input = bool(set(union.iloc[N_HVG:]['ensembl_id'].astype(str)).intersection(reopened.var_names.astype(str)))
        if shape != [len(valid), N_HVG] or extra_in_input:
            raise ValueError('Reopened input H5AD violates the core-only boundary')
    finally:
        reopened.file.close()
    audit = {'path': str(h5ad_path), 'sha256': sha256_file(h5ad_path), 'shape': shape, 'matrix_storage': 'CSR float32 log1p CP10K', 'var_primary_key': 'version-stripped Ensembl gene ID', 'extra_lm_genes_in_input': 0, 'input_only_contains_core_hvg': True}
    write_json(out_root / 'input_h5ad_audit.json', audit)
    return audit

def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues, kind='stable')
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result

def scanpy_wilcoxon_vs_control(values: sparse.csc_matrix, groups: np.ndarray, group_names: list[str], control_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_groups = len(group_names)
    n_genes = values.shape[1]
    group_index = {name: index for (index, name) in enumerate(group_names)}
    group_codes = np.array([group_index.get(str(value), -1) for value in groups], dtype=np.int16)
    group_sizes = np.bincount(group_codes[group_codes >= 0], minlength=n_groups).astype(np.int64)
    n_control = int(control_mask.sum())
    if n_control == 0 or (group_sizes == 0).any():
        raise ValueError('A cell line lacks control or treated cells for one or more drugs')
    pvalues = np.empty((n_groups, n_genes), dtype=np.float64)
    zscores = np.empty_like(pvalues)
    logfoldchanges = np.empty_like(pvalues)
    epsilon = 1e-09
    for gene in range(n_genes):
        (start, stop) = (values.indptr[gene], values.indptr[gene + 1])
        rows = values.indices[start:stop]
        data = values.data[start:stop].astype(np.float64, copy=False)
        control_values = np.sort(data[control_mask[rows]])
        control_positive_count = len(control_values)
        control_zeros = n_control - control_positive_count
        treated = ~control_mask[rows]
        treated_rows = rows[treated]
        treated_values = data[treated]
        treated_codes = group_codes[treated_rows]
        keep = treated_codes >= 0
        treated_values = treated_values[keep]
        treated_codes = treated_codes[keep]
        positive_counts = np.bincount(treated_codes, minlength=n_groups)
        positive_sums = np.bincount(treated_codes, weights=treated_values, minlength=n_groups)
        less = np.searchsorted(control_values, treated_values, side='left')
        right = np.searchsorted(control_values, treated_values, side='right')
        contributions = control_zeros + less + 0.5 * (right - less)
        u_positive = np.bincount(treated_codes, weights=contributions, minlength=n_groups)
        treated_zeros = group_sizes - positive_counts
        u = u_positive + treated_zeros * (0.5 * control_zeros)
        expected = group_sizes * n_control / 2.0
        std = np.sqrt(group_sizes * n_control * (group_sizes + n_control + 1) / 12.0)
        z = np.divide(u - expected, std, out=np.zeros(n_groups, dtype=float), where=std > 0)
        zscores[:, gene] = z
        pvalues[:, gene] = 2.0 * ndtr(-np.abs(z))
        control_mean = control_values.sum() / n_control
        treated_mean = positive_sums / group_sizes
        logfoldchanges[:, gene] = np.log2((np.expm1(treated_mean) + epsilon) / (np.expm1(control_mean) + epsilon))
        if (gene + 1) % 250 == 0:
            print(f'  Wilcoxon genes {gene + 1}/{n_genes}', flush=True)
    return (zscores, pvalues, logfoldchanges)

def generate_labels(out_root: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    labels_root = out_root / 'labels'
    stats_root = out_root / 'truth_full_stats'
    labels_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)
    valid = pd.read_parquet(metadata_root / 'valid_cells_pdata.parquet')
    union = pd.read_parquet(metadata_root / 'truth_union_genes.parquet')
    matrix = load_selected_csr(out_root, counts=False).tocsc()
    if matrix.shape != (len(valid), len(union)):
        raise ValueError('Truth-union matrix/metadata mismatch')
    all_audits: dict[str, Any] = {}
    for cell in VALID_CELL_TYPES:
        print(f'Generating expression labels for {cell}', flush=True)
        cell_mask = valid['cell_type'].astype(str).eq(cell).to_numpy()
        obs = valid.loc[cell_mask]
        cell_matrix = matrix[cell_mask].tocsc()
        control = obs['vehicle'].astype(str).eq('TRUE').to_numpy()
        treated = obs.loc[~control]
        treatment_to_drug = treated[['treatment', 'product_name']].drop_duplicates().set_index('treatment')['product_name'].astype(str).to_dict()
        group_names = sorted(treatment_to_drug)
        if len(group_names) != 188:
            raise ValueError(f'{cell}: expected 188 drugs, found {len(group_names)}')
        (zscores, pvalues, logfoldchanges) = scanpy_wilcoxon_vs_control(cell_matrix, obs['treatment'].astype(str).to_numpy(), group_names, control)
        adjusted = np.vstack([bh_adjust(row) for row in pvalues])
        frames = []
        gene_columns = union[['union_gene_index', 'ensembl_id', 'gene_short_name', 'gene_role']].copy()
        gene_columns = gene_columns.rename(columns={'gene_short_name': 'gene'})
        for (group_index, treatment) in enumerate(group_names):
            frame = gene_columns.copy()
            frame.insert(0, 'cell_line', cell)
            frame.insert(1, 'pert', treatment)
            frame.insert(2, 'drug', treatment_to_drug[treatment])
            frame['scores'] = zscores[group_index]
            frame['logfoldchanges'] = logfoldchanges[group_index]
            frame['pvals'] = pvalues[group_index]
            frame['pvals_adj'] = adjusted[group_index]
            frame['de_label_strict'] = frame['pvals_adj'].lt(0.05) & frame['logfoldchanges'].abs().gt(0.25)
            frame['direction_label'] = pd.Series(pd.NA, index=frame.index, dtype='Int8')
            de_rows = frame['de_label_strict']
            frame.loc[de_rows, 'direction_label'] = frame.loc[de_rows, 'logfoldchanges'].gt(0).astype(np.int8)
            frames.append(frame.sort_values('scores', ascending=False, kind='stable').reset_index(drop=True))
        stats = pd.concat(frames, ignore_index=True)
        stats.to_parquet(stats_root / f'{cell}_HVG2000_plus_ExtraLM_stats.parquet', index=False)
        stats.loc[stats['gene_role'].eq('extra_lm_not_hvg')].to_parquet(stats_root / f'{cell}_ExtraLM_only_stats.parquet', index=False)
        positives = stats.loc[stats['de_label_strict']].copy()
        positives['label'] = 1
        negatives = pd.concat([group.sample(n=min(200, len(group)), random_state=SEED) for (_, group) in stats.loc[stats['pvals'].gt(0.1)].groupby('pert', sort=False)], ignore_index=True)
        negatives['label'] = 0
        de = pd.concat([positives, negatives], ignore_index=True)
        de = de[['pert', 'drug', 'gene', 'ensembl_id', 'gene_role', 'label']]
        de.to_csv(labels_root / f'{cell}_DE.csv', index=False)
        de.loc[de['gene_role'].eq('extra_lm_not_hvg')].to_csv(labels_root / f'{cell}_ExtraLM_DE.csv', index=False)
        direction = positives.copy()
        direction['label'] = direction['logfoldchanges'].gt(0).astype(np.int8)
        direction = direction[['pert', 'drug', 'gene', 'ensembl_id', 'gene_role', 'label']]
        direction.to_csv(labels_root / f'{cell}_DIR.csv', index=False)
        direction.loc[direction['gene_role'].eq('extra_lm_not_hvg')].to_csv(labels_root / f'{cell}_ExtraLM_DIR.csv', index=False)
        all_audits[cell] = {'cells': int(cell_mask.sum()), 'control_cells': int(control.sum()), 'treated_cells': int((~control).sum()), 'drugs': len(group_names), 'truth_genes': int(len(union)), 'core_hvg_genes': N_HVG, 'extra_lm_genes': int(len(union) - N_HVG), 'full_stats_rows': int(len(stats)), 'de_positive_rows': int(len(positives)), 'de_sampled_negative_rows': int(len(negatives)), 'de_extra_lm_rows': int(de['gene_role'].eq('extra_lm_not_hvg').sum()), 'dir_extra_lm_rows': int(direction['gene_role'].eq('extra_lm_not_hvg').sum()), 'split_created': False, 'test': 'Wilcoxon rank-sum, Scanpy default tie_correct=False', 'multiple_testing': 'Benjamini-Hochberg within drug across H union ExtraLM', 'de_rule': 'abs(logfoldchanges)>0.25 and pvals_adj<0.05', 'negative_rule': 'pvals>0.1; at most 200 per drug; random_state=42', 'direction_rule': 'sign(logfoldchanges) among strict DE positives', 'dose_policy': 'all available doses and replicates pooled by treatment, matching public prepare.py'}
        write_json(labels_root / f'{cell}_label_audit.json', all_audits[cell])
        del cell_matrix, zscores, pvalues, adjusted, logfoldchanges, stats, positives, negatives, de, direction
    del matrix
    write_json(labels_root / 'label_audit_all_cells.json', all_audits)
    return all_audits

def run_final_audit(out_root: Path) -> dict[str, Any]:
    metadata_root = out_root / 'metadata'
    intermediate = out_root / 'intermediate'
    labels_root = out_root / 'labels'
    stats_root = out_root / 'truth_full_stats'
    pass_audits = [json.loads((intermediate / f'{stage}_audit.json').read_text(encoding='utf-8')) for stage in ('pass1', 'pass2', 'extract')]
    conserved_fields = ['coordinate_rows', 'human_coordinate_rows', 'valid_human_coordinate_rows', 'human_umis', 'valid_human_umis', 'duplicate_coordinates', 'max_feature_index', 'max_cell_index']
    for field in conserved_fields:
        if len({audit[field] for audit in pass_audits}) != 1:
            raise ValueError(f'Streaming passes disagree on {field}')
    cells = pd.read_parquet(metadata_root / 'valid_cells_pdata.parquet')
    union = pd.read_parquet(metadata_root / 'truth_union_genes.parquet')
    core = pd.read_csv(metadata_root / 'core_hvg2000.csv')
    extra = pd.read_csv(metadata_root / 'extra_lm_not_hvg.csv')
    if len(cells) != EXPECTED_TASK_VALID_CELLS or len(core) != N_HVG:
        raise ValueError('Frozen SciPlex3 dimensions changed')
    if len(union) != N_HVG + len(extra):
        raise ValueError('Truth-union size does not equal CoreHVG plus current Extra-LM')
    if set(core['ensembl_id'].astype(str)).intersection(extra['ensembl_id'].astype(str)):
        raise ValueError('Extra LINCS genes leaked into the core registry')
    summary = json.loads((intermediate / 'extract_summary.json').read_text(encoding='utf-8'))
    nnz = int(summary['selected_nnz'])
    indptr = np.fromfile(intermediate / 'selected_indptr.u64', dtype='<u8')
    indices = np.memmap(intermediate / 'selected_indices.u16', dtype='<u2', mode='r')
    counts = np.memmap(intermediate / 'selected_counts.u32', dtype='<u4', mode='r')
    logged = np.memmap(intermediate / 'selected_log_data.f32', dtype='<f4', mode='r')
    if len(indptr) != len(cells) + 1 or int(indptr[-1]) != nnz:
        raise ValueError('Selected CSR indptr is inconsistent')
    if len(indices) != nnz or len(counts) != nnz or len(logged) != nnz:
        raise ValueError('Selected CSR data lengths are inconsistent')
    if int(indices.max()) >= len(union) or int(indices.min()) < 0:
        raise ValueError('Selected CSR contains an invalid column')
    if int(counts.min()) <= 0 or not np.isfinite(logged).all() or float(logged.min()) <= 0:
        raise ValueError('Selected CSR contains invalid counts/log values')
    library = np.fromfile(intermediate / 'library_size.u64', dtype='<u8')
    rng = np.random.default_rng(SEED)
    sample_positions = np.sort(rng.choice(nnz, size=10000, replace=False))
    selected_rows = np.searchsorted(indptr, sample_positions, side='right') - 1
    source_rows = cells['source_cell_index'].to_numpy(np.int64)[selected_rows]
    expected_log = np.log1p(counts[sample_positions].astype(float) * 10000 / library[source_rows])
    maximum_log_error = float(np.max(np.abs(expected_log - logged[sample_positions].astype(float))))
    if maximum_log_error > 3e-06:
        raise ValueError(f'Selected CP10K/log1p values fail recomputation: {maximum_log_error}')
    if int(library[cells['source_cell_index'].to_numpy(np.int64)].sum()) != pass_audits[0]['valid_human_umis']:
        raise ValueError('Human-only library sizes do not conserve valid UMI count')
    h5ad_path = out_root / 'sciplex3_core_hvg2000_input.h5ad'
    adata = ad.read_h5ad(h5ad_path, backed='r')
    try:
        if adata.shape != (len(cells), N_HVG):
            raise ValueError('Input H5AD dimensions changed')
        if adata.var_names.astype(str).tolist() != core['ensembl_id'].astype(str).tolist():
            raise ValueError('Input H5AD var order differs from core HVG registry')
        if set(extra['ensembl_id'].astype(str)).intersection(adata.var_names.astype(str)):
            raise ValueError('Extra LINCS Ensembl IDs are present in input H5AD')
        row_sample = np.sort(rng.choice(len(cells), size=64, replace=False))
        h5_rows = adata.X[row_sample, :]
        if not sparse.issparse(h5_rows):
            h5_rows = sparse.csr_matrix(h5_rows)
        h5_rows = h5_rows.tocsr()
        for (local_index, selected_row) in enumerate(row_sample):
            (start, stop) = (int(indptr[selected_row]), int(indptr[selected_row + 1]))
            mask = indices[start:stop] < N_HVG
            expected = sparse.csr_matrix((np.asarray(logged[start:stop][mask]), np.asarray(indices[start:stop][mask], dtype=np.int32), np.array([0, int(mask.sum())], dtype=np.int64)), shape=(1, N_HVG))
            expected.sort_indices()
            difference = h5_rows[local_index] - expected
            if difference.nnz and float(np.max(np.abs(difference.data))) > 1e-07:
                raise ValueError('Input H5AD values differ from the streamed core projection')
    finally:
        adata.file.close()
    label_checks: dict[str, Any] = {}
    for cell in VALID_CELL_TYPES:
        stats = pd.read_parquet(stats_root / f'{cell}_HVG2000_plus_ExtraLM_stats.parquet')
        de = pd.read_csv(labels_root / f'{cell}_DE.csv')
        direction = pd.read_csv(labels_root / f'{cell}_DIR.csv')
        if len(stats) != 188 * len(union) or stats.duplicated(['pert', 'ensembl_id']).any():
            raise ValueError(f'{cell}: incomplete or duplicate truth grid')
        if 'split' in stats or 'split' in de or 'split' in direction:
            raise ValueError(f'{cell}: an unauthorized train/test split was created')
        if not np.isfinite(stats[['scores', 'logfoldchanges', 'pvals', 'pvals_adj']].to_numpy()).all():
            raise ValueError(f'{cell}: nonfinite DE statistics')
        expected_de = stats['pvals_adj'].lt(0.05) & stats['logfoldchanges'].abs().gt(0.25)
        if not np.array_equal(expected_de.to_numpy(), stats['de_label_strict'].to_numpy(bool)):
            raise ValueError(f'{cell}: strict DE flags disagree with thresholds')
        negative_counts = de.loc[de['label'].eq(0)].groupby('pert').size()
        if len(negative_counts) != 188 or not negative_counts.eq(200).all():
            raise ValueError(f'{cell}: Negative sampling is not 200 per drug')
        if not de.loc[de['label'].eq(0), 'gene_role'].isin(['core_hvg', 'extra_lm_not_hvg']).all():
            raise ValueError(f'{cell}: unknown negative-sample gene role')
        if not direction['label'].isin([0, 1]).all():
            raise ValueError(f'{cell}: invalid direction label')
        label_checks[cell] = {'full_grid_rows': int(len(stats)), 'de_rows': int(len(de)), 'dir_rows': int(len(direction)), 'negative_rows': int(de['label'].eq(0).sum()), 'extra_lm_de_rows': int(de['gene_role'].eq('extra_lm_not_hvg').sum()), 'extra_lm_dir_rows': int(direction['gene_role'].eq('extra_lm_not_hvg').sum())}
    audit = {'method_version': METHOD_VERSION, 'checks_passed': 18, 'streaming_pass_coordinate_identity': True, 'coordinate_rows': pass_audits[0]['coordinate_rows'], 'duplicate_coordinates': pass_audits[0]['duplicate_coordinates'], 'valid_human_umis_conserved': True, 'valid_human_umis': pass_audits[0]['valid_human_umis'], 'selected_csr_shape': [len(cells), len(union)], 'selected_csr_nnz': nnz, 'sampled_cp10k_log1p_max_abs_error': maximum_log_error, 'input_h5ad_shape': [len(cells), N_HVG], 'input_h5ad_core_projection_exact': True, 'extra_lm_in_input_h5ad': 0, 'core_extra_ensembl_overlap': 0, 'split_created': False, 'label_checks': label_checks}
    write_json(out_root / 'final_audit.json', audit)
    return audit

def final_manifest(raw_root: Path, out_root: Path, lincs: Path, ncbi: Path, official_valid_cells_path: Path) -> dict[str, Any]:
    inputs = {name: raw_root / filename for (name, filename) in RAW_NAMES.items()}
    inputs.update({'lincs_gene_info': lincs, 'ncbi_gene_info': ncbi, 'official_valid_cells': official_valid_cells_path})
    outputs = [out_root / 'sciplex3_core_hvg2000_input.h5ad', out_root / 'metadata' / 'core_hvg2000.csv', out_root / 'metadata' / 'extra_lm_not_hvg.csv', out_root / 'metadata' / 'truth_union_genes.csv', out_root / 'labels' / 'label_audit_all_cells.json']
    outputs.extend((out_root / 'labels' / f'{cell}_{task}.csv' for cell in VALID_CELL_TYPES for task in ('DE', 'DIR', 'ExtraLM_DE', 'ExtraLM_DIR')))
    require_files([*inputs.values(), *outputs])
    manifest = {'method_version': METHOD_VERSION, 'created_without_train_test_split': True, 'inputs': {name: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for (name, path) in inputs.items()}, 'outputs': {str(path.relative_to(out_root)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in outputs}, 'audits': {name: json.loads((out_root / relative).read_text(encoding='utf-8')) for (name, relative) in {'metadata': 'metadata/metadata_audit.json', 'hvg': 'metadata/hvg_audit.json', 'lincs': 'metadata/lincs_mapping_audit.json', 'selection': 'metadata/selection_audit.json', 'input_h5ad': 'input_h5ad_audit.json', 'labels': 'labels/label_audit_all_cells.json', 'pass1': 'intermediate/pass1_audit.json', 'pass2': 'intermediate/pass2_audit.json', 'extract': 'intermediate/extract_audit.json', 'extract_summary': 'intermediate/extract_summary.json', 'final': 'final_audit.json'}.items()}, 'leakage_boundary': {'model_input_h5ad_contains_only_core_hvg2000': True, 'extra_lm_selection_uses_response_labels': False, 'extra_lm_selection': 'official LINCS 978 intersect measured human genes minus core HVG', 'extra_lm_expression_used_only_for_truth_statistics': True, 'forced_extra_gene_count': False}}
    write_json(out_root / 'build_manifest.json', manifest)
    return manifest

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--lincs-gene-info', type=Path, required=True)
    parser.add_argument('--ncbi-gene-info', type=Path, required=True)
    parser.add_argument('--official-valid-cells', type=Path, required=True)
    parser.add_argument('--stage', choices=['metadata', 'pass1', 'pass2', 'select', 'extract', 'assemble', 'labels', 'audit', 'manifest', 'all'], default='all')
    parser.add_argument('--streamer-source', type=Path, default=Path(__file__).with_name('sciplex3_raw_stream.cpp'))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'intermediate').mkdir(parents=True, exist_ok=True)
    official_valid_cells_path = args.official_valid_cells.resolve()
    require_files([*(raw_root / value for value in RAW_NAMES.values()), args.lincs_gene_info, args.ncbi_gene_info, official_valid_cells_path])
    stages = ['metadata', 'pass1', 'pass2', 'select', 'extract', 'assemble', 'labels', 'audit', 'manifest'] if args.stage == 'all' else [args.stage]
    binary = out_root / 'intermediate' / 'sciplex3_raw_stream'
    if any((stage in stages for stage in ('pass1', 'pass2', 'extract'))):
        compile_streamer(args.streamer_source.resolve(), binary)
    if 'metadata' in stages:
        print(json.dumps(prepare_metadata(raw_root, out_root, official_valid_cells_path), indent=2), flush=True)
    if 'pass1' in stages:
        run_streamer(binary=binary, mode='pass1', raw_root=raw_root, out_root=out_root)
        print(json.dumps(compute_clip_values(out_root), indent=2), flush=True)
    if 'pass2' in stages:
        run_streamer(binary=binary, mode='pass2', raw_root=raw_root, out_root=out_root, extra=['--library-size', str(out_root / 'intermediate' / 'library_size.u64'), '--clip-value', str(out_root / 'intermediate' / 'clip_value.f64')])
    if 'select' in stages:
        print(json.dumps(select_hvgs(out_root), indent=2), flush=True)
        print(json.dumps(build_lincs_mapping(out_root, args.lincs_gene_info.resolve(), args.ncbi_gene_info.resolve()), indent=2), flush=True)
        print(json.dumps(prepare_selection(out_root), indent=2), flush=True)
    if 'extract' in stages:
        union = pd.read_parquet(out_root / 'metadata' / 'truth_union_genes.parquet')
        run_streamer(binary=binary, mode='extract', raw_root=raw_root, out_root=out_root, extra=['--library-size', str(out_root / 'intermediate' / 'library_size.u64'), '--selection-map', str(out_root / 'metadata' / 'feature_to_truth_union.i32'), '--n-selected', str(len(union))])
    if 'assemble' in stages:
        print(json.dumps(assemble_h5ad(out_root), indent=2), flush=True)
    if 'labels' in stages:
        print(json.dumps(generate_labels(out_root), indent=2), flush=True)
    if 'audit' in stages:
        print(json.dumps(run_final_audit(out_root), indent=2), flush=True)
    if 'manifest' in stages:
        print(json.dumps(final_manifest(raw_root, out_root, args.lincs_gene_info.resolve(), args.ncbi_gene_info.resolve(), official_valid_cells_path), indent=2), flush=True)
if __name__ == '__main__':
    main()
