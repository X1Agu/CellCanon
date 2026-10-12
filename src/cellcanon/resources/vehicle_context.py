from __future__ import annotations
import json
from pathlib import Path
import subprocess
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from cellcanon.resources.sciplex_knowledge import OUTPUT, SCIPLEX_ROOT, cohort, sha256, write_json

def summarize_matrix(logged, counts, libraries):
    n = logged.shape[0]
    assert n > 0 and (libraries > 0).all()
    log64 = logged.astype(np.float64)
    mean = np.asarray(log64.mean(axis=0)).ravel()
    second = np.asarray(log64.power(2).mean(axis=0)).ravel()
    return {'control_mean_log1p_cp10k': mean, 'control_sd_log1p_cp10k': np.sqrt(np.maximum(0, (second - mean ** 2) * n / max(1, n - 1))), 'control_detection_fraction': np.asarray((counts > 0).sum(axis=0)).ravel() / n, 'control_mean_raw_count': np.asarray(counts.astype(np.float64).mean(axis=0)).ravel(), 'control_mean_cp10k': np.asarray(counts.astype(np.float64).multiply((10000 / libraries)[:, None]).mean(axis=0)).ravel()}

def run(output=OUTPUT):
    root = output / 'cell_baseline'
    root.mkdir(parents=True, exist_ok=True)
    final = root / 'manifest.json'
    if final.exists():
        m = json.loads(final.read_text())
        for name, expected in m['output_sha256'].items():
            assert sha256(root / name) == expected, name
        print(json.dumps({'status': m['status'], 'cached': True}), flush=True)
        return
    buildpath = SCIPLEX_ROOT / 'build_manifest.json'
    build = json.loads(buildpath.read_text())
    if not {'inputs', 'audits', 'outputs'} <= build.keys():
        raise ValueError('A complete preprocessing manifest is required.')
    meta = SCIPLEX_ROOT / 'metadata'
    valid = pd.read_parquet(meta / 'valid_cells_pdata.parquet')
    controls = valid.loc[valid.vehicle.eq('TRUE')].sort_values('source_cell_index').copy()
    assert controls.treatment.eq('S0000').all() and controls.time_point.astype(str).eq('24').all()
    expected = {'A549': 3287, 'K562': 3359, 'MCF7': 6358}
    assert controls.groupby('cell_type').size().to_dict() == expected
    assert controls.index.is_unique and controls.source_cell_index.is_unique
    human = pd.read_parquet(meta / 'human_features.parquet').sort_values('human_gene_index')
    assert len(human) == 58347 and human.ensembl_id.is_unique
    allfeatures = pd.read_parquet(meta / 'all_features.parquet')
    n_cells = build['audits']['metadata']['source_cells']
    work = root / 'control_csr'
    work.mkdir(exist_ok=True)
    mask = np.zeros(n_cells, dtype=np.uint8)
    mask[controls.source_cell_index.to_numpy(int)] = 1
    mask.tofile(work / 'control_mask.u8')
    source = Path(__file__).with_name('sciplex3_raw_stream.cpp')
    binary = work / 'sciplex3_control_stream'
    subprocess.run(['g++', '-O3', '-std=c++17', str(source), '-lz', '-o', str(binary)], check=True)
    raw = Path(build['inputs']['matrix']['path'])
    print(json.dumps({'event': 'control_extraction_source_hash', 'bytes': raw.stat().st_size}), flush=True)
    assert sha256(raw) == build['inputs']['matrix']['sha256']
    library_path = SCIPLEX_ROOT / 'intermediate/library_size.u64'
    command = [str(binary), '--mode', 'extract', '--matrix', str(raw), '--feature-map', str(meta / 'feature_to_human.i32'), '--cell-valid', str(work / 'control_mask.u8'), '--out-dir', str(work), '--n-cells', str(n_cells), '--n-features', str(len(allfeatures)), '--n-human', str(len(human)), '--library-size', str(library_path), '--selection-map', str(meta / 'feature_to_human.i32'), '--n-selected', str(len(human))]
    print(json.dumps({'event': 'control_extraction_started', 'controls': len(controls), 'human_genes': len(human)}), flush=True)
    with (work / 'stream.log').open('w') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    extract = json.loads((work / 'extract_summary.json').read_text())
    audit = json.loads((work / 'extract_audit.json').read_text())
    assert extract['valid_cells'] == len(controls) and extract['selected_genes'] == len(human)
    nnz = extract['selected_nnz']
    indptr = np.fromfile(work / 'selected_indptr.u64', dtype='<u8')
    indices = np.fromfile(work / 'selected_indices.u16', dtype='<u2').astype(np.int32)
    rawcounts = np.fromfile(work / 'selected_counts.u32', dtype='<u4')
    data = np.fromfile(work / 'selected_log_data.f32', dtype='<f4')
    assert len(data) == len(indices) == len(rawcounts) == nnz
    assert len(indptr) == len(controls) + 1 and int(indptr[-1]) == nnz
    shape = (len(controls), len(human))
    logged = sparse.csr_matrix((data, indices, indptr), shape=shape)
    counts = sparse.csr_matrix((rawcounts, indices, indptr), shape=shape)
    assert logged.has_canonical_format and (rawcounts > 0).all()
    libraries = np.fromfile(library_path, dtype='<u8')[controls.source_cell_index.to_numpy(int)]
    assert np.array_equal(np.asarray(counts.sum(axis=1)).ravel(), libraries)
    expectedlog = counts.astype(np.float64).multiply((10000 / libraries)[:, None]).tocsr()
    expectedlog.data = np.log1p(expectedlog.data)
    log_error = float(np.max(np.abs(expectedlog.data - logged.data)))
    assert log_error < 1e-06
    _, genes = cohort()
    rolemap = genes.set_index('gene').role.to_dict()
    all_rows = []
    for cell, n in expected.items():
        selected = controls.cell_type.eq(cell).to_numpy()
        values = summarize_matrix(logged[selected], counts[selected], libraries[selected])
        frame = human[['ensembl_id', 'gene_short_name']].rename(columns={'gene_short_name': 'gene_symbol'}).copy()
        frame['cell_line'] = cell
        frame['control_cells'] = n
        frame['gene_role'] = frame.ensembl_id.map(rolemap).fillna('background_only')
        for name, array in values.items():
            frame[name] = array
        frame['response_truth_access'] = False
        frame['evidence_scope'] = 'observed_vehicle_control_24h_not_drug_response'
        all_rows.append(frame)
    baseline = pd.concat(all_rows, ignore_index=True)
    baseline.to_parquet(root / 'all_human_control_baseline.parquet', index=False)
    panel = genes.merge(baseline, left_on=['gene', 'gene_symbol'], right_on=['ensembl_id', 'gene_symbol'], how='left', validate='one_to_many')
    assert len(panel) == 3 * 2931 and panel.control_cells.notna().all()
    assert not panel.duplicated(['gene', 'cell_line']).any()
    panel.to_parquet(root / 'panel_control_baseline.parquet', index=False)
    controls[['source_cell_index', 'cell_type', 'time_point', 'vehicle', 'treatment', 'replicate']].to_parquet(root / 'control_cells.parquet')
    hpath = Path(build['audits']['input_h5ad']['path'])
    assert sha256(hpath) == build['audits']['input_h5ad']['sha256']
    h5 = ad.read_h5ad(hpath, backed='r')
    sample_errors = {}
    try:
        for cell in expected:
            chosen = h5.obs.vehicle.astype(str).eq('TRUE') & h5.obs.cell_type.astype(str).eq(cell)
            assert int(chosen.sum()) == expected[cell]
            control = h5[np.flatnonzero(chosen), :].X
            ref = baseline[baseline.cell_line.eq(cell)].set_index('ensembl_id').loc[h5.var.ensembl_id.astype(str)]
            error = float(np.max(np.abs(np.asarray(control.astype(np.float64).mean(axis=0)).ravel() - ref.control_mean_log1p_cp10k.to_numpy())))
            assert error < 1e-07
            detection = np.asarray((control > 0).sum(axis=0)).ravel() / expected[cell]
            assert np.array_equal(detection, ref.control_detection_fraction.to_numpy())
            sample_errors[cell] = error
    finally:
        h5.file.close()
    inputs = [buildpath, meta / 'valid_cells_pdata.parquet', meta / 'human_features.parquet', meta / 'feature_to_human.i32', library_path, source, hpath]
    manifest = {'status': 'SCIPLEX_CONTROL_BASELINE_COMPLETE', 'controls': expected, 'human_genes': len(human), 'all_human_rows': len(baseline), 'core_rows': 6000, 'extra_rows': 2793, 'control_nnz': nnz, 'control_human_umis': int(libraries.sum()), 'normalization': 'per-cell log1p(raw_count / human-only library size * 10000); means after transformation', 'identity': 'official-hash QC and 24h vehicle=S0000; source indices sorted; no treated cell retained', 'all_count_library_checks': len(controls), 'log1p_max_abs_error': log_error, 'independent_core_mean_max_error_by_cell': sample_errors, 'core_detection_checks': 6000, 'extract_summary': extract, 'raw_scan_audit': audit, 'scope': 'Background/control resources only. Extra/background genes do not alter the frozen numeric input contract.', 'source_sha256': {str(p): sha256(p) for p in inputs}, 'raw_matrix_sha256': build['inputs']['matrix']['sha256'], 'output_sha256': {p.name: sha256(p) for p in root.glob('*.parquet')}, 'llm_calls': 0, 'public_http_calls': 0, 'de_dir_logfc_label_reads': 0, 'treated_cells_retained': 0}
    write_json(final, manifest)
    print(json.dumps({k: manifest[k] for k in ['status', 'controls', 'human_genes', 'all_human_rows', 'core_rows', 'extra_rows', 'independent_core_mean_max_error_by_cell']}), flush=True)
