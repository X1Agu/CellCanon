from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from cellcanon.model.data import SCIPLEX_ROOT, metadata

def read(path):
    return json.loads(Path(path).read_text())

def obs_and_genes():
    obs = pd.read_parquet(SCIPLEX_ROOT / 'metadata/valid_cells_pdata.parquet')
    genes = pd.read_parquet(SCIPLEX_ROOT / 'metadata/truth_union_genes.parquet')
    assert len(genes) == 2931 and genes['is_core_hvg'].to_numpy()[:2000].all()
    _, expected = metadata()
    assert genes.ensembl_id.tolist() == expected.gene.tolist()
    obs = obs.copy()
    obs['dose_numeric'] = pd.to_numeric(obs['dose'])
    obs['control'] = obs['vehicle'].astype(str).eq('TRUE')
    assert set(obs.time_point.astype(str)) == {'24'}
    return (obs, genes)

def selected_expression(rows):
    root = SCIPLEX_ROOT / 'intermediate'
    meta = read(root / 'extract_summary.json')
    ptr = np.memmap(root / 'selected_indptr.u64', '<u8', mode='r', shape=(meta['valid_cells'] + 1,))
    idx = np.memmap(root / 'selected_indices.u16', '<u2', mode='r', shape=(meta['selected_nnz'],))
    val = np.memmap(root / 'selected_log_data.f32', '<f4', mode='r', shape=(meta['selected_nnz'],))
    rows = np.asarray(rows, dtype=np.int64)
    n = ptr[rows + 1] - ptr[rows]
    ip = np.r_[0, np.cumsum(n)].astype(np.int64)
    indices = np.empty(int(ip[-1]), dtype=np.int32)
    data = np.empty(int(ip[-1]), dtype=np.float32)
    for j, r in enumerate(rows):
        a, b = (int(ptr[r]), int(ptr[r + 1]))
        lo, hi = (int(ip[j]), int(ip[j + 1]))
        indices[lo:hi] = idx[a:b]
        data[lo:hi] = val[a:b]
    return sparse.csr_matrix((data, indices, ip), shape=(len(rows), meta['selected_genes']))
