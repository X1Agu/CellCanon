from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
SCIPLEX_ROOT = Path('<PREPROCESSED_DATA_ROOT>')
METADATA_ROOT = Path('<METADATA_ROOT>')
CELLS = ('A549', 'K562', 'MCF7')
EXPECTED = {}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class Panel:
    drugs: list[str]
    genes: list[str]
    symbols: list[str]
    continuous: np.ndarray
    de: np.ndarray
    up: np.ndarray
    sampled: np.ndarray
    evidence: list[dict]
    split: str
    cells: tuple[str, ...] = CELLS

    def subset(self, indices: np.ndarray) -> 'Panel':
        return Panel([self.drugs[i] for i in indices], self.genes, self.symbols, self.continuous[indices], self.de[indices], self.up[indices], self.sampled[indices], self.evidence, self.split, self.cells)

def metadata(root: Path=METADATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((root / 'prepare_manifest.json').read_text())
    frames = {}
    for name in ('drug_metadata', 'gene_order', 'extra_gene_order'):
        path = root / Path(manifest['outputs'][name]).name
        assert sha256(path) == manifest['output_sha256'][name], name
        frames[name] = pd.read_parquet(path)
    genes = pd.concat([frames['gene_order'].assign(role='core'), frames['extra_gene_order'].assign(role='extra')], ignore_index=True)
    assert len(genes) == 2931 and genes['gene'].nunique() == 2931
    drug = frames['drug_metadata'].sort_values('drug').reset_index(drop=True)
    assert drug.groupby('split').size().to_dict() == {'support': 56, 'test': 113, 'validation': 19}
    return (drug, genes)

def load_panel(split: str, *, root: Path=SCIPLEX_ROOT, allow_test: bool=False) -> Panel:
    if split not in ('support', 'validation', 'test'):
        raise ValueError(split)
    if split == 'test' and (not allow_test):
        raise ValueError('Loading test responses requires an explicit allow_test=True argument for evaluation or simulated observations.')
    evidence = []
    for relative, expected in EXPECTED.items():
        path = root / relative
        actual = sha256(path)
        assert actual == expected, relative
        evidence.append({'path': str(path), 'sha256': actual, 'kind': 'dataset_manifest'})
    drug_meta, gene_meta = metadata()
    drugs = sorted(drug_meta.loc[drug_meta['split'].eq(split), 'drug'].tolist())
    genes = gene_meta['gene'].astype(str).tolist()
    symbols = gene_meta['gene_symbol'].astype(str).tolist()
    shape = (len(drugs), len(CELLS), len(genes))
    y = np.empty(shape[:2] + (2000,), dtype=np.float64)
    de = np.empty(shape, dtype=np.float64)
    up = np.empty(shape, dtype=np.float64)
    sampled = np.zeros(shape, dtype=bool)
    directory = 'locked_test' if split == 'test' else split
    drug_index = {v: i for i, v in enumerate(drugs)}
    gene_index = {v: i for i, v in enumerate(genes)}
    for ci, cell in enumerate(CELLS):
        base = root / 'splits/drug_zero_shot_v1' / directory
        path = base / 'truth_full_stats' / f'{cell}_HVG2000_plus_ExtraLM_stats.parquet'
        frame = pd.read_parquet(path)
        assert not frame.duplicated(['pert', 'ensembl_id']).any()
        assert set(frame['pert']) == set(drugs)

        def array(col):
            return frame.pivot(index='pert', columns='ensembl_id', values=col).reindex(index=drugs, columns=genes).to_numpy(dtype=float)
        y[:, ci] = array('logfoldchanges')[:, :2000].astype(np.float32).astype(float)
        de[:, ci] = array('de_label_strict')
        up[:, ci] = array('direction_label')
        evidence.append({'path': str(path), 'sha256': sha256(path), 'kind': split + '_truth'})
        for suffix in ('DE', 'ExtraLM_DE'):
            path = base / 'labels' / f'{cell}_{suffix}.csv'
            labels = pd.read_csv(path)
            di = labels['pert'].map(drug_index).to_numpy()
            gi = labels['ensembl_id'].map(gene_index).to_numpy()
            assert pd.notna(di).all() and pd.notna(gi).all()
            assert np.array_equal(de[di, ci, gi], labels['label'].to_numpy())
            sampled[di, ci, gi] = True
            evidence.append({'path': str(path), 'sha256': sha256(path), 'kind': split + '_sampled_labels'})
    assert np.isfinite(y).all() and np.isfinite(de).all()
    assert np.isfinite(up[de == 1]).all() and np.isin(up[de == 1], [0, 1]).all()
    up[de == 0] = np.nan
    return Panel(drugs, genes, symbols, y, de, up, sampled, evidence, split)

def drug_kernels(drug_meta: pd.DataFrame) -> dict[str, np.ndarray]:
    bits = np.stack([np.unpackbits(np.frombuffer(x, dtype=np.uint8), bitorder='little') for x in drug_meta['morgan_fingerprint']]).astype(float)
    dot = bits @ bits.T
    chemical = dot / np.maximum(bits.sum(1)[:, None] + bits.sum(1)[None] - dot, 1)
    targets = [set(map(str.casefold, row)) for row in drug_meta['target_roles']]
    ignored = {'unknown', 'other', 'others', 'nan', 'none', '', 'na'}
    targets = [x - ignored for x in targets]
    target = np.array([[len(a & b) / max(1, len(a | b)) for b in targets] for a in targets])
    moa = drug_meta['drug_moa'].str.casefold().to_numpy()
    moa_kernel = (moa[:, None] == moa[None]).astype(float)
    for i, name in enumerate(moa):
        if name in ignored:
            moa_kernel[i, :] = 0
            moa_kernel[:, i] = 0
    return {'chemical': chemical, 'target': target, 'moa': moa_kernel, 'biology': 0.5 * target + 0.5 * moa_kernel, 'chemical_biology': 0.5 * chemical + 0.25 * target + 0.25 * moa_kernel}
