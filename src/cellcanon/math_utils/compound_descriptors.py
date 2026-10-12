from __future__ import annotations
import re
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from cellcanon.contracts import assert_unique, require_columns

def build_sciplex_retrieval_metadata(registry: pd.DataFrame, *, radius: int, fp_size: int) -> pd.DataFrame:
    require_columns(registry, ['drug', 'smiles', 'target', 'pathway'], 'Compound registry')
    assert_unique(registry, ['drug'], 'Compound registry')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    rows = []
    for row in registry.itertuples(index=False):
        smiles = str(row.smiles).strip()
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        fingerprint = bytes(DataStructs.BitVectToBinaryText(generator.GetFingerprint(molecule))) if molecule is not None else None
        target_roles = sorted({token.strip() for token in re.split('[,;|]', str(row.target)) if token.strip()})
        pathway = str(row.pathway).strip()
        rows.append({'drug': str(row.drug), 'canonical_smiles': Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None, 'morgan_fingerprint': fingerprint, 'morgan_radius': radius if fingerprint is not None else None, 'morgan_fp_size': fp_size if fingerprint is not None else None, 'chemical_available': fingerprint is not None, 'chemical_parse_error': None if fingerprint is not None else 'rdkit_mol_from_smiles_failed', 'target_roles': target_roles, 'target_available': bool(target_roles), 'drug_moa': pathway, 'moa_available': bool(pathway), 'metadata_proxy': False})
    result = pd.DataFrame(rows).sort_values('drug', kind='stable').reset_index(drop=True)
    assert_unique(result, ['drug'], 'Compound metadata')
    return result
