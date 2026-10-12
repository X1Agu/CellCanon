from __future__ import annotations
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def annotation_key(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
