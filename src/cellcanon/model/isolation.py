from __future__ import annotations
import numpy as np

def subset_by_drugs(panel, drugs):
    allowed = set(drugs)
    return panel.subset(np.array([i for i, d in enumerate(panel.drugs) if d in allowed]))
