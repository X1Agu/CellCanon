from __future__ import annotations
import numpy as np

def goal_loss(values, goal):
    x = np.asarray(values, float)
    return np.abs(x[..., 0] - goal) + 5 * np.maximum(np.abs(x[..., 1]) - 0.5, 0)
