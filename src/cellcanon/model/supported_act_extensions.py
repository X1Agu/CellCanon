from __future__ import annotations
import numpy as np
from cellcanon.model.condition_reasoner import goal_loss

def task_loss(values, goal):
    x = np.asarray(values, float)
    if goal.get('mode', 'exact') == 'exact':
        return goal_loss(x, goal['value'])
    if goal['mode'] == 'threshold':
        sign = 1 if goal['value'] > 0 else -1
        return np.maximum(abs(goal['value']) - sign * x[..., 0], 0) + 5 * np.maximum(abs(x[..., 1]) - 0.5, 0)
    raise ValueError('Unknown goal mode')
