from __future__ import annotations

def label(value):
    return 'decrease' if value < -0.25 else 'increase' if value > 0.25 else 'near_zero'
