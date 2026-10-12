from __future__ import annotations
from collections.abc import Iterable
import pandas as pd

class SchemaError(ValueError):
    pass

def require_columns(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise SchemaError(f'{table_name} is missing required columns: {missing}')

def assert_unique(frame: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    columns = list(columns)
    require_columns(frame, columns, table_name)
    duplicate_mask = frame.duplicated(columns, keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, columns].head(5).to_dict('records')
        raise SchemaError(f'{table_name} has duplicate keys {columns}: {examples}')
