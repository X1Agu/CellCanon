from __future__ import annotations
from typing import Any

def _gene_symbol(component: dict[str, Any]) -> tuple[str | None, str]:
    symbols = sorted({str(row.get('component_synonym')) for row in component.get('target_component_synonyms', []) if row.get('syn_type') == 'GENE_SYMBOL' and row.get('component_synonym')})
    if len(symbols) == 1:
        return (symbols[0], 'exact_chembl_component_gene_symbol')
    if not symbols:
        return (None, 'gene_symbol_unavailable')
    return (None, 'ambiguous_component_gene_symbols')
