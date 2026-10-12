from __future__ import annotations
import math
from cellcanon.resources.identifiers import _gene_symbol
TYPES = ('Ki', 'Kd', 'IC50')
UNIT_NM = {'M': 1000000000.0, 'mM': 1000000.0, 'uM': 1000.0, 'µM': 1000.0, 'μM': 1000.0, 'nM': 1.0, 'pM': 0.001}
RELATIONS = ('=', '<', '<=', '>', '>=')

def measurement(activity):
    if activity.get('assay_type') != 'B':
        return (None, 'not_binding_assay')
    if activity.get('standard_type') not in TYPES:
        return (None, 'measurement_type_not_selected')
    if activity.get('standard_flag') != 1:
        return (None, 'not_standardized')
    if activity.get('data_validity_comment') not in (None, '', 'Manually validated'):
        return (None, 'data_validity_flag')
    if activity.get('potential_duplicate') != 0:
        return (None, 'duplicate_or_unknown_duplicate_flag')
    if activity.get('standard_upper_value') not in (None, ''):
        return (None, 'unsupported_reported_interval')
    relation = activity.get('standard_relation')
    unit = activity.get('standard_units')
    if relation not in RELATIONS:
        return (None, 'unsupported_relation')
    if unit not in UNIT_NM:
        return (None, 'unsupported_unit')
    try:
        value = float(activity.get('standard_value')) * UNIT_NM[unit]
    except (TypeError, ValueError):
        return (None, 'invalid_numeric_value')
    if not math.isfinite(value) or value <= 0:
        return (None, 'nonpositive_or_nonfinite_value')
    potency = 9 - math.log10(value)
    return ({'measurement_type': activity['standard_type'], 'relation': relation, 'value_nM': value, 'reported_value': str(activity['standard_value']), 'reported_units': unit, 'potency_lower_p': potency if relation in ('=', '<', '<=') else None, 'potency_upper_p': potency if relation in ('=', '>', '>=') else None, 'lower_p_inclusive': relation in ('=', '<='), 'upper_p_inclusive': relation in ('=', '>='), 'reported_pchembl': activity.get('pchembl_value')}, None)

def threshold_evidence(value_nM, relation, threshold_nM):
    if relation == '=':
        return (float(value_nM <= threshold_nM), float(value_nM > threshold_nM))
    if relation in ('<', '<='):
        return (float(value_nM <= threshold_nM), 0.0)
    if relation == '>':
        return (0.0, float(value_nM >= threshold_nM))
    if relation == '>=':
        return (0.0, float(value_nM > threshold_nM))
    raise ValueError(relation)

def curate(activity, assay, target):
    record, reason = measurement(activity)
    if reason:
        return (None, reason)
    if not assay or not target:
        return (None, 'missing_assay_or_target_record')
    if activity.get('target_chembl_id') != assay.get('target_chembl_id') or activity.get('target_chembl_id') != target.get('target_chembl_id'):
        return (None, 'assay_target_disagreement')
    if assay.get('assay_type') != 'B' or assay.get('confidence_score') != 9 or assay.get('relationship_type') != 'D':
        return (None, 'not_confident_direct_binding_assignment')
    if activity.get('target_organism') != 'Homo sapiens' or target.get('organism') != 'Homo sapiens' or target.get('tax_id') != 9606 or (assay.get('assay_organism') != 'Homo sapiens'):
        return (None, 'not_explicit_human_assignment')
    if assay.get('variant_sequence') or activity.get('assay_variant_accession') or activity.get('assay_variant_mutation'):
        return (None, 'annotated_variant')
    components = target.get('target_components') or []
    if target.get('target_type') != 'SINGLE PROTEIN' or len(components) != 1:
        return (None, 'not_single_protein_component')
    gene, resolution = _gene_symbol(components[0])
    if not gene:
        return (None, 'ambiguous_gene_symbol')
    record.update({'activity_id': activity['activity_id'], 'molecule_chembl_id': activity['molecule_chembl_id'], 'assay_chembl_id': activity['assay_chembl_id'], 'target_chembl_id': activity['target_chembl_id'], 'target_gene_symbol': gene, 'gene_symbol_resolution': resolution, 'target_uniprot': components[0].get('accession'), 'document_chembl_id': activity.get('document_chembl_id'), 'document_year': activity.get('document_year'), 'target_assignment_confidence': 9, 'relationship_type': 'D', 'bao_format': assay.get('bao_format'), 'single_protein_assay_format': assay.get('bao_format') == 'BAO_0000357', 'assay_cell_type': assay.get('assay_cell_type'), 'assay_tissue': assay.get('assay_tissue'), 'assay_description': assay.get('description'), 'source_release': 'ChEMBL_37'})
    return (record, None)
