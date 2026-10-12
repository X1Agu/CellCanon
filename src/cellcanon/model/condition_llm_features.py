from __future__ import annotations
import re
import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
VIEWS = {'static_text': ('structure', 'text')}

def sanitized(text, packet):
    for value in [packet['drug'], packet.get('drug_name', ''), packet['cell']]:
        if value:
            text = re.sub(re.escape(value), ' ', text, flags=re.I)
    return re.sub('\\bS[0-9]{4,}\\b', ' ', text)

def row_features(packet):
    structure = {'intercept': 1.0, 'cell=' + packet['cell']: 1.0, 'target_count': np.log1p(packet['targets_available']), 'route_count': float(len(packet['route_groups']))}
    text = [str(packet['native_pathway']), *map(str, packet['native_roles'])]
    for role in packet['native_roles']:
        structure['role=' + str(role)] = 1.0
    structure['native_pathway=' + str(packet['native_pathway'])] = 1.0
    for b in packet['background']:
        text.append(b['text'])
    for t in packet['targets']:
        key = t['target']
        structure['target=' + key] = 1.0
        for a in t['actions']:
            structure['target_action=' + key + ':' + str(a)] = 1.0
        detection = t['control'].get('minimum_detection')
        structure['target_detection=' + key] = float(detection or 0.0)
        structure['target_detection_missing=' + key] = float(detection is None)
        if t['function']:
            text.append(t['function']['text'])
        text.append(key)
    for g in packet['route_groups']:
        structure['route_gene=' + g['gene']] = 1.0
        structure['route_ambiguous=' + g['gene']] = float(len(g['polarities']) > 1)
        structure['route_cell_compatible=' + g['gene']] = float(bool(g['cell_compatible_polarities']))
    return {'structure': structure, 'text': sanitized(' '.join(text), packet)}

class FeatureMap:

    def __init__(self, view, dimensions=16):
        self.view = view
        self.dimensions = dimensions
        self.maps = {}

    def fit(self, rows, identities):
        if len(rows) != len(identities):
            raise ValueError('Training identity mismatch.')
        self.fit_identities = list(identities)
        blocks = []
        for field in VIEWS[self.view]:
            if field in {'structure', 'llm_numeric'}:
                mapper = DictVectorizer(sparse=True)
            else:
                mapper = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float64)
            block = mapper.fit_transform([r[field] for r in rows])
            self.maps[field] = mapper
            blocks.append(normalize(block, norm='l2'))
        matrix = hstack(blocks).tocsr() / np.sqrt(len(blocks))
        k = max(1, min(self.dimensions, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=20260910)
        x = self.svd.fit_transform(matrix)
        self.scale = max(float(np.sqrt(np.mean(np.sum(x * x, axis=1)))), 1e-08)
        self.feature_count = matrix.shape[1]
        return x / self.scale

    def transform(self, rows):
        blocks = [normalize(self.maps[field].transform([r[field] for r in rows]), norm='l2') for field in VIEWS[self.view]]
        matrix = hstack(blocks).tocsr() / np.sqrt(len(blocks))
        return self.svd.transform(matrix) / self.scale
