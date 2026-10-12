from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from cellcanon.model.condition_llm_packets import digest, wire
from cellcanon.model.condition_pilot_knowledge import packet, normalized, verify
from cellcanon.model.data import metadata
MODEL = 'Qwen/Qwen3-Embedding-0.6B'
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
SNAPSHOT = Path('<EMBEDDING_MODEL_ROOT>')
CHUNK_TOKENS = 768

def normalize_record(card, cell, record, aliases):
    p = packet(card, cell)
    if record['packet_sha256'] != digest(p):
        raise ValueError('Compiler packet hash changed.')
    raw = record.get('result')
    result = normalized(p, raw)
    if isinstance(result, dict) and aliases.get(str(result.get('drug', '')).casefold()) == card['drug']:
        result['drug'] = card['drug']
    errors = verify(p, result)
    return {'valid': not errors, 'errors': errors, 'raw_valid': record['valid'], 'result': result, 'packet_sha256': digest(p), 'normalization': 'exact_metadata_alias_and_sourced_quote_length_v1'}

def aliases():
    md, _ = metadata()
    groups = {}
    for row in md.to_dict('records'):
        for key in ('canonical_name', 'legacy_drug', 'drug'):
            value = row.get(key)
            if isinstance(value, str) and value:
                groups.setdefault(value.casefold(), set()).add(row['drug'])
    return {k: next(iter(v)) for k, v in groups.items() if len(v) == 1}

def target_weights(card, cell, selection=None):
    weights = {}
    if selection is None:
        for t in card['targets']:
            active = any((m.get('supports_active_1uM') for m in t['measurements']))
            weights[t['target']] = 1.0 if t['actions'] or active else 0.25
    else:
        for t in selection:
            weights[t['target']] = {'high': 1.0, 'medium': 0.5, 'low': 0.1, 'unknown': 0.25}[t['relevance']]
    for t in card['targets']:
        if t['target'] in weights:
            detect = t['control'][cell].get('detection')
            weights[t['target']] *= 0.25 + 0.75 * np.sqrt(np.clip(detect if detect is not None else 0.5, 0, 1))
    return weights

def structure(card, cell, selection=None):
    out = {'target_count': np.log1p(card['available_target_count']), 'omitted_targets': np.log1p(card['omitted_target_count'])}
    for role in card['roles']:
        out['role=' + str(role)] = 1.0
    out['pathway=' + str(card['pathway'])] = 1.0
    selected = {s['target']: s for s in selection} if selection is not None else None
    for t in card['targets']:
        if selected is not None and t['target'] not in selected:
            continue
        key = t['target']
        out['target=' + key] = 1.0
        for action in t['actions']:
            out['action=' + key + ':' + str(action)] = 1.0
        ctrl = t['control'][cell]
        for field in ('detection', 'mean_log1p_cp10k'):
            value = ctrl.get(field)
            out[field + '=' + key] = float(value or 0.0)
            out[field + '_missing=' + key] = float(value is None)
        for m in t['measurements']:
            kind = str(m.get('measurement_type'))
            relation = str(m.get('relation'))
            out['measurement=' + key + ':' + kind + ':' + relation] = 1.0
            for field in ('potency_lower_p', 'potency_upper_p'):
                if m.get(field) is not None:
                    f = field + '=' + key + ':' + kind + ':' + relation
                    out[f] = max(out.get(f, 0.0), float(np.clip(m[field], 0, 12)) / 12)
            out['single_protein=' + key + ':' + kind] = max(out.get('single_protein=' + key + ':' + kind, 0.0), float(bool(m.get('single_protein_assay_format'))))
        if selected is not None:
            out['relevance=' + key + ':' + selected[key]['relevance']] = 1.0
            out['assay_scope=' + key + ':' + selected[key]['assay_scope']] = 1.0
    if selection is not None:
        out['selected_count'] = float(len(selection))
    return out

class LocalEmbedder:

    def __init__(self, root):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not list(SNAPSHOT.glob('*.safetensors')):
            raise FileNotFoundError('Local embedding weights have not finished downloading.')
        self.tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, padding_side='left')
        self.model = AutoModel.from_pretrained(SNAPSHOT, local_files_only=True, torch_dtype=torch.float16, attn_implementation='sdpa').eval().to('cuda')
        torch.set_num_threads(2)

    def encode(self, texts):
        unique = {}
        first_key = {}
        canonical = {}
        for key, text in texts.items():
            if text not in first_key:
                first_key[text] = key
                unique[key] = text
            canonical[key] = first_key[text]
        vectors = self._encode_unique(unique)
        return {key: vectors[original] for key, original in canonical.items()}

    def _encode_unique(self, texts):
        torch = self.torch
        result = {}
        missing = []
        for key, text in texts.items():
            identity = {'model': MODEL, 'revision': REVISION, 'chunk_tokens': CHUNK_TOKENS, 'pooling': 'last_nonpad_per_chunk_then_length_weighted_normalized_mean', 'text': text}
            h = digest(identity)
            path = self.root / (h + '.npz')
            if path.exists():
                with np.load(path) as z:
                    result[key] = z['embedding']
            else:
                tokens = self.tokenizer.encode(text or 'Unknown biological evidence.', add_special_tokens=True)
                chunks = [tokens[i:i + CHUNK_TOKENS] for i in range(0, len(tokens), CHUNK_TOKENS)]
                missing.append((key, h, path, chunks, text))
        flat = [(i, chunk) for i, item in enumerate(missing) for chunk in item[3]]
        outputs = [[] for _ in missing]
        for offset in range(0, len(flat), 8):
            batch = flat[offset:offset + 8]
            encoded = self.tokenizer.pad({'input_ids': [x[1] for x in batch]}, padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state[:, -1].float()
                vectors = torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy()
            for (index, tokens), vector in zip(batch, vectors):
                outputs[index].append((len(tokens), vector))
            if offset % 128 == 0:
                print(wire({'embedding_chunks': min(offset + 8, len(flat)), 'total': len(flat)}), flush=True)
        for item, chunks in zip(missing, outputs):
            key, h, path, tokens, text = item
            vector = np.average(np.stack([x[1] for x in chunks]), axis=0, weights=[x[0] for x in chunks])
            vector = vector / max(np.linalg.norm(vector), 1e-12)
            assert np.isfinite(vector).all()
            np.savez_compressed(path, embedding=vector.astype(np.float32), token_count=sum((x[0] for x in chunks)), chunk_count=len(chunks), identity=h)
            result[key] = vector.astype(np.float32)
        return result

class PilotMap:

    def __init__(self, view='compiled_semantic', dimensions=96):
        if view != 'compiled_semantic':
            raise ValueError('Only the main compiled-evidence model is included.')
        self.view = view
        self.dimensions = dimensions

    def fit(self, rows, identities):
        self.fit_identities = list(identities)
        self.structure_field = 'compiled_structure'
        self.dv = DictVectorizer()
        a = normalize(self.dv.fit_transform([r[self.structure_field] for r in rows]))
        self.vector_fields = ('compiled_vector', 'compiled_target_vector')
        b = hstack([csr_matrix(normalize(np.asarray([r[f] for r in rows]))) for f in self.vector_fields]) / np.sqrt(2)
        joined = hstack([a, b]).tocsr() / np.sqrt(2)
        self.svd = TruncatedSVD(n_components=min(self.dimensions, joined.shape[0] - 1, joined.shape[1] - 1), random_state=20260911)
        x = self.svd.fit_transform(joined)
        self.scale = max(float(np.sqrt(np.mean(np.sum(x * x, axis=1)))), 1e-09)
        return x / self.scale

    def transform(self, rows):
        a = normalize(self.dv.transform([r[self.structure_field] for r in rows]))
        b = hstack([csr_matrix(normalize(np.asarray([r[f] for r in rows]))) for f in self.vector_fields]) / np.sqrt(2)
        return self.svd.transform(hstack([a, b]).tocsr() / np.sqrt(2)) / self.scale
