"""Fit whole-drug out-of-fold parents for Update.

No model service calls or validation/test response reads. Architecture, epoch and
head penalties are inherited from the already validation-selected parent.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import time
import joblib
import numpy as np
import torch
from cellcanon.model.unified_atlas_condition import AtlasBcMLP, identity_mask
from cellcanon.model.condition_pilot_features import PilotMap
from cellcanon.model.condition_llm_learning import fit_classification, predict_classification
from cellcanon.model.data import CELLS, sha256
from cellcanon.workflow import read, write
ART = Path('<RUN_ROOT>')
PARENT = Path('<RUN_ROOT>')
ROOT = Path('<RUN_ROOT>')
FEATURES = Path('<RUN_ROOT>')

def fit():
    out = ROOT / 'oof'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Completed OOF is immutable.')
    manifest = read(PARENT / 'inputs/manifest.json')
    selection = read(PARENT / 'learning/selection.json')['selected']['compiled_semantic']
    protocol = read(PARENT / 'protocol.json')
    for name in ['support_labels.npz', 'similarities.npz']:
        assert sha256(PARENT / 'inputs' / name) == manifest['files_sha256'][name]
    assert sha256(FEATURES) == read(FEATURES.with_name('features_complete_manifest.json'))['feature_sha256']
    rows = read(FEATURES)
    drugs = manifest['train_drugs']
    ids = [manifest['identity_groups'][d] for d in drugs]
    assert len(set(ids)) == len(ids)
    labels = dict(np.load(PARENT / 'inputs/support_labels.npz'))
    similarity = np.load(PARENT / 'inputs/similarities.npz')['values'][:len(drugs)]
    folds = np.asarray(manifest['folds'])
    write(out / 'protocol.json', {'parent_selection_sha256': sha256(PARENT / 'learning/selection.json'), 'feature_sha256': sha256(FEATURES), 'support_labels_sha256': sha256(PARENT / 'inputs/support_labels.npz'), 'folds': folds.tolist(), 'train_drugs': drugs, 'identity_groups': ids, 'method': '5 outer whole-drug folds; refit 96D mapper, means, normalization, Atlas and frontend on fit drugs; fresh inner5 Atlas masks; fixed parent epochs and penalties; average all3 seeds', 'validation_response_reads': 0, 'test_response_reads': 0, 'new_api_calls': 0})
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = lambda x: torch.as_tensor(x, device=device)
    result = {k: np.empty_like(labels[k], dtype=float) for k in ['continuous', 'de', 'up']}
    records = []
    started = time.monotonic()
    for fold in range(5):
        dest = out / f'fold{fold}'
        dest.mkdir(exist_ok=True)
        if (dest / 'manifest.json').exists():
            rec = read(dest / 'manifest.json')
            for (name, h) in rec['files_sha256'].items():
                assert sha256(dest / name) == h
            held = np.array(rec['held_indices'])
            pred = dict(np.load(dest / 'predictions.npz'))
            for k in result:
                result[k][held] = pred[k]
            records.append(rec)
            continue
        fitix = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        fd = [drugs[i] for i in fitix]
        hd = [drugs[i] for i in held]
        mapper = PilotMap('compiled_semantic', 96)
        getrows = lambda ds: [rows[d + '_' + c] for d in ds for c in CELLS]
        z = mapper.fit(getrows(fd), [(d, c) for d in fd for c in CELLS]).reshape(len(fd), 3, -1).astype(np.float32)
        zh = mapper.transform(getrows(hd)).reshape(len(hd), 3, -1).astype(np.float32)
        joblib.dump(mapper, dest / 'mapper.joblib')
        train = SimpleNamespace(**{k: v[fitix] for (k, v) in labels.items()})
        mean = train.continuous.mean(0)
        scale = max(float(np.sqrt(np.square(train.continuous - mean).mean())), 1.0)
        memory = np.concatenate([(train.continuous - mean) / scale, train.de, np.where(train.de == 1, train.up, 0.0)], axis=-1).astype(np.float32)
        order = np.random.default_rng(20260916).permutation(len(fd))
        inner = np.empty(len(fd), int)
        inner[order] = np.arange(len(fd)) % 5
        mask = identity_mask([ids[i] for i in fitix], [ids[i] for i in fitix], excluded=torch.tensor(inner[:, None] == inner[None, :]), device=device)
        hm = identity_mask([ids[i] for i in held], [ids[i] for i in fitix], device=device)
        ta = (tensor(z), tensor(z), tensor(memory), tensor(similarity[fitix][:, fitix]), mask)
        ha = (tensor(zh), tensor(z), tensor(memory), tensor(similarity[held][:, fitix]), hm)
        np.savez_compressed(dest / 'normalization.npz', mean=mean, scale=scale, fit_indices=fitix, held_indices=held, inner_folds=inner, allowed=mask.cpu().numpy())
        preds = []
        for ix in selection['indices']:
            spec = protocol['specs'][ix]
            frozen = torch.load(PARENT / 'learning' / f'compiled_semantic_{ix:02d}' / 'continuous.pt', map_location='cpu', weights_only=False)
            epochs = int(frozen['epoch'])
            del frozen
            torch.manual_seed(spec['seed'])
            np.random.seed(spec['seed'])
            model = AtlasBcMLP(z.shape[-1], memory.shape[-1], hidden=spec['hidden']).to(device)
            model.response_mean.copy_(tensor(mean.astype(np.float32)))
            model.response_scale.fill_(scale)
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=spec['weight_decay'])
            truth = tensor(train.continuous.astype(np.float32))
            for _ in range(epochs):
                model.train()
                loss = ((model(*ta) - truth) / scale).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            model.eval()
            with torch.inference_mode():
                p = {'continuous': model(*ha).cpu().numpy().astype(float)}
                x = model.encode(*ta).cpu().numpy()
                xh = model.encode(*ha).cpu().numpy()
            torch.save({'state_dict': model.state_dict(), 'spec': spec, 'epoch': epochs}, dest / f'model{ix}.pt')
            for task in ['de', 'up']:
                ridge = selection['classification'][task]['ridge']
                torch.manual_seed(spec['seed'])
                head = fit_classification(x, train, ridge, epochs=protocol['classifier_epochs'])
                np.savez_compressed(dest / f'model{ix}_{task}.npz', **head)
                p[task] = predict_classification(xh, head)[task]
            preds.append(p)
            print(json.dumps({'stage': 'oof_seed', 'fold': fold, 'index': ix, 'epochs': epochs, 'elapsed': time.monotonic() - started}), flush=True)
        pred = {k: np.mean([p[k] for p in preds], axis=0) for k in result}
        for k in result:
            result[k][held] = pred[k]
        np.savez_compressed(dest / 'predictions.npz', **pred)
        rec = {'fold': fold, 'fit_indices': fitix.tolist(), 'held_indices': held.tolist(), 'fit_drugs': fd, 'held_drugs': hd, 'files_sha256': {p.name: sha256(p) for p in dest.iterdir() if p.is_file()}}
        write(dest / 'manifest.json', rec)
        records.append(rec)
    np.savez_compressed(out / 'predictions.npz', **result)
    write(out / 'manifest.json', {'status': 'WHOLE_DRUG_OOF_COMPLETE', 'folds': records, 'drugs': drugs, 'prediction_sha256': sha256(out / 'predictions.npz'), 'protocol_sha256': sha256(out / 'protocol.json'), 'code_sha256': sha256(Path(__file__)), 'continuous_fits': 15, 'classification_fits': 30, 'test_response_reads': 0, 'validation_response_reads': 0, 'new_api_calls': 0, 'seconds': time.monotonic() - started})
