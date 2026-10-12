from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from cellcanon.model.condition_llm_learning import fit_classification, predict_classification, classification_score, continuous_score, utility
from cellcanon.model.condition_llm_packets import write
from cellcanon.model.data import CELLS, load_panel, sha256
from cellcanon.model.decoding import fit_matched_decoder, apply_matched_decoder
from cellcanon.model.isolation import subset_by_drugs
from cellcanon.model.screen import clean_json
from cellcanon.model.unified_atlas_condition import AtlasBcMLP, identity_mask
ROOT = Path('<RUN_ROOT>') / 'model'
VIEWS = ('compiled_semantic',)

def read(path):
    return json.loads(Path(path).read_text())

def log(**values):
    print(json.dumps(clean_json(values)), flush=True)

def checked(path, expected):
    if sha256(Path(path)) != expected:
        raise ValueError('Frozen input changed: ' + str(path))

def source_hashes():
    return {}

class Inputs:

    def __init__(self, root, view, device='cpu'):
        self.root = Path(root)
        self.device = device
        inp = self.root / 'inputs'
        self.manifest = read(inp / 'manifest.json')
        m = self.manifest
        for name, h in m['files_sha256'].items():
            checked(inp / name, h)
        self.tensor = lambda x: torch.as_tensor(x, device=device)
        z = np.load(inp / (view + '.npz'))['knowledge']
        self.z = self.tensor(z)
        self.sim = self.tensor(np.load(inp / 'similarities.npz')['values'])
        bank = dict(np.load(inp / 'atlas.npz'))
        self.bank = self.tensor(bank['observations'])
        self.mean = self.tensor(bank['continuous_mean'].astype(np.float32))
        self.scale = float(bank['continuous_scale'])
        self.nt = len(m['train_drugs'])
        self.training_allowed = self.tensor(bank['train_allowed'])
        self.indices = {d: i for i, d in enumerate(m['all_drugs'])}

    def args(self, drugs, *, training=False):
        ix = [self.indices[d] for d in drugs]
        ids = self.manifest['identity_groups']
        bank = self.manifest['train_drugs']
        if training:
            if list(drugs) != bank:
                raise ValueError('Training queries must match the response atlas drug order.')
            allowed = self.training_allowed
        else:
            if {ids[d] for d in drugs} & {ids[d] for d in bank}:
                raise ValueError('Inference drug overlaps training identity.')
            allowed = identity_mask([ids[d] for d in drugs], [ids[d] for d in bank], device=self.device)
        return (self.z[ix], self.z[:self.nt], self.bank, self.sim[ix], allowed)

def new_model(data, spec):
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    model = AtlasBcMLP(data.z.shape[-1], data.bank.shape[-1], hidden=spec['hidden']).to(data.device)
    model.response_mean.copy_(data.mean)
    model.response_scale.fill_(data.scale)
    return model

def load_model(data, path):
    state = torch.load(path, map_location=data.device, weights_only=False)
    model = new_model(data, state['spec'])
    model.load_state_dict(state['state_dict'])
    return model.eval()

def fit(root=ROOT):
    root = Path(root)
    p = read(root / 'protocol.json')
    out = root / 'learning'
    out.mkdir(exist_ok=True)
    if (out / 'selection.json').exists():
        raise FileExistsError('Fitted models and their selection already exist. Use a new model directory to train again.')
    for path, h in p['code_sha256'].items():
        checked(path, h)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = read(root / 'inputs/manifest.json')
    train = subset_by_drugs(load_panel('support'), m['train_drugs'])
    valid = subset_by_drugs(load_panel('validation'), m['validation_drugs'])
    records = []
    selected = {}
    for view in VIEWS:
        data = Inputs(root, view, device)
        ta = data.args(train.drugs, training=True)
        va = data.args(valid.drugs)
        truth = torch.as_tensor(train.continuous, dtype=torch.float32, device=device)
        for i, spec in enumerate(p['specs']):
            dest = out / f'{view}_{i:02d}'
            dest.mkdir(exist_ok=True)
            if (dest / 'fit.json').exists():
                record = read(dest / 'fit.json')
                records.append(record)
                continue
            model = new_model(data, spec)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=spec['weight_decay'])
            best = -float('inf')
            best_state = None
            history = []
            start = time.monotonic()
            for epoch in range(1, p['epochs'] + 1):
                model.train()
                pred = model(*ta)
                loss = ((pred - truth) / data.scale).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if epoch in p['checkpoints']:
                    model.eval()
                    with torch.inference_mode():
                        value = model(*va).cpu().numpy().astype(float)
                    metrics = continuous_score(valid.continuous, value)
                    score = utility(metrics, 'continuous', data.scale)
                    history.append({'epoch': epoch, 'train_loss': float(loss.detach().cpu()), 'validation': metrics, 'utility': score})
                    if score > best:
                        best = score
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        best_pred = value
            torch.save({'spec': spec, 'state_dict': best_state, 'epoch': best_epoch}, dest / 'continuous.pt')
            np.savez_compressed(dest / 'validation.npz', continuous=best_pred)
            record = {'view': view, 'index': i, 'spec': spec, 'selected_epoch': best_epoch, 'history': history, 'seconds': time.monotonic() - start, 'device': device, 'parameters': sum((p.numel() for p in model.parameters()))}
            write(dest / 'fit.json', clean_json(record))
            records.append(record)
            log(stage='continuous_fit', view=view, index=i, epoch=best_epoch, rmse=history[[x['epoch'] for x in history].index(best_epoch)]['validation']['rmse'])
        best = (-float('inf'), None, None)
        for start in range(0, len(p['specs']), 3):
            ix = list(range(start, start + 3))
            predictions = [np.load(out / f'{view}_{i:02d}' / 'validation.npz')['continuous'] for i in ix]
            value = np.mean(predictions, axis=0)
            metrics = continuous_score(valid.continuous, value)
            score = utility(metrics, 'continuous', data.scale)
            if score > best[0]:
                best = (score, ix, {'metrics': metrics, 'prediction': value})
        indices = best[1]
        class_preds = {}
        class_records = []
        for i in indices:
            dest = out / f'{view}_{i:02d}'
            model = load_model(data, dest / 'continuous.pt')
            with torch.inference_mode():
                x = model.encode(*ta).cpu().numpy()
                xv = model.encode(*va).cpu().numpy()
            for ridge in p['classifier_ridges']:
                torch.manual_seed(p['specs'][i]['seed'])
                head = fit_classification(x, train, ridge, epochs=p['classifier_epochs'])
                name = f'classification_r{ridge:g}.npz'
                np.savez_compressed(dest / name, **head)
                pred = predict_classification(xv, head)
                class_preds[i, ridge] = pred
                class_records.append({'index': i, 'ridge': ridge, 'head': str(dest / name)})
        final = {'continuous': best[2]['prediction']}
        choices = {}
        for task in ('de', 'up'):
            options = []
            for ridge in p['classifier_ridges']:
                value = np.mean([class_preds[i, ridge][task] for i in indices], axis=0)
                metrics = classification_score(valid, value, task)
                options.append((utility(metrics, task), ridge, value, metrics))
            winner = max(options, key=lambda x: x[0])
            final[task] = winner[2]
            choices[task] = {'ridge': winner[1], 'validation': winner[3]}
        decoder = fit_matched_decoder(valid, final['de'])
        final['hard_de'] = apply_matched_decoder(valid.drugs, valid.genes, final['de'], decoder)
        write(out / (view + '_decoder.json'), clean_json(decoder))
        np.savez_compressed(out / (view + '_validation.npz'), **final)
        selected[view] = {'indices': indices, 'continuous_validation': best[2]['metrics'], 'classification': choices, 'head_fits': class_records}
        log(stage='view_selected', view=view, indices=indices, rmse=best[2]['metrics']['rmse'])
    write(out / 'selection.json', clean_json({'status': 'UNIFIED_MODELS_SELECTED_BEFORE_TEST', 'selected': selected, 'continuous_fits': len(records), 'classification_fits': sum((len(s['head_fits']) for s in selected.values())), 'protocol_sha256': sha256(root / 'protocol.json'), 'code_sha256': source_hashes(), 'files_sha256': {str(q.relative_to(out)): sha256(q) for q in out.rglob('*') if q.is_file()}, 'old_predictor_outputs_used': False, 'new_api_calls': 0, 'test_truth_reads': 0}))

class UnifiedAtlasPredictor:

    def __init__(self, root=ROOT, *, view='compiled_semantic', device='cpu'):
        self.root = Path(root)
        self.view = view
        self.data = Inputs(root, view, device)
        selection = read(self.root / 'learning/selection.json')
        for path, h in selection['code_sha256'].items():
            checked(path, h)
        for name, h in selection['files_sha256'].items():
            checked(self.root / 'learning' / name, h)
        self.choice = selection['selected'][view]
        self.models = []
        self.classifiers = []
        for i in self.choice['indices']:
            path = self.root / 'learning' / f'{view}_{i:02d}'
            self.models.append(load_model(self.data, path / 'continuous.pt'))
            self.classifiers.append({task: dict(np.load(path / f"classification_r{self.choice['classification'][task]['ridge']:g}.npz")) for task in ('de', 'up')})
        self.decoder = read(self.root / 'learning' / (view + '_decoder.json'))

    def predict(self, drugs, *, cells=CELLS):
        drugs = list(drugs)
        cells = list(cells)
        m = self.data.manifest
        if len(set(drugs)) != len(drugs) or not drugs:
            raise ValueError('Unique nonempty drug queries required.')
        if len(set(cells)) != len(cells) or not cells or (not set(cells) <= set(CELLS)):
            raise ValueError('Known unique cell lines required.')
        if self.view == 'compiled_semantic' and (not set(drugs) <= set(m['validation_drugs'] + m['compiled_test_drugs'])):
            raise ValueError('Requested drugs must be in validation_drugs or compiled_test_drugs in inputs/manifest.json.')
        args = list(self.data.args(drugs))
        output = {k: [] for k in ('continuous', 'de', 'up')}
        attention = []
        with torch.inference_mode():
            for model, head in zip(self.models, self.classifiers):
                y, info = model(*args, return_retrieval=True)
                z = model.encode(*args).cpu().numpy()
                output['continuous'].append(y.cpu().numpy().astype(float))
                attention.append(info['weights'].cpu().numpy())
                for task in ('de', 'up'):
                    output[task].append(predict_classification(z, head[task])[task])
        pred = {k: np.mean(v, axis=0) for k, v in output.items()}
        pred['hard_de'] = apply_matched_decoder(drugs, m['genes'], pred['de'], self.decoder)
        ci = [CELLS.index(c) for c in cells]
        return {'drugs': drugs, 'cells': cells, 'predictions': {k: v[:, ci] for k, v in pred.items()}, 'retrieval_weights': np.mean(attention, axis=0)[:, ci], 'atlas_drugs': m['train_drugs'], 'continuous_genes': m['genes'][:2000], 'classification_genes': m['genes'], 'old_predictor_outputs_used': False, 'query_response_labels_used': False}

def freeze(root=ROOT):
    root = Path(root)
    out = root / 'freeze'
    if out.exists():
        raise FileExistsError('Prediction outputs already exist in the model freeze directory.')
    out.mkdir()
    m = read(root / 'inputs/manifest.json')
    predictor = UnifiedAtlasPredictor(root, view='compiled_semantic', device='cuda' if torch.cuda.is_available() else 'cpu')
    result = predictor.predict(m['test_drugs'])
    np.savez_compressed(out / 'compiled_semantic.npz', **result['predictions'])
    np.savez_compressed(out / 'retrieval.npz', weights=result['retrieval_weights'])
    write(out / 'manifest.json', {'scopes': {'compiled_semantic': m['test_drugs']}, 'selection_sha256': sha256(root / 'learning/selection.json'), 'files_sha256': {p.name: sha256(p) for p in out.glob('*.npz')}, 'test_truth_reads': 0})
