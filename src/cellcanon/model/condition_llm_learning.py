from __future__ import annotations
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch
from torch.nn import functional as F
from cellcanon.model.metrics import row_correlation, safe_ap

def fit_classification(x, panel, ridge, epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    n, cells, genes = panel.de.shape
    if not np.isin(panel.de, [0, 1]).all() or not np.isin(panel.up[panel.de == 1], [0, 1]).all():
        raise ValueError('DE and conditional direction labels must be observed binary outcomes.')
    de = tensor(panel.de)
    mask = de
    masked_up = np.where(panel.de == 1, panel.up, 0.0)
    up = tensor(masked_up)
    prior_de = (panel.de.sum(0) + 0.5) / (n + 1)
    prior_up = (masked_up.sum(0) + 0.5) / (panel.de.sum(0) + 1)
    off_de = tensor(np.log(prior_de / (1 - prior_de)))
    off_up = tensor(np.log(prior_up / (1 - prior_up)))
    features = tensor(x)
    beta_de = torch.nn.Parameter(torch.zeros((cells, x.shape[-1], genes), device=device))
    beta_up = torch.nn.Parameter(torch.zeros_like(beta_de))
    intercept_de = torch.nn.Parameter(torch.zeros((cells, genes), device=device))
    intercept_up = torch.nn.Parameter(torch.zeros_like(intercept_de))
    parameters = [beta_de, beta_up, intercept_de, intercept_up]
    optimizer = torch.optim.Adam(parameters, lr=0.04)
    losses = []
    for epoch in range(epochs):
        zd = off_de + intercept_de + torch.einsum('dcr,crg->dcg', features, beta_de)
        zu = off_up + intercept_up + torch.einsum('dcr,crg->dcg', features, beta_up)
        likelihood = F.binary_cross_entropy_with_logits(zd, de)
        likelihood += (F.binary_cross_entropy_with_logits(zu, up, reduction='none') * mask).sum() / mask.sum().clamp_min(1)
        penalty = beta_de.square().sum(1).mean() + beta_up.square().sum(1).mean()
        penalty += intercept_de.square().mean() + intercept_up.square().mean()
        loss = likelihood + ridge / (2 * n) * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in (0, 49, 99, epochs - 1):
            losses.append(float(loss.detach().cpu()))
    model = {name: value.detach().cpu().numpy() for name, value in zip(['beta_de', 'beta_up', 'intercept_de', 'intercept_up'], parameters)}
    model.update(offset_de=off_de.cpu().numpy(), offset_up=off_up.cpu().numpy(), losses=np.array(losses))
    if not all((np.isfinite(v).all() for v in model.values())):
        raise ValueError('Nonfinite classification fit.')
    return model

def predict_classification(x, model):
    return {key: expit(model['offset_' + key] + model['intercept_' + key] + np.einsum('dcr,crg->dcg', x, model['beta_' + key])) for key in ('de', 'up')}

def continuous_score(y, p):
    within = row_correlation(rankdata(y.transpose(1, 2, 0), axis=-1), rankdata(p.transpose(1, 2, 0), axis=-1))
    return {'pearson': float(np.nan_to_num(row_correlation(y, p)).mean()), 'rmse': float(np.sqrt(((y - p) ** 2).mean(-1)).mean()), 'mae': float(np.abs(y - p).mean()), 'within_gene_spearman': float(np.nan_to_num(within).mean()), 'direction_accuracy': float((np.sign(y) == np.sign(p)).mean())}

def classification_score(panel, p, key):
    result = {}
    for scope, sl in [('core', slice(0, 2000)), ('extra', slice(2000, len(panel.genes)))]:
        if key == 'de':
            y = panel.de[..., sl]
            pred = p[..., sl]
            sample = panel.sampled[..., sl]
            result[scope + '_dense_ap'] = float(safe_ap(y.ravel(), pred.ravel()))
            result[scope + '_sampled_ap'] = float(safe_ap(y[sample], pred[sample]))
            result[scope + '_brier'] = float(np.square(y - pred).mean())
        else:
            mask = panel.de[..., sl] == 1
            y = panel.up[..., sl][mask]
            pred = p[..., sl][mask]
            result[scope + '_dir_ap'] = float((safe_ap(y, pred) + safe_ap(1 - y, 1 - pred)) / 2)
            result[scope + '_brier'] = float(np.square(y - pred).mean())
    return result

def utility(metric, task, scale=1.0):
    if task == 'continuous':
        return metric['pearson'] - metric['rmse'] / scale + 0.25 * metric['within_gene_spearman']
    ap = [v for k, v in metric.items() if k.endswith('_ap')]
    brier = [v for k, v in metric.items() if k.endswith('_brier')]
    return float(np.nanmean(ap) - 0.1 * np.nanmean(brier))
