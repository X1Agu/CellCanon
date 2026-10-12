from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

@dataclass(frozen=True)
class ResidualObservation:
    condition: int
    genes: np.ndarray
    residual: np.ndarray
    variance: np.ndarray

class ResidualUpdater:

    def __init__(self, residuals, *, rank=16, noise_ratio=1.0, neighbors=8):
        r = np.asarray(residuals, dtype=float)
        if r.ndim != 3 or len(r) < 3 or (not np.isfinite(r).all()):
            raise ValueError('Complete finite support residuals N x condition x gene required.')
        if not np.isfinite(noise_ratio) or noise_ratio <= 0 or neighbors < 1 or (rank < 1) or (int(rank) != rank) or (int(neighbors) != neighbors):
            raise ValueError('Positive regularization, neighbor count and rank required.')
        self.residuals = r.copy()
        self.residuals.setflags(write=False)
        self.requested_rank = int(rank)
        self.shape = r.shape[1:]
        self.noise_ratio = float(noise_ratio)
        self.neighbors = min(int(neighbors), len(r))
        flat = r.reshape(len(r), -1)
        gram = flat @ flat.T / max(1, len(r) - 1)
        values, vectors = np.linalg.eigh(gram)
        keep = np.argsort(values)[::-1][:min(rank, len(r) - 1)]
        keep = keep[values[keep] > 1e-10]
        self.basis = flat.T @ vectors[:, keep] / np.sqrt(max(1, len(r) - 1))
        self.second_moment = np.mean(r * r, axis=0)
        floor = max(float(np.median(self.second_moment)) * 0.0001, 1e-06)
        self.diagonal = np.maximum(self.second_moment.ravel() - np.sum(self.basis ** 2, axis=1), floor)
        self.scale = np.maximum(np.std(r, axis=0), np.sqrt(floor))

    def _observed(self, observations):
        records = {}
        for obs in observations:
            c = int(obs.condition)
            raw_genes = np.asarray(obs.genes)
            if not np.issubdtype(raw_genes.dtype, np.integer) or c != obs.condition:
                raise ValueError('Observation coordinates must be integers.')
            genes = raw_genes.astype(int)
            values = np.asarray(obs.residual, dtype=float)
            variance = np.asarray(obs.variance, dtype=float)
            if not 0 <= c < self.shape[0] or genes.ndim != 1 or values.shape != genes.shape or (variance.shape != genes.shape) or (len(np.unique(genes)) != len(genes)) or (not len(genes)) or np.any(genes < 0) or np.any(genes >= self.shape[1]) or (not np.isfinite(values).all()) or (not np.isfinite(variance).all()) or np.any(variance < 0):
                raise ValueError('Invalid partial residual observation.')
            for g, v, var in zip(genes, values, variance):
                idx = c * self.shape[1] + int(g)
                records.setdefault(idx, []).append((float(v), max(float(var), 1e-06)))
        indices = np.array(sorted(records), dtype=int)
        values = np.zeros(len(indices))
        variance = np.zeros(len(indices))
        for j, idx in enumerate(indices):
            rows = np.asarray(records[idx])
            w = 1 / rows[:, 1]
            values[j] = np.sum(w * rows[:, 0]) / w.sum()
            variance[j] = 1 / w.sum()
        return (indices, values, variance)

    def infer(self, observations, *, mode='gaussian', strength=1.0):
        if mode != 'gaussian':
            raise ValueError('Only the selected Gaussian residual updater is included.')
        if not 0 <= strength <= 1:
            raise ValueError('Update strength must be in [0, 1].')
        indices, values, noise = self._observed(observations)
        prior_var = self.second_moment.copy()
        if not len(indices) or strength == 0:
            return (np.zeros(self.shape), prior_var, {'observed_coordinates': len(indices), 'mode': mode, 'strength': strength})
        b = self.basis[indices]
        d = self.noise_ratio * self.diagonal[indices] + noise
        precision = np.eye(b.shape[1]) + b.T / d @ b
        covariance = solve(precision, np.eye(len(precision)), assume_a='pos')
        latent = covariance @ (b.T @ (values / d))
        gaussian = (self.basis @ latent).reshape(self.shape)
        post_var = (self.diagonal + np.sum(self.basis @ covariance * self.basis, axis=1)).reshape(self.shape)
        normalized = (self.residuals.reshape(len(self.residuals), -1)[:, indices] - values) / self.scale.ravel()[indices]
        distances = np.mean(normalized ** 2, axis=1)
        chosen = np.argsort(distances, kind='stable')[:self.neighbors]
        temperature = max(float(np.median(distances)), 1e-08)
        weights = np.exp(-(distances[chosen] - distances[chosen].min()) / temperature)
        weights /= weights.sum()
        delta = gaussian
        variance = (1 - strength) * prior_var + strength * post_var + strength * (1 - strength) * delta ** 2
        return (strength * delta, variance, dict(mode=mode, strength=float(strength), observed_coordinates=len(indices), empirical_uncertainty=True, nearest_support_indices=chosen.tolist(), weights=weights.tolist(), nearest_distance=float(distances[chosen[0]]), residual_rms=float(np.sqrt(np.mean(values ** 2))), proposed_rms=float(np.sqrt(np.mean(delta ** 2)))))

    def save(self, path):
        np.savez_compressed(path, residuals=self.residuals, rank=self.requested_rank, noise_ratio=self.noise_ratio, neighbors=self.neighbors)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(data['residuals'], rank=int(data['rank']), noise_ratio=float(data['noise_ratio']), neighbors=int(data['neighbors']))
