from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F

def identity_mask(query_identities: Sequence[str], bank_identities: Sequence[str], *, excluded: torch.Tensor | None=None, device: torch.device | str | None=None) -> torch.Tensor:
    if not query_identities or not bank_identities:
        raise ValueError('Nonempty query and training Atlas identities required.')
    if len(set(bank_identities)) != len(bank_identities):
        raise ValueError('Training Atlas must have unique canonical drug identities.')
    if any((not isinstance(x, str) or not x for x in (*query_identities, *bank_identities))):
        raise ValueError('Use resolved nonempty canonical identities.')
    allowed = torch.tensor([[q != b for b in bank_identities] for q in query_identities], dtype=torch.bool, device=device)
    if excluded is not None:
        if excluded.dtype != torch.bool or excluded.shape != allowed.shape:
            raise ValueError('Whole-fold exclusions must be a matching boolean matrix.')
        allowed = allowed & ~excluded.to(device=allowed.device)
    if not allowed.any(-1).all():
        raise ValueError('Every query needs at least one independent Atlas donor.')
    return allowed

class UnifiedAtlasCondition(nn.Module):

    def __init__(self, knowledge_dimensions: int, observation_dimensions: int, *, cells: int=3, hidden: int=64, output_dimensions: int=32, dropout: float=0.1):
        super().__init__()
        if min(knowledge_dimensions, observation_dimensions, cells, hidden, output_dimensions) < 1:
            raise ValueError('Positive feature, cell and hidden dimensions required.')
        self.knowledge_dimensions = knowledge_dimensions
        self.observation_dimensions = observation_dimensions
        self.cells = cells
        self.hidden = hidden
        self.output_dimensions = output_dimensions
        self.query_map = nn.Linear(knowledge_dimensions, hidden)
        self.key_map = nn.Linear(knowledge_dimensions, hidden, bias=False)
        self.value_map = nn.Sequential(nn.Linear(observation_dimensions, hidden), nn.GELU())
        self.biological_prior_weights = nn.Parameter(torch.zeros(3))
        self.context_map = nn.Linear(hidden, hidden, bias=False)
        self.context_gate = nn.Linear(hidden, hidden)
        self.condition_norm = nn.LayerNorm(hidden)
        self.condition_mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, output_dimensions)) for _ in range(cells)])

    def forward(self, query: torch.Tensor, bank_knowledge: torch.Tensor, bank_observations: torch.Tensor, similarities: torch.Tensor, allowed: torch.Tensor, *, return_retrieval: bool=False):
        n, c, f = query.shape
        m = bank_knowledge.shape[0]
        if (c, f) != (self.cells, self.knowledge_dimensions):
            raise ValueError('Query feature/cell contract differs from fitted frontend.')
        if bank_knowledge.shape != (m, c, f):
            raise ValueError('Atlas knowledge must share the query feature/cell contract.')
        if bank_observations.shape != (m, c, self.observation_dimensions):
            raise ValueError('Atlas observation dimensions or cell order mismatch.')
        if similarities.shape != (n, m, 3):
            raise ValueError('Three response-blind similarity channels required.')
        if allowed.shape != (n, m) or allowed.dtype != torch.bool:
            raise ValueError('Supply an explicit identity/fold donor mask.')
        if not allowed.any(-1).all():
            raise ValueError('All donors excluded for at least one query.')
        for value in (query, bank_knowledge, bank_observations, similarities):
            if not torch.isfinite(value).all():
                raise ValueError('Knowledge, observations and priors must be finite.')
        if (similarities < 0).any() or (similarities > 1).any():
            raise ValueError('Similarity priors must lie in [0, 1].')
        q = self.query_map(query)
        k = self.key_map(bank_knowledge)
        v = self.value_map(bank_observations)
        scores = torch.einsum('nch,mch->ncm', q, k) / self.hidden ** 0.5
        prior = torch.einsum('nmk,k->nm', similarities, F.softplus(self.biological_prior_weights))
        scores = (scores + prior[:, None, :]).masked_fill(~allowed[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum('ncm,mch->nch', weights, v)
        unified = self.condition_norm(q + torch.sigmoid(self.context_gate(q)) * self.context_map(context))
        representation = torch.stack([model(unified[:, ci]) for ci, model in enumerate(self.condition_mlp)], dim=1)
        if return_retrieval:
            return (representation, {'weights': weights, 'unified_condition': unified})
        return representation

class AtlasBcMLP(nn.Module):

    def __init__(self, knowledge_dimensions: int, observation_dimensions: int, *, cells: int=3, hidden: int=64, latent: int=32, genes: int=2000, dropout: float=0.1):
        super().__init__()
        self.frontend = UnifiedAtlasCondition(knowledge_dimensions, observation_dimensions, cells=cells, hidden=hidden, output_dimensions=latent, dropout=dropout)
        self.bc_mlp = nn.ModuleList([nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, genes)) for _ in range(cells)])
        for block in self.bc_mlp:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
        self.register_buffer('response_mean', torch.zeros(cells, genes))
        self.register_buffer('response_scale', torch.ones(()))

    def encode(self, *args, **kwargs):
        return self.frontend(*args, **kwargs)

    def forward(self, *args, return_retrieval: bool=False, **kwargs):
        if return_retrieval:
            z, info = self.encode(*args, return_retrieval=True, **kwargs)
        else:
            z = self.encode(*args, **kwargs)
        y = torch.stack([block(z[:, ci]) for ci, block in enumerate(self.bc_mlp)], dim=1)
        y = self.response_mean + self.response_scale * y
        return (y, info) if return_retrieval else y
