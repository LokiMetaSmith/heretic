import torch
from torch import Tensor
from typing import Dict, Tuple

def precompute_ridge_terms(
    good_activations: Dict[str, Tensor],
    bad_activations: Dict[str, Tensor]
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """
    Precomputes the covariance matrix X^T X and target vector X^T y for each component.
    """
    terms = {}
    for key in good_activations.keys():
        X_good = good_activations[key].to(torch.float32)
        X_bad = bad_activations[key].to(torch.float32)

        X = torch.cat([X_good, X_bad], dim=0)
        y = torch.cat([torch.zeros(X_good.shape[0]), torch.ones(X_bad.shape[0])], dim=0).to(torch.float32)

        XX = torch.matmul(X.T, X)
        Xy = torch.matmul(X.T, y)

        terms[key] = (XX, Xy)

    return terms

def train_ridge_probes(
    precomputed_terms: Dict[str, Tuple[Tensor, Tensor]],
    alpha: float = 1.0
) -> Dict[str, Tensor]:
    """
    Trains a Ridge Regression probe for each component using precomputed terms.
    """
    probes = {}

    for key, (XX, Xy) in precomputed_terms.items():
        D = XX.shape[0]
        I = torch.eye(D, device=XX.device, dtype=torch.float32)

        A = XX + alpha * I
        w_probe = torch.linalg.solve(A, Xy)
        probes[key] = w_probe

    return probes
