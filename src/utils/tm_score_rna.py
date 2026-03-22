"""
RNA-specific TM-score using competition formula.
d0 = 0.6*(Lref-0.5)^0.5 - 2.5  for Lref >= 30
d0 = 0.3/0.4/0.5/0.6/0.7        for Lref < 30
"""
import numpy as np


def d0_rna(L: int) -> float:
    if L >= 30: return 0.6 * (L - 0.5) ** 0.5 - 2.5
    elif L < 15: return 0.3
    elif L < 18: return 0.4
    elif L < 20: return 0.5
    elif L < 24: return 0.6
    else:        return 0.7


def tm_score_rna(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    Competition-accurate TM-score for RNA using US-align formula.
    pred, ref: (L, 3) float32 arrays — C1' coordinates.
    L_ref = len(ref), normalization is by L_ref.
    """
    if len(pred) == 0 or len(ref) == 0:
        return 0.0

    L_ref = len(ref)
    L     = min(len(pred), len(ref))
    d0    = d0_rna(L_ref)

    p = pred[:L].copy()
    r = ref[:L].copy()

    # Center
    p -= p.mean(0)
    r -= r.mean(0)

    # Kabsch rotation
    H        = p.T @ r
    U, S, Vt = np.linalg.svd(H)
    d        = np.linalg.det(Vt.T @ U.T)
    D        = np.diag([1.0, 1.0, d])
    R        = Vt.T @ D @ U.T
    p_rot    = p @ R.T

    dists = np.sum((p_rot - r) ** 2, axis=1)
    tm    = np.sum(1.0 / (1.0 + dists / d0 ** 2)) / L_ref
    return float(tm)
