"""utils/tm_score.py — TM-score computation utilities."""

import subprocess
import tempfile
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_tm_score(
    pred_coords: np.ndarray,
    ref_coords: np.ndarray,
    sequence: str,
    use_usalign: bool = True,
) -> float:
    """
    Compute TM-score between predicted and reference C1' coordinates.

    Args:
        pred_coords: shape (L, 3) predicted C1' coordinates
        ref_coords: shape (L, 3) reference C1' coordinates
        sequence: RNA sequence (for PDB file writing)
        use_usalign: if True, use US-align binary; otherwise use numpy approximation

    Returns:
        TM-score in [0, 1]
    """
    if use_usalign:
        return _tm_usalign(pred_coords, ref_coords, sequence)
    else:
        return _tm_approx(pred_coords, ref_coords)


def _tm_usalign(pred: np.ndarray, ref: np.ndarray, sequence: str) -> float:
    """Run US-align for TM-score computation (competition standard)."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_pdb = os.path.join(tmpdir, "pred.pdb")
            ref_pdb = os.path.join(tmpdir, "ref.pdb")
            _write_c1_pdb(pred, sequence, pred_pdb)
            _write_c1_pdb(ref, sequence, ref_pdb)

            result = subprocess.run(
                ["USalign", pred_pdb, ref_pdb, "-mol", "RNA"],
                capture_output=True, text=True, timeout=30
            )
            # Parse TM-score from output
            for line in result.stdout.split("\n"):
                if line.startswith("TM-score="):
                    parts = line.split()
                    return float(parts[1])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning(f"US-align failed: {e}. Using approximation.")
    return _tm_approx(pred, ref)


def _tm_approx(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    Fast numpy TM-score approximation (not competition-accurate).
    Uses the analytical formula after optimal superposition (Kabsch algorithm).
    """
    if pred.shape != ref.shape:
        return 0.0
    L = len(pred)
    d0 = 0.5 if L <= 15 else 1.24 * (L - 15) ** (1 / 3) - 1.8
    d0 = max(d0, 0.5)

    # Center both structures
    p = pred - pred.mean(0)
    r = ref - ref.mean(0)

    # Kabsch rotation
    H = p.T @ r
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    p_rot = p @ R.T

    # TM-score
    dists = np.sum((p_rot - r) ** 2, axis=1)
    tm = np.sum(1.0 / (1.0 + dists / d0 ** 2)) / L
    return float(tm)


def _write_c1_pdb(coords: np.ndarray, sequence: str, path: str):
    """Write C1' coordinates as a minimal PDB file."""
    resname_map = {"A": "  A", "C": "  C", "G": "  G", "U": "  U"}
    with open(path, "w") as f:
        for i, (res, xyz) in enumerate(zip(sequence, coords)):
            resname = resname_map.get(res.upper(), "  N")
            f.write(
                f"ATOM  {i+1:5d}  C1' {resname} A{i+1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"  1.00  0.00           C\n"
            )
        f.write("END\n")
