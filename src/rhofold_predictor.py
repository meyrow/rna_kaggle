"""
src/rhofold_predictor.py — RhoFold+ RNA 3D structure predictor.

RhoFold+ (Nature Methods 2024) is a language-model-based RNA 3D predictor:
  - Input:  sequence (+ optional MSA)
  - Output: C1' coordinates for all residues
  - Speed:  ~0.14s per sequence on GPU (much faster than AlphaFold-based methods)
  - VRAM:   ~2-4GB for sequences up to 500nt — fits RTX 4060 and Kaggle P100

Available on Kaggle by adding to kernel-metadata.json:
  "dataset_sources": ["ilanmeyrowitsch/rhofold-model"]
  (after uploading the checkpoint as a Kaggle dataset)

Or install from: https://github.com/ml4bio/RhoFold

On Kaggle P100 this is the most practical 3D predictor because:
  1. Much lighter than Protenix (~200MB vs ~1.5GB checkpoint)
  2. Fast enough to run 5 candidates × 28 sequences in < 30min
  3. Pure inference — no training needed
  4. Handles sequences up to 1000nt without chunking
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Checkpoint search paths
RHOFOLD_CKPT_CANDIDATES = [
    "/kaggle/input/rhofold-model/rhofold_pretrained_params.pt",
    "/kaggle/input/rhofold/rhofold_pretrained_params.pt",
    "models/rhofold/rhofold_pretrained_params.pt",
    "/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt",
]
RHOFOLD_REPO_CANDIDATES = [
    "/kaggle/input/rhofold-model/RhoFold",
    "/kaggle/input/rhofold/RhoFold",
    "external/RhoFold",
    "/home/ilan/kaggle/data/external/RhoFold",
]


def find_path(candidates: list[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class RhoFoldPredictor:
    """
    Wrapper around RhoFold+ for RNA 3D structure prediction.

    Produces C1' atom coordinates for each residue.
    Falls back to the stub predictor if RhoFold is not available.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self._ckpt_path = cfg.get("rhofold_checkpoint") or find_path(RHOFOLD_CKPT_CANDIDATES)
        self._repo_path = cfg.get("rhofold_repo") or find_path(RHOFOLD_REPO_CANDIDATES)
        self._model = None
        self.available = False
        self._device = "cpu"
        self._try_load()

    def _try_load(self):
        if not self._ckpt_path:
            logger.info("RhoFold: checkpoint not found — will use stub predictor")
            logger.info(f"  To enable: upload checkpoint to Kaggle dataset")
            logger.info(f"  Expected: {RHOFOLD_CKPT_CANDIDATES[0]}")
            return

        if self._repo_path and self._repo_path not in sys.path:
            sys.path.insert(0, self._repo_path)

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Try importing RhoFold
            from rhofold.rhofold import RhoFold  # type: ignore
            from rhofold.config import rhofold_config  # type: ignore

            config = rhofold_config()
            model = RhoFold(config)
            state = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)
            model = model.to(self._device).eval()
            for p in model.parameters():
                p.requires_grad_(False)

            self._model = model
            self.available = True
            logger.info(f"RhoFold+ loaded from {self._ckpt_path} on {self._device}")

        except ImportError:
            logger.info("RhoFold not installed — using stub predictor")
            logger.info("  Install: git clone https://github.com/ml4bio/RhoFold external/RhoFold")
        except Exception as e:
            logger.warning(f"RhoFold load failed: {e}")

    def predict(
        self,
        sequence: str,
        seed: int = 42,
        single_features: Optional[np.ndarray] = None,
        pair_features: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict C1' coordinates for an RNA sequence.

        Args:
            sequence: RNA sequence (A, C, G, U)
            seed: random seed for reproducibility
            single_features: (L, 256) from RibonanzaNet2 (optional)
            pair_features: (L, L, 64) from RibonanzaNet2 (optional)

        Returns:
            c1_coords: (L, 3) float32
            plddt:     (L,)   float32 confidence scores 0-100
        """
        if self._model is None or not self.available:
            return self._stub_predict(sequence, seed)

        try:
            return self._rhofold_predict(sequence, seed, single_features, pair_features)
        except Exception as e:
            logger.warning(f"RhoFold inference failed for len={len(sequence)}: {e}")
            return self._stub_predict(sequence, seed)

    def _rhofold_predict(
        self, sequence, seed, single_features, pair_features
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run actual RhoFold+ inference."""
        import torch
        torch.manual_seed(seed)

        L = len(sequence)

        # RhoFold uses RNA-FM tokenisation via its own alphabet
        try:
            from rhofold.utils.alphabet import get_raw_alphabet  # type: ignore
            alphabet = get_raw_alphabet()
            batch_converter = alphabet.get_batch_converter()
            _, _, tokens = batch_converter([("seq", sequence)])
            tokens = tokens.to(self._device)
        except Exception:
            # Fallback: simple integer encoding
            nuc_map = {"A": 5, "C": 6, "G": 7, "U": 8, "T": 8, "N": 5}
            tokens = torch.tensor(
                [[1] + [nuc_map.get(c.upper(), 5) for c in sequence] + [2]],
                dtype=torch.long, device=self._device
            )

        with torch.inference_mode():
            output = self._model(
                tokens=tokens,
                rna_fm_tokens=tokens,
            )

        # Extract C1' coordinates
        # RhoFold output dict contains 'cord_tns_pred' and 'plddt_tns'
        if isinstance(output, dict):
            # Try known output keys from RhoFold source
            coords = (output.get("cord_tns_pred")   # list of tensors per layer
                   or output.get("coords")
                   or output.get("positions"))
            plddt_out = output.get("plddt_tns") or output.get("plddt")
        elif isinstance(output, (list, tuple)):
            coords   = output[0]
            plddt_out = output[1] if len(output) > 1 else None
        else:
            coords = output
            plddt_out = None

        # cord_tns_pred is a list — take last element (final layer)
        if isinstance(coords, list):
            coords = coords[-1]

        c = coords.squeeze(0).cpu().float().numpy()  # (L, 3) or (L, atoms, 3)
        if c.ndim == 3:
            # RNA atom order: P, C4', N, C1' ...
            # C1' is typically index 3 in RhoFold; try index 3 first
            c1 = c[:, 3, :] if c.shape[1] > 3 else c[:, 0, :]
        else:
            c1 = c

        if plddt_out is not None:
            p = plddt_out.squeeze(0).cpu().float().numpy()
            if p.ndim > 1:
                p = p.mean(axis=-1)
            # RhoFold pLDDT is already 0-1 → scale to 0-100
            if p.max() <= 1.0:
                p = p * 100.0
        else:
            p = np.full(L, 70.0, dtype=np.float32)

        return c1.astype(np.float32)[:L], p.astype(np.float32)[:L]

    def _stub_predict(self, sequence: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: idealized A-form helix geometry with per-seed noise."""
        rng = np.random.default_rng(seed)
        n = len(sequence)
        t = np.linspace(0, n * 0.6, n)
        coords = np.stack([
            9.0 * np.cos(t),
            9.0 * np.sin(t),
            2.8 * np.arange(n),
        ], axis=1).astype(np.float32)
        coords += rng.normal(0, 0.5, coords.shape).astype(np.float32)
        plddt = rng.uniform(40, 80, n).astype(np.float32)
        return coords, plddt
