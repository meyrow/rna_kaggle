"""
src/rhofold_predictor.py — RhoFold+ RNA 3D structure predictor.

RhoFold+ (Nature Methods 2024) is a language-model-based RNA 3D predictor:
  - Input:  sequence (single-sequence mode, no MSA required)
  - Output: C1' coordinates for all residues + pLDDT confidence
  - Speed:  ~0.14s per sequence on GPU
  - VRAM:   ~2-4GB for sequences up to 500nt

RhoFold forward API (from rhofold/rhofold.py):
  model.forward(tokens, rna_fm_tokens, seq)
  - tokens / rna_fm_tokens: shape (bs, msa_depth, L) — same for single-seq
  - seq: list of residue-type ints (0=A, 1=C, 2=G, 3=U)
  Output keys: 'cords' (C1' coords), 'plddt'

Checkpoint: https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt
Repo:       https://github.com/ml4bio/RhoFold
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 0}

# Checkpoint search paths
RHOFOLD_CKPT_CANDIDATES = [
    "/kaggle/input/rhofold-model/rhofold_pretrained_params.pt",
    "/kaggle/input/rhofold-model/RhoFold/pretrained/rhofold_pretrained_params.pt",
    "models/rhofold/rhofold_pretrained_params.pt",
    "/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt",
]
RHOFOLD_REPO_CANDIDATES = [
    "/kaggle/input/rhofold-model/RhoFold",
    "external/RhoFold",
    "/home/ilan/kaggle/data/external/RhoFold",
]


def find_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class RhoFoldPredictor:
    """RhoFold+ wrapper for RNA 3D structure prediction."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self._ckpt_path = cfg.get("rhofold_checkpoint") or find_path(RHOFOLD_CKPT_CANDIDATES)
        self._repo_path = cfg.get("rhofold_repo")       or find_path(RHOFOLD_REPO_CANDIDATES)
        self._model  = None
        self._device = "cpu"
        self.available = False
        self._try_load()

    def _try_load(self):
        if not self._ckpt_path:
            logger.warning("RhoFold: checkpoint not found")
            logger.warning(f"  Expected: {RHOFOLD_CKPT_CANDIDATES[0]}")
            return

        # Add repo root to sys.path so 'import rhofold' works
        if self._repo_path:
            if self._repo_path not in sys.path:
                sys.path.insert(0, self._repo_path)
        else:
            logger.warning("RhoFold: repo path not found")
            logger.warning(f"  Expected: {RHOFOLD_REPO_CANDIDATES[0]}")
            return

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            from rhofold.rhofold import RhoFold        # type: ignore
            from rhofold.config import rhofold_config  # type: ignore

            config = rhofold_config  # ConfigDict, not callable
            model  = RhoFold(config)

            state = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)
            model = model.to(self._device).eval()
            for p in model.parameters():
                p.requires_grad_(False)

            self._model    = model
            self.available = True
            logger.info(f"RhoFold+ ready on {self._device} ✓")

        except Exception as e:
            logger.warning(f"RhoFold load failed: {type(e).__name__}: {e}")
            logger.warning(f"  ckpt : {self._ckpt_path}")
            logger.warning(f"  repo : {self._repo_path}")

    # ── Public API ──────────────────────────────────────────────────────

    def predict(
        self,
        sequence: str,
        seed: int = 42,
        single_features=None,
        pair_features=None,
    ) -> tuple:
        """
        Predict C1' coordinates for an RNA sequence.
        Returns (c1_coords: (L,3) float32, plddt: (L,) float32).
        """
        if self._model is None or not self.available:
            return self._stub_predict(sequence, seed)
        try:
            return self._rhofold_predict(sequence, seed)
        except Exception as e:
            logger.warning(f"RhoFold inference failed (len={len(sequence)}): {e}")
            return self._stub_predict(sequence, seed)

    # ── Internal ────────────────────────────────────────────────────────

    def _rhofold_predict(self, sequence: str, seed: int):
        import torch
        torch.manual_seed(seed)
        L = len(sequence)

        # ── Tokenise (no BOS/EOS — both tensors must be length L) ──────
        # embedders.py: torch.cat([rna_fm_output, msa_fea], dim=-1)
        # rna_fm output length == rna_fm_tokens length
        # msa_fea length == tokens length (stripped)
        # → both must be L, no BOS/EOS padding

        seq_int = [NUC_TO_IDX.get(c.upper(), 0) for c in sequence]

        # RNA-FM token IDs without BOS/EOS: A=5, C=6, G=7, U=8
        nuc_fm = {"A": 5, "C": 6, "G": 7, "U": 8, "T": 8, "N": 5}
        raw_ids = [nuc_fm.get(c.upper(), 5) for c in sequence]   # length L
        raw_tok = torch.tensor([raw_ids], dtype=torch.long, device=self._device)

        rna_fm_tokens = raw_tok            # (1, L)   2D — RNA-FM model
        tokens        = raw_tok.unsqueeze(1)  # (1, 1, L) 3D — MSAEmbedder

        # ── Forward pass ──────────────────────────────────────────────
        with torch.inference_mode():
            output = self._model(
                tokens=tokens,
                rna_fm_tokens=rna_fm_tokens,   # 2D for RNA-FM
                seq=seq_int,
            )

        # ── Extract C1' coordinates ───────────────────────────────────
        # forward() collects outputs across recycle iterations → list of dicts
        # Take the last element (final prediction)
        if isinstance(output, list):
            output = output[-1]

        if isinstance(output, dict):
            # structure_module stores coords in 'cord_tns_pred' (list of tensors)
            cords = output.get("cord_tns_pred") or output.get("cords") or output.get("cord")
        else:
            # Unexpected type — print for debugging
            logger.warning(f"Unexpected output type: {type(output)}: {output}")
            return self._stub_predict(sequence, seed)

        if cords is None:
            logger.warning(f"RhoFold output keys: {list(output.keys())}")
            return self._stub_predict(sequence, seed)

        # cord_tns_pred is a list of tensors (one per recycle layer) — take last
        if isinstance(cords, list):
            cords = cords[-1]

        c = cords.squeeze(0).cpu().float().numpy()   # (L, atoms, 3) or (L, 3)
        if c.ndim == 3:
            # RhoFold atom ordering: [P, C4', N1/N9, C1', ...]  C1' = index 3
            c1 = c[:, 3, :] if c.shape[1] > 3 else c[:, 0, :]
        else:
            c1 = c

        # ── Extract pLDDT ─────────────────────────────────────────────
        plddt_out = output.get("plddt") if isinstance(output, dict) else None
        if plddt_out is not None:
            # plddt may be a tuple of tensors — take the last one
            if isinstance(plddt_out, (list, tuple)):
                plddt_out = plddt_out[-1]
            p = plddt_out.squeeze().cpu().float().numpy()
            if p.ndim > 1:
                p = p.mean(-1)
            p = np.atleast_1d(p)  # handle 0-d scalar
            if p.max() <= 1.0:
                p = p * 100.0
            if p.shape[0] == 1:
                p = np.repeat(p, L)  # broadcast scalar to per-residue
        else:
            p = np.full(L, 70.0, dtype=np.float32)

        return c1.astype(np.float32)[:L], p.astype(np.float32)[:L]

    def _stub_predict(self, sequence: str, seed: int):
        """A-form helix placeholder — used when RhoFold is unavailable."""
        rng = np.random.default_rng(seed)
        n   = len(sequence)
        t   = np.linspace(0, n * 0.6, n)
        c   = np.stack([9.0*np.cos(t), 9.0*np.sin(t), 2.8*np.arange(n)], axis=1)
        c   = (c + rng.normal(0, 0.5, c.shape)).astype(np.float32)
        p   = rng.uniform(40, 80, n).astype(np.float32)
        return c, p
