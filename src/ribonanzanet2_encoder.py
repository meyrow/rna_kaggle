"""
src/ribonanzanet2_encoder.py — RibonanzaNet2 as a frozen sequence encoder.

RibonanzaNet2 (100M params) encodes RNA sequences into:
  - Single representation: (L, 256) per-residue features
  - Pair representation:   (L, L, 64) pairwise features

These features replace the stub and feed downstream structure prediction.

On Kaggle the checkpoint is available by adding to kernel-metadata.json:
  "model_sources": ["shujun717/ribonanzanet2/pyTorch/alpha/1"]
Mount path: /kaggle/input/ribonanzanet2/

On local Ubuntu the checkpoint lives at:
  models/ribonanzanet2/pytorch_model_fsdp.bin
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Nucleotide → integer encoding used by RibonanzaNet2
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 0}

# Known checkpoint locations (tried in order)
CHECKPOINT_CANDIDATES = [
    "/kaggle/input/ribonanzanet2/pytorch_model_fsdp.bin",          # Kaggle model mount
    "/kaggle/input/ribonanzanet2/pytorch/alpha/1/pytorch_model_fsdp.bin",
    "models/ribonanzanet2/pytorch_model_fsdp.bin",                 # local
    "/home/ilan/kaggle/data/models/ribonanzanet2/pytorch_model_fsdp.bin",
]
CONFIG_CANDIDATES = [
    "/kaggle/input/ribonanzanet2/pairwise.yaml",
    "/kaggle/input/ribonanzanet2/pytorch/alpha/1/pairwise.yaml",
    "models/ribonanzanet2/pairwise.yaml",
    "/home/ilan/kaggle/data/models/ribonanzanet2/pairwise.yaml",
]


def find_file(candidates: list[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class RibonanzaNet2Encoder:
    """
    Frozen RibonanzaNet2 encoder for RNA sequence → features.

    Usage:
        encoder = RibonanzaNet2Encoder()
        if encoder.available:
            single, pair = encoder.encode("ACGUACGU")
            # single: (L, 256), pair: (L, L, 64)
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self._checkpoint = cfg.get("checkpoint") or find_file(CHECKPOINT_CANDIDATES)
        self._config    = cfg.get("network_config") or find_file(CONFIG_CANDIDATES)
        self._model     = None
        self._device    = None
        self.available  = False
        self._try_load()

    def _try_load(self):
        if not self._checkpoint:
            logger.warning("RibonanzaNet2: checkpoint not found. Encoder unavailable.")
            logger.warning(f"  Tried: {CHECKPOINT_CANDIDATES[:2]}")
            return
        if not self._config:
            logger.warning("RibonanzaNet2: config yaml not found. Encoder unavailable.")
            return

        try:
            import torch
            import yaml

            # Determine device
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load network config
            with open(self._config) as f:
                net_cfg = yaml.safe_load(f)

            # Import RibonanzaNet2 network
            # The checkpoint dir contains Network.py and dropout.py
            ckpt_dir = str(Path(self._checkpoint).parent)
            import sys
            if ckpt_dir not in sys.path:
                sys.path.insert(0, ckpt_dir)

            from Network import RibonanzaNet  # type: ignore
            model = RibonanzaNet(**net_cfg)

            # Load weights — handle FSDP sharded checkpoint
            state = torch.load(self._checkpoint, map_location="cpu", weights_only=True)
            # FSDP checkpoints may have a 'model' key
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            # Strip FSDP prefixes if present
            cleaned = {}
            for k, v in state.items():
                new_k = k.replace("_fsdp_wrapped_module.", "").replace("module.", "")
                cleaned[new_k] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                logger.warning(f"RibonanzaNet2: {len(missing)} missing keys (non-fatal)")

            model = model.to(self._device).eval()
            # Freeze — we only use it as a feature extractor
            for p in model.parameters():
                p.requires_grad_(False)

            self._model = model
            self.available = True
            logger.info(f"RibonanzaNet2 loaded from {self._checkpoint} on {self._device}")

        except Exception as e:
            logger.warning(f"RibonanzaNet2 load failed: {e}")
            logger.warning("  Falling back to one-hot encoding.")
            self._model = None
            self.available = False

    def encode(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode an RNA sequence.

        Returns:
            single: (L, 256) float32 — per-residue features
            pair:   (L, L, 64) float32 — pairwise features
                    (approximated as outer product if model unavailable)
        """
        if self._model is not None and self.available:
            return self._encode_with_model(sequence)
        else:
            return self._encode_onehot(sequence)

    def _encode_with_model(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through RibonanzaNet2."""
        try:
            import torch

            L = len(sequence)
            # Tokenise: (1, L) integer tensor
            tokens = torch.tensor(
                [[NUC_TO_IDX.get(c.upper(), 0) for c in sequence]],
                dtype=torch.long, device=self._device
            )

            with torch.inference_mode():
                out = self._model(tokens)

            # Extract single and pair representations
            # RibonanzaNet2 returns a dict or tuple depending on version
            if isinstance(out, dict):
                single = out.get("single", out.get("embeddings", None))
                pair   = out.get("pair",   out.get("pairwise",   None))
            elif isinstance(out, (list, tuple)):
                single = out[0]
                pair   = out[1] if len(out) > 1 else None
            else:
                single = out
                pair   = None

            single_np = single[0].cpu().float().numpy() if single is not None \
                        else self._onehot(sequence)
            pair_np   = pair[0].cpu().float().numpy()   if pair   is not None \
                        else self._outer_product(single_np)

            return single_np, pair_np

        except Exception as e:
            logger.warning(f"RibonanzaNet2 forward pass failed: {e}. Using one-hot.")
            return self._encode_onehot(sequence)

    def _encode_onehot(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: 4-dim one-hot expanded to match expected feature dims."""
        L = len(sequence)
        onehot = np.zeros((L, 4), dtype=np.float32)
        for i, c in enumerate(sequence):
            onehot[i, NUC_TO_IDX.get(c.upper(), 0)] = 1.0
        # Pad to 256 dims with zeros
        single = np.pad(onehot, ((0, 0), (0, 256 - 4)))
        pair   = self._outer_product(single)
        return single, pair

    def _onehot(self, sequence: str) -> np.ndarray:
        L = len(sequence)
        onehot = np.zeros((L, 256), dtype=np.float32)
        for i, c in enumerate(sequence):
            onehot[i, NUC_TO_IDX.get(c.upper(), 0)] = 1.0
        return onehot

    @staticmethod
    def _outer_product(single: np.ndarray) -> np.ndarray:
        """Approximate pair features as outer product of single features."""
        L, D = single.shape
        # Use first 8 dims for memory efficiency: outer → (L, L, 64)
        s8 = single[:, :8]
        pair = np.einsum("id,jd->ijd", s8, s8)           # (L, L, 8)
        # Pad to 64 dims
        return np.pad(pair, ((0,0),(0,0),(0,56))).astype(np.float32)
