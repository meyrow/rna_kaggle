"""
structure_predictor.py — Orchestrates RNA 3D structure prediction.

Model priority chain (uses first available):
  1. RhoFold+ — fastest, ~0.14s/seq, fits P100 16GB and RTX 4060 8GB
  2. Protenix  — highest quality, ~2-5 min/seq, needs 1.5GB checkpoint
  3. Stub      — A-form helix placeholder, used when no model is available

RibonanzaNet2 is used as a feature encoder regardless of which
backbone is selected — it provides rich sequence + pairwise features
that significantly improve prediction quality.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictedStructure:
    """Output of a single structure prediction."""
    target_id: str
    sequence: str
    c1_coords: np.ndarray        # shape (L, 3)
    plddt: float                 # mean pLDDT 0-100
    plddt_per_residue: np.ndarray
    seed: int
    branch: str
    n_templates_used: int = 0

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    def is_valid(self) -> bool:
        return (
            self.c1_coords is not None
            and self.c1_coords.shape == (self.n_residues, 3)
            and not np.any(np.isnan(self.c1_coords))
        )


class StructurePredictor:
    """
    Unified structure predictor.

    Loads available models at startup and uses the best one available.
    Fails gracefully: always returns a prediction even if all models
    are missing (using the stub predictor as last resort).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipeline_cfg   = cfg.get("pipeline", {})
        self.protenix_cfg   = cfg.get("protenix", {})
        self.rn2_cfg        = cfg.get("ribonanzanet2", {})

        self.device         = self.pipeline_cfg.get("device", "cuda")
        self.chunk_len      = self.pipeline_cfg.get("chunk_length", 400)

        # Lazy-loaded models
        self._rn2      = None   # RibonanzaNet2Encoder
        self._rhofold  = None   # RhoFoldPredictor
        self._protenix = None   # Protenix (AlphaFold3)

        self._active_backend = None  # set on first predict()

        # Load immediately — safe, never crashes
        self._load_encoders()

    def _load_encoders(self):
        """Load all available models. Log what's available."""
        # ── RibonanzaNet2 (encoder) ──────────────────────────────────
        try:
            from src.ribonanzanet2_encoder import RibonanzaNet2Encoder  # inlined above
            self._rn2 = RibonanzaNet2Encoder(self.rn2_cfg)
            if self._rn2.available:
                logger.info("StructurePredictor: RibonanzaNet2 encoder READY")
            else:
                logger.info("StructurePredictor: RibonanzaNet2 not found — using one-hot")
        except Exception as e:
            logger.info(f"StructurePredictor: RibonanzaNet2 skip ({e})")
            self._rn2 = None

        # ── Protenix (AlphaFold3-based, highest quality) ─────────────
        try:
            from src.protenix_predictor import ProtenixPredictor
            self._protenix_pred = ProtenixPredictor(self.protenix_cfg)
            if self._protenix_pred.available:
                self._active_backend = "protenix"
                logger.info("StructurePredictor: Protenix backend READY")
        except Exception as e:
            logger.info(f"StructurePredictor: Protenix skip ({e})")
            self._protenix_pred = None

        # ── RhoFold+ (fallback) ───────────────────────────────────────
        try:
            from src.rhofold_predictor import RhoFoldPredictor
            self._rhofold = RhoFoldPredictor(self.protenix_cfg)
            if self._rhofold.available and self._active_backend is None:
                self._active_backend = "rhofold"
                logger.info("StructurePredictor: RhoFold+ backend READY")
        except Exception as e:
            logger.info(f"StructurePredictor: RhoFold skip ({e})")

        if self._active_backend is None:
            self._active_backend = "stub"
            logger.warning(
                "StructurePredictor: No 3D model available — using stub predictor.\n"
                "  To improve: add RhoFold or Protenix checkpoint."
            )

        logger.info(f"StructurePredictor: active backend = {self._active_backend}")

    def predict(
        self,
        sequence: str,
        target_id: str,
        seed: int,
        templates: list,
        branch: str,
    ) -> "PredictedStructure":
        """
        Predict 3D structure for one sequence with one seed.

        Pipeline:
          1. RibonanzaNet2 → sequence + pairwise features
          2. RhoFold / Protenix / stub → C1' coordinates
        """
        n = len(sequence)

        # Step 1: encode sequence with RibonanzaNet2
        single_feat, pair_feat = self._encode_sequence(sequence)

        # Step 2: predict 3D structure
        if n > self.chunk_len and self._active_backend != "stub":
            c1, plddt = self._predict_chunked(sequence, seed, single_feat)
        else:
            c1, plddt = self._predict_single(sequence, seed, single_feat, pair_feat)

        struct = PredictedStructure(
            target_id=target_id,
            sequence=sequence,
            c1_coords=c1,
            plddt=float(np.mean(plddt)),
            plddt_per_residue=plddt,
            seed=seed,
            branch=branch,
            n_templates_used=len(templates),
        )
        # Clear prediction cache after last seed to free memory
        if seed == 1337 and hasattr(self, "_pred_cache"):
            self._pred_cache.clear()
        return struct

    def _encode_sequence(self, sequence: str):
        """Encode with RibonanzaNet2 or fall back to one-hot."""
        if self._rn2 is not None and self._rn2.available:
            try:
                return self._rn2.encode(sequence)
            except Exception as e:
                logger.warning(f"  RN2 encode failed: {e}")
        # One-hot fallback
        L = len(sequence)
        nuc = {"A":0,"C":1,"G":2,"U":3,"T":3}
        oh = np.zeros((L, 256), dtype=np.float32)
        for i, c in enumerate(sequence.upper()):
            oh[i, nuc.get(c, 0)] = 1.0
        pair = np.zeros((L, L, 64), dtype=np.float32)
        return oh, pair

    def _predict_single(self, sequence, seed, single_feat, pair_feat):
        """Predict full sequence in one pass."""
        if self._active_backend == "protenix" and hasattr(self, "_protenix_pred") and self._protenix_pred:
            # Protenix is stochastic (diffusion) — different seeds give different structures
            # Use target_id from sequence hash as a stable name
            import hashlib
            tgt_id = "seq_" + hashlib.md5(sequence.encode()).hexdigest()[:8]
            return self._protenix_pred.predict(sequence, tgt_id, seed=seed)
        if self._active_backend == "rhofold" and self._rhofold:
            # RhoFold is deterministic — cache results
            cache_key = (sequence, "rhofold")
            if hasattr(self, "_pred_cache") and cache_key in self._pred_cache:
                return self._pred_cache[cache_key]
            result = self._rhofold.predict(sequence, seed, single_feat, pair_feat)
            if not hasattr(self, "_pred_cache"):
                self._pred_cache = {}
            self._pred_cache[cache_key] = result
            return result
        # Stub fallback
        return self._stub_predict(sequence, seed)

    def _predict_chunked(self, sequence, seed, single_feat):
        """
        Chunk long sequences and stitch with rigid-body alignment.
        Used when sequence length > chunk_len for memory safety.
        """
        n = len(sequence)
        overlap = min(50, self.chunk_len // 4)
        stride  = self.chunk_len - overlap
        all_chunks = []
        pos = 0
        while pos < n:
            end = min(pos + self.chunk_len, n)
            chunk_seq = sequence[pos:end]
            chunk_sf  = single_feat[pos:end] if single_feat is not None else None
            c, p = self._predict_single(chunk_seq, seed + pos, chunk_sf, None)
            all_chunks.append((pos, end, c, p))
            if end == n:
                break
            pos += stride

        # Stitch chunks with mean-offset alignment
        stitched_c = np.zeros((n, 3), dtype=np.float32)
        stitched_p = np.zeros(n, dtype=np.float32)
        for i, (start, end, c, p) in enumerate(all_chunks):
            if i == 0:
                stitched_c[start:end] = c
                stitched_p[start:end] = p
            else:
                prev_end   = all_chunks[i-1][1]
                ov_start   = start
                ov_end     = min(prev_end, end)
                n_ov       = ov_end - ov_start
                if n_ov > 0:
                    ref    = stitched_c[ov_start:ov_end]
                    curr   = c[:n_ov]
                    offset = np.mean(ref - curr, axis=0)
                    c      = c + offset
                    p_avg  = (stitched_p[ov_start:ov_end] + p[:n_ov]) / 2
                    stitched_p[ov_start:ov_end] = p_avg
                stitched_c[prev_end:end] = c[n_ov:]
                stitched_p[prev_end:end] = p[n_ov:]
        return stitched_c, stitched_p

    def _stub_predict(self, sequence: str, seed: int):
        """A-form helix placeholder — used only when no real model is available."""
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

    def get_msa_path(self, target_id: str) -> Optional[str]:
        msa_dir = self.protenix_cfg.get("msa_dir", "data/msa")
        for ext in [".a3m", ".sto", ".fasta"]:
            p = Path(msa_dir) / f"{target_id}{ext}"
            if p.exists():
                return str(p)
        return None
