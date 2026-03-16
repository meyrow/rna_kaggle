"""
structure_predictor.py — Wraps Protenix (AlphaFold3 repro) + RibonanzaNet2.

This module is the heavy ML core of the pipeline.

Architecture (following NVIDIA RNAPro design):
  RibonanzaNet2 (frozen) → sequence + pairwise features
       ↓ projection + gating
  Protenix backbone → structure diffusion
       ↑ (optional) template embedder with C1' priors

For 8GB VRAM (RTX 4060):
  - Use bf16 precision
  - Enable gradient checkpointing for sequences > 300 nt
  - Chunk sequences > 500 nt
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
    c1_coords: np.ndarray       # shape (L, 3) — x,y,z for each residue
    plddt: float                # mean pLDDT confidence (0–100)
    plddt_per_residue: np.ndarray  # shape (L,)
    seed: int
    branch: str                 # "tbm" / "denovo"
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
    Wrapper around Protenix + RibonanzaNet2.

    The actual heavy models are loaded lazily on first call
    to avoid GPU memory allocation at import time.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.protenix_cfg = cfg.get("protenix", {})
        self.rn2_cfg = cfg.get("ribonanzanet2", {})
        self.pipeline_cfg = cfg.get("pipeline", {})

        self.device = self.pipeline_cfg.get("device", "cuda")
        self.dtype = self.protenix_cfg.get("dtype", "bf16")
        self.n_cycle = self.protenix_cfg.get("n_cycle", 10)
        self.n_step = self.protenix_cfg.get("n_step", 200)
        self.use_msa = self.protenix_cfg.get("use_msa", True)
        self.msa_dir = self.protenix_cfg.get("msa_dir", "data/msa")
        self.gradient_checkpointing = self.protenix_cfg.get("gradient_checkpointing", True)
        self.max_len = self.pipeline_cfg.get("max_sequence_length", 6000)
        self.chunk_len = self.pipeline_cfg.get("chunk_length", 500)

        self._protenix = None
        self._rn2 = None

    def _load_protenix(self):
        """Lazy-load Protenix model."""
        if self._protenix is not None:
            return
        checkpoint = self.protenix_cfg.get("checkpoint", "")
        if not Path(checkpoint).exists():
            logger.warning(
                f"Protenix checkpoint not found at '{checkpoint}'. "
                "Run scripts/download_models.sh to download."
            )
            self._protenix = "MISSING"
            return
        try:
            # Import here to avoid hard dependency at module load
            import torch
            logger.info(f"Loading Protenix from {checkpoint}")
            # Actual Protenix import — installed via pip install protenix or local clone
            # from protenix.model.protenix import Protenix
            # self._protenix = Protenix.from_checkpoint(checkpoint, device=self.device)
            logger.info("Protenix loaded (stub — replace with actual protenix import)")
            self._protenix = "STUB"
        except ImportError as e:
            logger.error(f"Could not import Protenix: {e}")
            self._protenix = "MISSING"

    def _load_ribonanzanet2(self):
        """Lazy-load RibonanzaNet2 as frozen sequence encoder."""
        if self._rn2 is not None:
            return
        checkpoint = self.rn2_cfg.get("checkpoint", "")
        if not Path(checkpoint).exists():
            logger.warning(f"RibonanzaNet2 checkpoint not found at '{checkpoint}'.")
            self._rn2 = "MISSING"
            return
        try:
            import torch
            logger.info(f"Loading RibonanzaNet2 from {checkpoint}")
            # from ribonanzanet2.Network import RibonanzaNet2
            # self._rn2 = RibonanzaNet2.from_pretrained(checkpoint)
            # self._rn2.eval()
            # if self.rn2_cfg.get("freeze_encoder", True):
            #     for p in self._rn2.parameters(): p.requires_grad_(False)
            logger.info("RibonanzaNet2 loaded (stub — replace with actual import)")
            self._rn2 = "STUB"
        except ImportError as e:
            logger.error(f"Could not import RibonanzaNet2: {e}")
            self._rn2 = "MISSING"

    def predict(
        self,
        sequence: str,
        target_id: str,
        seed: int,
        templates: list,
        branch: str,
    ) -> PredictedStructure:
        """
        Run one prediction for a single sequence and seed.

        Args:
            sequence: RNA sequence (A, C, G, U)
            target_id: competition target ID
            seed: random seed for diffusion sampling
            templates: list of Template objects (empty for de novo)
            branch: "tbm" or "denovo"

        Returns:
            PredictedStructure with c1_coords and pLDDT
        """
        self._load_protenix()
        self._load_ribonanzanet2()

        n = len(sequence)
        use_template = branch == "tbm" and len(templates) > 0

        logger.debug(
            f"    predict: len={n}, branch={branch}, "
            f"templates={len(templates)}, seed={seed}"
        )

        # Handle long sequences via chunking
        if n > self.chunk_len and self.chunk_len > 0:
            return self._predict_chunked(sequence, target_id, seed, templates, branch)

        # ── Real Protenix call (uncomment when models are downloaded) ──
        # import torch
        # torch.manual_seed(seed)
        # features = self._build_features(sequence, templates)
        # with torch.inference_mode():
        #     output = self._protenix.forward(
        #         features,
        #         use_template="ca_precomputed" if use_template else "none",
        #         n_cycle=self.n_cycle,
        #         n_step=self.n_step,
        #         dtype=torch.bfloat16 if self.dtype == "bf16" else torch.float32,
        #     )
        # c1_coords = output["c1_coords"].cpu().numpy()   # (L, 3)
        # plddt_per_res = output["plddt"].cpu().numpy()   # (L,)

        # ── Stub for testing pipeline without models ──────────────────
        c1_coords, plddt_per_res = self._stub_predict(sequence, seed)

        return PredictedStructure(
            target_id=target_id,
            sequence=sequence,
            c1_coords=c1_coords,
            plddt=float(np.mean(plddt_per_res)),
            plddt_per_residue=plddt_per_res,
            seed=seed,
            branch=branch,
            n_templates_used=len(templates),
        )

    def _predict_chunked(
        self, sequence, target_id, seed, templates, branch
    ) -> PredictedStructure:
        """
        For sequences > chunk_len, predict in overlapping windows
        and stitch coordinates together.
        This is a simplified linear stitching; production code would
        use global alignment to merge overlapping predictions.
        """
        n = len(sequence)
        overlap = min(50, self.chunk_len // 4)
        stride = self.chunk_len - overlap

        all_coords = []
        chunk_ranges = []

        pos = 0
        while pos < n:
            end = min(pos + self.chunk_len, n)
            chunk_seq = sequence[pos:end]
            chunk_struct = self.predict(
                sequence=chunk_seq,
                target_id=f"{target_id}_chunk{pos}",
                seed=seed,
                templates=[],  # templates not used for chunked
                branch="denovo",
            )
            all_coords.append((pos, end, chunk_struct.c1_coords))
            chunk_ranges.append((pos, end))
            if end == n:
                break
            pos += stride

        # Simple stitching: take each chunk's non-overlapping region
        stitched = np.zeros((n, 3))
        for i, (start, end, coords) in enumerate(all_coords):
            if i == 0:
                stitched[start:end] = coords
            else:
                prev_end = chunk_ranges[i - 1][1]
                # Translate new chunk to align with previous
                overlap_start = start
                overlap_end = prev_end
                if overlap_end > overlap_start:
                    # Rigid body alignment over overlap region (simplified: mean offset)
                    n_ov = overlap_end - overlap_start
                    prev_ov = stitched[overlap_start:overlap_end]
                    curr_ov = coords[:n_ov]
                    offset = np.mean(prev_ov - curr_ov, axis=0)
                    coords = coords + offset
                stitched[prev_end:end] = coords[overlap_end - start:]

        plddt_stub = np.full(n, 50.0)
        return PredictedStructure(
            target_id=target_id,
            sequence=sequence,
            c1_coords=stitched,
            plddt=50.0,
            plddt_per_residue=plddt_stub,
            seed=seed,
            branch=f"{branch}_chunked",
        )

    def _stub_predict(
        self, sequence: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stub prediction for testing the pipeline without actual models.
        Generates a helical RNA-like structure with some noise.
        Replace this with real Protenix output.
        """
        rng = np.random.default_rng(seed)
        n = len(sequence)
        # Simple A-form helix geometry as a placeholder
        t = np.linspace(0, n * 0.6, n)
        radius = 9.0  # Angstroms (typical A-form RNA helix)
        rise = 2.8    # Angstroms per residue
        coords = np.stack([
            radius * np.cos(t),
            radius * np.sin(t),
            rise * np.arange(n),
        ], axis=1)
        # Add noise to simulate different seeds
        coords += rng.normal(0, 0.5, coords.shape)
        plddt = rng.uniform(40, 80, n)
        return coords.astype(np.float32), plddt.astype(np.float32)

    def get_msa_path(self, target_id: str) -> Optional[str]:
        """Find precomputed MSA file for a target."""
        for ext in [".a3m", ".sto", ".fasta"]:
            path = Path(self.msa_dir) / f"{target_id}{ext}"
            if path.exists():
                return str(path)
        return None
