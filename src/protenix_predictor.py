"""
src/protenix_predictor.py — Protenix RNA 3D structure predictor.

Protenix (ByteDance, AlphaFold3-based) is the highest-quality RNA predictor
available. The top Kaggle notebooks (LB 0.438) all use it.

Checkpoint: https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt
Install:    pip install protenix --no-deps

On Kaggle: add as dataset source (1.37GB checkpoint)
On local:  symlink to ~/.local/lib/.../site-packages/release_data/checkpoint/
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROTENIX_CKPT_CANDIDATES = [
    # Kaggle dataset mount (to be added to kernel-metadata.json)
    "/kaggle/input/protenix-model/protenix_base_default_v0.5.0.pt",
    # Local Ubuntu path
    "/home/ilan/kaggle/data/models/protenix/protenix_base_default_v0.5.0.pt",
    # Protenix package default dir
    os.path.expanduser("~/.local/lib/python3.10/site-packages/release_data/checkpoint/protenix_base_default_v0.5.0.pt"),
]


def find_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class ProtenixPredictor:
    """
    Protenix RNA 3D structure predictor via CLI subprocess.

    Calls 'protenix predict' for each sequence, parses the output CIF
    to extract C1' atom coordinates.

    This approach avoids complex Python API integration and matches
    exactly what the top Kaggle notebooks do.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self._ckpt = cfg.get("checkpoint") or find_path(PROTENIX_CKPT_CANDIDATES)
        self._out_dir = cfg.get("protenix_out_dir", "/tmp/protenix_out")
        self.available = False
        self._setup_checkpoint()

    def _setup_checkpoint(self):
        """Symlink checkpoint to where Protenix expects it."""
        if not self._ckpt:
            logger.warning("Protenix: checkpoint not found")
            logger.warning(f"  Tried: {PROTENIX_CKPT_CANDIDATES[:2]}")
            return

        # Check protenix is installed
        try:
            result = subprocess.run(
                ["protenix", "--help"], capture_output=True, timeout=10
            )
            if result.returncode not in (0, 1, 2):
                logger.warning("Protenix CLI not available")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Protenix not installed (pip install protenix --no-deps)")
            return

        # Find the package checkpoint dir
        try:
            import configs.configs_inference as ci
            pkg_dir = os.path.dirname(os.path.dirname(ci.__file__))
            ckpt_dir = os.path.join(pkg_dir, "release_data", "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_target = os.path.join(ckpt_dir, "protenix_base_default_v0.5.0.pt")
            if not os.path.exists(ckpt_target):
                os.symlink(self._ckpt, ckpt_target)
                logger.info(f"Protenix: symlinked checkpoint to {ckpt_dir}")
        except Exception as e:
            logger.warning(f"Protenix: checkpoint setup failed: {e}")
            return

        self.available = True
        logger.info("Protenix ready ✓")

    def predict(
        self,
        sequence: str,
        target_id: str,
        seed: int = 101,
        n_samples: int = 1,
    ) -> tuple:
        """
        Predict RNA 3D structure using Protenix.

        Returns:
            c1_coords: (L, 3) float32 — C1' atom coordinates
            plddt:     (L,)   float32 — per-residue confidence 0-100
        """
        if not self.available:
            return self._stub(sequence, seed)

        # Skip very long sequences — Protenix needs O(L^2) memory
        max_len = 500  # RTX 4060: 8GB. P100: can handle ~1000nt
        if len(sequence) > max_len:
            logger.warning(f"Protenix: skipping {target_id} len={len(sequence)} > {max_len} (use RhoFold/stub)")
            return self._stub(sequence, seed)

        try:
            return self._protenix_predict(sequence, target_id, seed, n_samples)
        except Exception as e:
            logger.warning(f"Protenix failed for {target_id} (len={len(sequence)}): {e}")
            return self._stub(sequence, seed)

    def _protenix_predict(self, sequence, target_id, seed, n_samples):
        """Run protenix predict CLI and parse output CIF."""
        L = len(sequence)
        out_dir = os.path.join(self._out_dir, target_id, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)

        # Write input JSON
        input_json = os.path.join(out_dir, "input.json")
        input_data = [{
            "name": target_id,
            "sequences": [{
                "rnaSequence": {"sequence": sequence, "count": 1}
            }]
        }]
        with open(input_json, "w") as f:
            json.dump(input_data, f)

        # Run Protenix
        cmd = [
            "protenix", "predict",
            "--input",    input_json,
            "--out_dir",  out_dir,
            "--seeds",    str(seed),
            "--model_name", "protenix_base_default_v0.5.0",
            "--use_msa",  "false",
            "--use_default_params", "false",
            "--sample",   str(n_samples),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.warning(f"Protenix exit {result.returncode}: {result.stderr[-300:]}")
            return self._stub(sequence, seed)

        # Find output CIF
        cif_pattern = f"{out_dir}/{target_id}/seed_{seed}/predictions/*.cif"
        import glob
        cif_files = sorted(glob.glob(cif_pattern))
        if not cif_files:
            # Try alternate path
            cif_files = sorted(glob.glob(f"{out_dir}/**/*.cif", recursive=True))
        if not cif_files:
            logger.warning(f"Protenix: no CIF output found in {out_dir}")
            return self._stub(sequence, seed)

        return self._parse_cif(cif_files[0], L)

    def _parse_cif(self, cif_path: str, L: int) -> tuple:
        """
        Extract C1' coordinates and pLDDT from Protenix output CIF.

        CIF ATOM line format (space-separated):
        ATOM type atom_name alt res chain entity res_seq ins auth_seq
             auth_res auth_chain auth_atom plddt x y z ...

        C1' example:
        ATOM C "C1'" . G A 1 1 . 1 G A "C1'" 74.5 29.06 -0.81 11.02 1 13 1.0
        """
        c1_coords = []
        plddt_vals = []

        with open(cif_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                parts = line.split()
                # Find C1' atom — atom_name is at index 2
                # Handle quoted atom names: "C1'" or C1'
                atom_name = parts[2].strip('"')
                if atom_name != "C1'":
                    continue
                try:
                    # plddt=col13, x=col14, y=col15, z=col16
                    plddt = float(parts[13])
                    x = float(parts[14])
                    y = float(parts[15])
                    z = float(parts[16])
                    c1_coords.append([x, y, z])
                    plddt_vals.append(plddt)
                except (IndexError, ValueError) as e:
                    logger.debug(f"CIF parse error: {e} — line: {line.strip()}")

        if not c1_coords:
            logger.warning(f"No C1' atoms found in {cif_path}")
            return self._stub_coords(L), np.full(L, 50.0, dtype=np.float32)

        coords = np.array(c1_coords, dtype=np.float32)
        plddt  = np.array(plddt_vals, dtype=np.float32)

        # Trim or pad to exact sequence length
        if len(coords) > L:
            coords = coords[:L]
            plddt  = plddt[:L]
        elif len(coords) < L:
            pad = L - len(coords)
            coords = np.pad(coords, ((0, pad), (0, 0)))
            plddt  = np.pad(plddt,  (0, pad), constant_values=50.0)

        return coords, plddt

    def _stub(self, sequence: str, seed: int) -> tuple:
        """A-form helix fallback."""
        return self._stub_coords(len(sequence)), np.full(len(sequence), 50.0, dtype=np.float32)

    def _stub_coords(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        t = np.linspace(0, n * 0.6, n)
        c = np.stack([9.0*np.cos(t), 9.0*np.sin(t), 2.8*np.arange(n)], axis=1)
        return (c + rng.normal(0, 0.5, c.shape)).astype(np.float32)
