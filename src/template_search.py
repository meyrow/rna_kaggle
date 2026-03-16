"""
template_search.py — Template-Based Modeling (TBM) search module.

Searches a local PDB RNA C1' coordinate database using MMseqs2.
Returns ranked structural templates with expected TM-score estimates.

Based on the approach from:
  - Notebook B (gourabr0y555): Protenix + TBM
  - jaejohn (Part 1 1st place): TBM-only approach
  - NVIDIA RNAPro: MMseqs2 3D RNA Template Identification
"""

import subprocess
import pickle
import logging
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.family_classifier import FamilyResult

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """A single structural template retrieved from PDB."""
    pdb_id: str
    chain_id: str
    sequence: str
    seq_identity: float       # fraction (0–1)
    coverage: float           # query coverage (0–1)
    expected_tm: float        # estimated TM-score for this template
    c1_coords: Optional[np.ndarray] = None  # shape (L, 3)
    alignment: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.pdb_id}_{self.chain_id}"

    def to_dict(self) -> dict:
        return {
            "pdb_id": self.pdb_id,
            "chain_id": self.chain_id,
            "sequence": self.sequence,
            "seq_identity": self.seq_identity,
            "coverage": self.coverage,
            "expected_tm": self.expected_tm,
            "c1_coords": self.c1_coords,
        }


class TemplateSearcher:
    """
    Two-stage template search:
      1. MMseqs2 for fast sequence similarity search against PDB RNA sequences
      2. Load C1' coordinates for top hits from local cache
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", True)
        self.mmseqs2_db = cfg.get("mmseqs2_db", "data/pdb_cache/pdb_rna_db")
        self.pdb_c1_cache = cfg.get("pdb_c1_cache", "data/pdb_cache/pdb_c1_coords.pkl")
        self.max_templates = cfg.get("max_templates", 10)
        self.min_seq_identity = cfg.get("min_seq_identity", 0.25)
        self.min_coverage = cfg.get("min_coverage", 0.5)

        self._mmseqs_cmd = "mmseqs"
        self._mmseqs2_available = self._check_mmseqs2()
        self._c1_cache = self._load_c1_cache()

    def _check_mmseqs2(self) -> bool:
        try:
            for cmd in ["mmseqs", "mmseqs-avx2", "mmseqs-sse4.1"]:
                try:
                    r = subprocess.run([cmd, "version"], capture_output=True, timeout=5)
                    if r.returncode == 0:
                        self._mmseqs_cmd = cmd
                        return True
                except FileNotFoundError:
                    continue
            r = type('r', (), {'returncode': 1})()
            return r.returncode == 0
        except FileNotFoundError:
            logger.warning("MMseqs2 not found. Template search disabled.")
            return False

    def _load_c1_cache(self) -> dict:
        """Load prebuilt C1' coordinate cache from disk."""
        cache_path = Path(self.pdb_c1_cache)
        if cache_path.exists():
            logger.info(f"Loading C1' cache from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        else:
            logger.warning(f"C1' cache not found at {cache_path}. Run build_pdb_cache.sh first.")
            return {}

    def search(self, sequence: str, family: FamilyResult) -> list[Template]:
        """
        Search for structural templates for the given sequence.
        Returns list of Template objects sorted by expected TM-score descending.
        """
        if not self.enabled:
            return []
        if not self._mmseqs2_available:
            return []
        if not Path(self.mmseqs2_db).exists():
            logger.warning(f"MMseqs2 DB not found at {self.mmseqs2_db}")
            return []

        raw_hits = self._run_mmseqs2(sequence)
        templates = []

        for hit in raw_hits[:self.max_templates * 3]:  # search wider, filter after
            if hit["seq_identity"] < self.min_seq_identity:
                continue
            if hit["coverage"] < self.min_coverage:
                continue

            # Estimate TM-score from sequence identity (empirical formula)
            expected_tm = self._estimate_tm_from_seqid(
                hit["seq_identity"], hit["coverage"], len(sequence)
            )

            # Load C1' coordinates from cache
            key = f"{hit['pdb_id']}_{hit['chain_id']}"
            c1_coords = self._c1_cache.get(key)

            templates.append(Template(
                pdb_id=hit["pdb_id"],
                chain_id=hit["chain_id"],
                sequence=hit["target_seq"],
                seq_identity=hit["seq_identity"],
                coverage=hit["coverage"],
                expected_tm=expected_tm,
                c1_coords=c1_coords,
                alignment=hit.get("alignment", {}),
            ))

        # Sort by expected TM-score
        templates.sort(key=lambda t: t.expected_tm, reverse=True)
        return templates[:self.max_templates]

    def _run_mmseqs2(self, sequence: str) -> list[dict]:
        """Run MMseqs2 easy-search and parse hits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            query_fa = os.path.join(tmpdir, "query.fa")
            result_tsv = os.path.join(tmpdir, "result.tsv")
            tmp_mmseqs = os.path.join(tmpdir, "tmp")

            with open(query_fa, "w") as f:
                f.write(f">query\n{sequence}\n")

            cmd = [
                self._mmseqs_cmd, "easy-search",
                query_fa, self.mmseqs2_db, result_tsv, tmp_mmseqs,
                "--format-output",
                "query,target,pident,alnlen,qstart,qend,tstart,tend,evalue,bits,qaln,taln",
                "--min-seq-id", str(self.min_seq_identity),
                "-c", str(self.min_coverage),
                "--cov-mode", "0",
                "-e", "0.001",
                "--threads", "4",
                "-v", "0",
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
            except subprocess.TimeoutExpired:
                logger.warning("MMseqs2 search timed out")
                return []

            if result.returncode != 0:
                logger.error(f"MMseqs2 failed: {result.stderr[:300]}")
                return []

            return self._parse_mmseqs2_output(result_tsv, len(sequence))

    def _parse_mmseqs2_output(self, tsv_path: str, query_len: int) -> list[dict]:
        """Parse MMseqs2 tabular output."""
        hits = []
        if not os.path.exists(tsv_path):
            return hits
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 10:
                    continue
                try:
                    target = parts[1]  # e.g. "4XWF_A"
                    pident = float(parts[2]) / 100.0
                    aln_len = int(parts[3])
                    coverage = aln_len / query_len if query_len > 0 else 0

                    # Parse PDB ID and chain
                    if "_" in target:
                        pdb_id, chain_id = target.rsplit("_", 1)
                    else:
                        pdb_id, chain_id = target, "A"

                    hits.append({
                        "pdb_id": pdb_id.upper(),
                        "chain_id": chain_id.upper(),
                        "seq_identity": pident,
                        "coverage": coverage,
                        "target_seq": parts[11] if len(parts) > 11 else "",
                        "evalue": float(parts[8]),
                    })
                except (ValueError, IndexError):
                    continue
        return hits

    @staticmethod
    def _estimate_tm_from_seqid(
        seq_id: float, coverage: float, query_len: int
    ) -> float:
        """
        Empirical formula to estimate TM-score from sequence identity.
        Calibrated from Part 1 competition results.
        Better seq_id + coverage → higher expected TM-score.
        """
        # Base estimate from identity (log-linear fit to CASP data)
        if seq_id >= 0.90:
            base = 0.90
        elif seq_id >= 0.70:
            base = 0.75 + (seq_id - 0.70) / 0.20 * 0.15
        elif seq_id >= 0.50:
            base = 0.60 + (seq_id - 0.50) / 0.20 * 0.15
        elif seq_id >= 0.30:
            base = 0.45 + (seq_id - 0.30) / 0.20 * 0.15
        else:
            base = seq_id * 1.5

        # Penalize low coverage
        tm_est = base * (0.5 + 0.5 * coverage)

        # Penalize very long sequences (harder to fold correctly)
        if query_len > 200:
            tm_est *= 0.95
        if query_len > 500:
            tm_est *= 0.90

        return min(tm_est, 0.99)
