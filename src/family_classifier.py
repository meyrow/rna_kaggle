"""
family_classifier.py — Classify RNA sequences into structural families.

Uses Infernal (cmscan) against the Rfam covariance model database.
Falls back to sequence-pattern heuristics if Infernal is unavailable.

Families relevant to this competition:
  - riboswitch     (aptamer domain + expression platform)
  - tRNA           (cloverleaf, universal template)
  - ribosomal      (rRNA fragments)
  - viral          (IRES, frameshifting elements, etc.)
  - aptamer        (synthetic / in-vitro selected)
  - ribozyme       (catalytic RNA)
  - unknown        → force de novo branch
"""

import subprocess
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.secondary_structure import SecondaryStructure

logger = logging.getLogger(__name__)

# Rfam family ID → human-readable category mapping
RFAM_CATEGORY = {
    # Riboswitches
    "RF00050": "riboswitch",  # FMN
    "RF00059": "riboswitch",  # TPP
    "RF00162": "riboswitch",  # SAM-I
    "RF00174": "riboswitch",  # Cobalamin
    "RF00234": "riboswitch",  # glmS
    "RF01057": "riboswitch",  # SAH
    "RF01786": "riboswitch",  # ZMP/ZTP
    # tRNA / tRNA-like
    "RF00005": "tRNA",
    "RF00023": "tRNA",        # tmRNA
    # Ribosomal RNA
    "RF00177": "ribosomal",   # SSU rRNA
    "RF02540": "ribosomal",   # LSU rRNA
    "RF01960": "ribosomal",   # SSU rRNA archaea
    # Ribozymes
    "RF00008": "ribozyme",    # Hammerhead type III
    "RF00163": "ribozyme",    # Hammerhead type I
    "RF00622": "ribozyme",    # HDV
    # Viral
    "RF00164": "viral",       # Corona 3' UTR
    "RF00165": "viral",       # Corona 5' UTR
    # Large ncRNA
    "RF02348": "large_ncrna", # OLE RNA
    "RF02357": "large_ncrna", # GOLLD RNA
    "RF02544": "large_ncrna", # ROOL RNA
}


@dataclass
class FamilyResult:
    """Result of family classification."""
    name: str               # e.g. "riboswitch", "tRNA", "unknown"
    rfam_id: Optional[str]  # e.g. "RF00059"
    score: float            # bit score from cmscan, or heuristic confidence
    evalue: float
    is_known: bool          # True if a Rfam hit was found

    def __str__(self):
        return f"{self.name}({self.rfam_id or 'heuristic'} e={self.evalue:.1e})"


class FamilyClassifier:
    """
    RNA family classifier.
    Priority order:
      1. Rfam cmscan (if Infernal installed + Rfam.cm available)
      2. Sequence heuristics (motif patterns)
      3. Unknown
    """

    def __init__(self, cfg: dict):
        self.rfam_db = cfg.get("rfam_db", "data/rfam/Rfam.cm")
        self.evalue_threshold = cfg.get("evalue_threshold", 1e-5)
        self.known_families = cfg.get("known_families", [])
        self._infernal_available = self._check_infernal()
        self._rfam_available = Path(self.rfam_db).exists()
        if not self._rfam_available:
            logger.warning(f"Rfam CM not found at {self.rfam_db}. Using heuristics.")

    def _check_infernal(self) -> bool:
        try:
            result = subprocess.run(
                ["cmscan", "-h"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("cmscan not found. Using sequence heuristics for family classification.")
            return False

    def classify(self, sequence: str, sec_struct: SecondaryStructure) -> FamilyResult:
        """Classify an RNA sequence into a structural family."""
        if self._infernal_available and self._rfam_available:
            result = self._classify_cmscan(sequence)
            if result is not None:
                return result

        # Fallback to sequence/structure heuristics
        return self._classify_heuristic(sequence, sec_struct)

    def _classify_cmscan(self, sequence: str) -> Optional[FamilyResult]:
        """Run cmscan against Rfam and parse the best hit."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(f">query\n{sequence}\n")
            fasta_path = f.name
        try:
            result = subprocess.run(
                [
                    "cmscan", "--tblout", "/dev/stdout",
                    "-E", str(self.evalue_threshold),
                    "--noali", "--cpu", "2",
                    self.rfam_db, fasta_path,
                ],
                capture_output=True, text=True, timeout=120
            )
        except subprocess.TimeoutExpired:
            logger.warning("cmscan timed out")
            return None
        finally:
            os.unlink(fasta_path)

        if result.returncode != 0:
            logger.warning(f"cmscan error: {result.stderr[:200]}")
            return None

        # Parse tblout format
        best_hit = None
        for line in result.stdout.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 16:
                continue
            # tblout columns: target_name, accession, query, clan, mdl, mdl_from, mdl_to,
            #                  seq_from, seq_to, strand, trunc, pass, gc, bias, score, E-value
            try:
                rfam_acc = parts[1]
                score = float(parts[14])
                evalue = float(parts[15])
                if best_hit is None or evalue < best_hit[2]:
                    best_hit = (rfam_acc, score, evalue)
            except (ValueError, IndexError):
                continue

        if best_hit is None:
            return None

        rfam_id, score, evalue = best_hit
        category = RFAM_CATEGORY.get(rfam_id, "other_ncrna")
        return FamilyResult(
            name=category,
            rfam_id=rfam_id,
            score=score,
            evalue=evalue,
            is_known=True,
        )

    def _classify_heuristic(
        self, sequence: str, sec_struct: SecondaryStructure
    ) -> FamilyResult:
        """
        Fast heuristic classification based on:
        - Sequence length
        - Nucleotide composition
        - Hairpin count and sizes
        - Known motif patterns
        """
        n = len(sequence)

        # tRNA heuristic: ~73-93 nt, cloverleaf structure (4 stems)
        if 70 <= n <= 100 and len(sec_struct.stems) >= 3:
            return FamilyResult("tRNA", None, 0.6, 0.1, False)

        # Short ribozyme heuristic: <60 nt, highly structured
        if n < 70 and sec_struct.pair_fraction > 0.5:
            return FamilyResult("ribozyme", None, 0.4, 0.5, False)

        # Riboswitch-like: 80-300 nt, multiple stems
        if 80 <= n <= 350 and len(sec_struct.stems) >= 2:
            return FamilyResult("riboswitch", None, 0.3, 1.0, False)

        # Large ncRNA (Part 2 specific): >300 nt
        if n > 300:
            return FamilyResult("large_ncrna", None, 0.2, 5.0, False)

        return FamilyResult("unknown", None, 0.0, 999.0, False)
