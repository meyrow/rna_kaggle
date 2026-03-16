"""
submission.py — Build the competition submission.csv.

Exact format (confirmed from sample_submission.csv):
  ID, resname, resid, x_1,y_1,z_1, x_2,y_2,z_2, x_3,y_3,z_3, x_4,y_4,z_4, x_5,y_5,z_5

  - ID      = {target_id}_{resid}  (1-indexed residue number)
  - resname = single-letter RNA nucleotide (A/C/G/U)
  - resid   = 1-indexed residue position
  - x_i..z_i = C1' coordinates for prediction i (5 predictions total)

NOTE on validation_labels.csv:
  - Has up to 40 reference structures (x_1..z_40) — experimental conformations
  - Missing slots use sentinel -1e+18
  - Scoring: best-of-5 predictions vs best available reference → mean TM-score
  - The submission only needs the 5-slot format shown above

NOTE on multi-chain targets:
  - For U:8 octamers (e.g. 9MME, 4640 nt), the sequence column contains ALL copies
    concatenated — 8 × 580 nt = 4640 rows in the submission
  - Use the full sequence as-is; residue numbering is continuous
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESNAME_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "U",
    "a": "A", "c": "C", "g": "G", "u": "U", "N": "N",
}

# Submission columns in exact competition order
COORD_COLS = [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]]
SUBMISSION_COLS = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]]
# Re-order to match sample_submission: x_1,y_1,z_1, x_2,y_2,z_2, ...
SUBMISSION_COLS = ["ID", "resname", "resid"] + [f"{ax}_{i}" for i in range(1, 6) for ax in ["x", "y", "z"]]


class SubmissionBuilder:
    """
    Builds the submission.csv in the exact format required by the competition.
    Validated against sample_submission.csv format.
    """

    def build(self, predictions: list[dict], output_path: str):
        rows = []
        n_targets = 0
        n_failed = 0

        for pred in predictions:
            target_id = pred["target_id"]
            sequence  = pred["sequence"]
            structures = pred.get("structures", [])

            if not structures:
                logger.warning(f"No structures for {target_id}, skipping")
                n_failed += 1
                continue

            # Pad to exactly 5 structures
            while len(structures) < 5:
                last = structures[-1]
                rng = np.random.default_rng(len(structures) + 999)
                noisy = last.c1_coords + rng.normal(0, 0.01, last.c1_coords.shape)
                from src.structure_predictor import PredictedStructure
                structures.append(PredictedStructure(
                    target_id=last.target_id, sequence=last.sequence,
                    c1_coords=noisy.astype(np.float32), plddt=last.plddt,
                    plddt_per_residue=last.plddt_per_residue,
                    seed=last.seed + 1000, branch=last.branch,
                ))
            structures = structures[:5]

            n_res = len(sequence)
            for j in range(n_res):
                resname = RESNAME_MAP.get(sequence[j].upper(), "N")
                resid   = j + 1   # 1-indexed, as in sample_submission
                row = {"ID": f"{target_id}_{resid}", "resname": resname, "resid": resid}
                for k, struct in enumerate(structures):
                    if struct.c1_coords.shape[0] != n_res:
                        # Coordinate length mismatch — use zeros
                        row[f"x_{k+1}"] = 0.0
                        row[f"y_{k+1}"] = 0.0
                        row[f"z_{k+1}"] = 0.0
                    else:
                        xyz = struct.c1_coords[j]
                        row[f"x_{k+1}"] = round(float(xyz[0]), 3)
                        row[f"y_{k+1}"] = round(float(xyz[1]), 3)
                        row[f"z_{k+1}"] = round(float(xyz[2]), 3)
                rows.append(row)
            n_targets += 1

        if not rows:
            logger.error("No rows to write — submission empty!")
            return

        df = pd.DataFrame(rows)[SUBMISSION_COLS]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Submission saved: {output_path}")
        logger.info(f"  {n_targets} targets | {len(df):,} rows | {n_failed} failed")

    def validate(self, output_path: str) -> bool:
        """Validate submission against competition format."""
        try:
            df = pd.read_csv(output_path)
            missing = [c for c in SUBMISSION_COLS if c not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return False
            extra = [c for c in df.columns if c not in SUBMISSION_COLS]
            if extra:
                logger.warning(f"Extra columns (harmless but unexpected): {extra}")
            coord_cols = [c for c in SUBMISSION_COLS if c not in ("ID", "resname", "resid")]
            n_nan = df[coord_cols].isna().sum().sum()
            if n_nan > 0:
                logger.warning(f"NaN values: {n_nan}")
            n_zero = (df[coord_cols] == 0).all(axis=1).sum()
            if n_zero > 0:
                logger.warning(f"All-zero rows: {n_zero} (baseline score, not a real prediction)")
            logger.info(f"Submission valid: {len(df):,} rows, {df['ID'].nunique():,} unique residues")
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def compare_with_sample(self, output_path: str, sample_path: str) -> dict:
        """
        Check that our submission has the same targets/residues as sample_submission.csv.
        Returns dict with match stats.
        """
        our   = pd.read_csv(output_path)
        sample = pd.read_csv(sample_path)
        our_ids    = set(our["ID"])
        sample_ids = set(sample["ID"])
        missing_from_ours   = sample_ids - our_ids
        extra_in_ours       = our_ids - sample_ids
        result = {
            "match": our_ids == sample_ids,
            "n_our": len(our_ids),
            "n_sample": len(sample_ids),
            "missing": len(missing_from_ours),
            "extra": len(extra_in_ours),
        }
        if missing_from_ours:
            logger.warning(f"Missing {len(missing_from_ours)} residue IDs vs sample: {list(missing_from_ours)[:5]}")
        if extra_in_ours:
            logger.warning(f"Extra {len(extra_in_ours)} residue IDs vs sample: {list(extra_in_ours)[:5]}")
        return result

