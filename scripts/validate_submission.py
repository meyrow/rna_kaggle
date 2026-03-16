"""
scripts/validate_submission.py — Score your submission against validation_labels.csv.

Computes approximate TM-score using numpy Kabsch alignment
(not identical to US-align, but fast and strongly correlated).

Usage:
    python scripts/validate_submission.py \
        --submission outputs/submission.csv \
        --labels     /home/ilan/kaggle/data/validation_labels.csv

Output:
    Per-target scores + overall mean TM-score
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.tm_score import _tm_approx


SENTINEL = -1e18


def load_coords(df: pd.DataFrame, target: str, n_slots: int) -> list[np.ndarray]:
    """Extract non-sentinel coordinate arrays for a target."""
    sub = df[df["target"] == target].sort_values("resid")
    coords_list = []
    for i in range(1, n_slots + 1):
        x_col, y_col, z_col = f"x_{i}", f"y_{i}", f"z_{i}"
        if x_col not in sub.columns:
            break
        if (sub[x_col] == SENTINEL).all():
            continue
        # Only take rows where this slot has real coords
        mask = sub[x_col] != SENTINEL
        if not mask.any():
            continue
        coords = sub.loc[mask, [x_col, y_col, z_col]].values.astype(np.float32)
        coords_list.append(coords)
    return coords_list


def score_target(
    pred_coords_list: list[np.ndarray],
    ref_coords_list:  list[np.ndarray],
) -> float:
    """
    Compute best-of-5 vs best-reference TM-score for one target.
    Returns max over all (pred, ref) pairs.
    """
    if not pred_coords_list or not ref_coords_list:
        return 0.0

    best = 0.0
    for pred in pred_coords_list:
        for ref in ref_coords_list:
            if pred.shape[0] != ref.shape[0]:
                # Trim to shorter (handles partial structures)
                n = min(pred.shape[0], ref.shape[0])
                p, r = pred[:n], ref[:n]
            else:
                p, r = pred, ref
            tm = _tm_approx(p, r)
            if tm > best:
                best = tm
    return best


def main():
    parser = argparse.ArgumentParser(description="Score submission against validation labels")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--labels",     default="/home/ilan/kaggle/data/validation_labels.csv")
    parser.add_argument("--sample",     default="data/raw/sample_submission.csv")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("Submission Validation (approximate TM-score)")
    print(f"{'='*55}")

    # ── Load submission ──────────────────────────────────────────
    print(f"\nLoading submission: {args.submission}")
    sub = pd.read_csv(args.submission)
    sub["target"] = sub["ID"].str.rsplit("_", n=1).str[0]
    sub_targets = sorted(sub["target"].unique())
    print(f"  Targets in submission: {len(sub_targets)}")

    # ── Load labels ──────────────────────────────────────────────
    print(f"Loading labels:     {args.labels}")
    lbl = pd.read_csv(args.labels)
    lbl = lbl.copy()
    lbl["target"] = lbl["ID"].str.rsplit("_", n=1).str[0]
    lbl_targets = sorted(lbl["target"].unique())
    print(f"  Targets in labels:     {len(lbl_targets)}")

    # Detect available slots in labels (up to 40)
    n_ref_slots = sum(1 for i in range(1, 41) if f"x_{i}" in lbl.columns)
    n_sub_slots = sum(1 for i in range(1, 6)  if f"x_{i}" in sub.columns)
    print(f"  Label slots:  {n_ref_slots} reference conformations")
    print(f"  Sub slots:    {n_sub_slots} predicted structures")

    # ── Score each target ────────────────────────────────────────
    results = []
    missing = []

    for tgt in lbl_targets:
        if tgt not in sub_targets:
            missing.append(tgt)
            results.append({"target": tgt, "tm_score": 0.0, "n_pred": 0, "n_ref": 0})
            continue

        ref_list  = load_coords(lbl, tgt, n_ref_slots)
        pred_list = load_coords(sub, tgt, n_sub_slots)

        tm = score_target(pred_list, ref_list)
        results.append({"target": tgt, "tm_score": tm,
                         "n_pred": len(pred_list), "n_ref": len(ref_list)})

    df_results = pd.DataFrame(results).sort_values("tm_score", ascending=False)

    # ── Report ───────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"{'Target':<10} {'TM-score':>9} {'N_pred':>7} {'N_ref':>7} {'Quality'}")
    print(f"{'─'*55}")
    for _, row in df_results.iterrows():
        quality = "✓ correct fold" if row["tm_score"] >= 0.45 else \
                  "~ partial"      if row["tm_score"] >= 0.25 else \
                  "✗ wrong fold"
        print(f"{row['target']:<10} {row['tm_score']:>9.4f} {int(row['n_pred']):>7} {int(row['n_ref']):>7}  {quality}")

    mean_tm = df_results["tm_score"].mean()
    n_correct = (df_results["tm_score"] >= 0.45).sum()

    print(f"{'─'*55}")
    print(f"\n{'MEAN TM-SCORE':>20}: {mean_tm:.4f}")
    print(f"{'Correct folds (≥0.45)':>20}: {n_correct}/{len(df_results)}")
    print(f"{'Vfold human expert':>20}: ~0.55  (Part 1 baseline)")
    print(f"{'Top Part 1 teams':>20}: ~0.59–0.64")

    if missing:
        print(f"\n⚠ Missing targets in submission: {missing}")

    # ── Save results ─────────────────────────────────────────────
    out_csv = Path(args.submission).with_suffix(".scores.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nPer-target scores saved: {out_csv}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
