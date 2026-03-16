"""
scripts/analyze_data.py — Analyze real competition data from your local kaggle/data/ folder.

Usage:
    python scripts/analyze_data.py --data_dir /home/ilan/kaggle/data

Reads:
    - test_sequences.csv
    - validation_sequences.csv
    - validation_labels.csv
    - sample_submission.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze(data_dir: str):
    d = Path(data_dir)
    print(f"\n{'='*65}")
    print("RNA 3D Folding Part 2 — Data Analysis")
    print(f"Data directory: {d}")
    print(f"{'='*65}")

    # ── Load ──────────────────────────────────────────────────────
    tseq = pd.read_csv(d / "test_sequences.csv")
    vlbl = pd.read_csv(d / "validation_labels.csv")
    ssub = pd.read_csv(d / "sample_submission.csv")
    print(f"\nLoaded: {len(tseq)} test sequences, {len(vlbl):,} label rows, {len(ssub):,} submission rows")

    # ── Enrich ────────────────────────────────────────────────────
    tseq["len"]        = tseq["sequence"].str.len()
    tseq["gc"]         = tseq["sequence"].apply(lambda s: (s.count("G")+s.count("C"))/len(s))
    tseq["has_ligand"] = tseq["ligand_ids"].notna() & (tseq["ligand_ids"].astype(str).str.len() > 1)
    tseq["n_copies"]   = tseq["stoichiometry"].str.extract(r":(\d+)$").astype(float).fillna(1).astype(int)
    tseq["is_complex"] = tseq["stoichiometry"].str.contains(";") | (tseq["n_copies"] > 1)

    # Count real reference structures (avoid DataFrame fragmentation)
    vlbl = vlbl.copy()  # defragment before column ops
    vlbl["target"] = vlbl["ID"].str.rsplit("_", n=1).str[0]
    x_cols = [f"x_{i}" for i in range(1, 41) if f"x_{i}" in vlbl.columns]
    sentinel_mask = vlbl[x_cols] != -1e18
    real_slots = sentinel_mask.groupby(vlbl["target"]).any().sum(axis=1).to_dict()
    tseq["n_ref"] = tseq["target_id"].map(real_slots).fillna(0).astype(int)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("SEQUENCE STATISTICS")
    print(f"{'─'*40}")
    print(f"  Count:          {len(tseq)}")
    print(f"  Length min/max: {tseq['len'].min()} / {tseq['len'].max()} nt")
    print(f"  Length median:  {tseq['len'].median():.0f} nt")
    print(f"  GC content:     {tseq['gc'].mean():.1%} avg")
    print(f"  Has ligands:    {tseq['has_ligand'].sum()} / {len(tseq)}")
    print(f"  Multi-chain:    {tseq['is_complex'].sum()} / {len(tseq)}")
    print(f"  Max copies:     {tseq['n_copies'].max()} (9MME octamer U:8)")

    print(f"\n{'─'*40}")
    print("LABEL / REFERENCE STRUCTURE STATISTICS")
    print(f"{'─'*40}")
    print(f"  Label rows:         {len(vlbl):,}")
    print(f"  Max ref structures: {tseq['n_ref'].max()} (9LJN has 11 conformations)")
    print(f"  Avg ref structures: {tseq['n_ref'].mean():.1f}")
    sentinel_count = (vlbl[[f"x_{i}" for i in range(1,41) if f"x_{i}" in vlbl.columns]] == -1e18).sum().sum()
    total_cells    = len(vlbl) * 40
    print(f"  Sentinel (-1e18):   {sentinel_count:,} / {total_cells:,} ({100*sentinel_count/total_cells:.1f}%)")

    print(f"\n{'─'*40}")
    print("HARDWARE TIER BREAKDOWN (RTX 4060, 8GB VRAM)")
    print(f"{'─'*40}")
    tier_map = {
        "A: <200nt":     tseq["len"] < 200,
        "B: 200–500nt":  (tseq["len"] >= 200) & (tseq["len"] < 500),
        "C: 500–1500nt": (tseq["len"] >= 500) & (tseq["len"] < 1500),
        "D: >1500nt":    tseq["len"] >= 1500,
    }
    for tier, mask in tier_map.items():
        sub = tseq[mask]
        ids = ", ".join(sub["target_id"])
        print(f"  {tier:20s}: {len(sub)} targets  [{ids}]")

    print(f"\n{'─'*40}")
    print("MULTI-CHAIN TARGETS (require special handling)")
    print(f"{'─'*40}")
    for _, r in tseq[tseq["is_complex"]].sort_values("len").iterrows():
        print(f"  {r['target_id']:8s}  {r['len']:5d} nt  stoich={r['stoichiometry']:15s}  {r['description'][:50]}")

    print(f"\n{'─'*40}")
    print("LIGAND-BOUND TARGETS")
    print(f"{'─'*40}")
    for _, r in tseq[tseq["has_ligand"]].sort_values("len").iterrows():
        print(f"  {r['target_id']:8s}  {r['len']:5d} nt  ligands={str(r['ligand_ids']):20s}  {r['description'][:45]}")

    print(f"\n{'─'*40}")
    print("SAMPLE SUBMISSION FORMAT")
    print(f"{'─'*40}")
    print(f"  Columns: {list(ssub.columns)}")
    print(f"  Rows: {len(ssub):,}")
    print(f"  Format: 5 structure slots per residue (x_1..z_5)")
    print(f"  Sample:")
    print(ssub.head(2).to_string())

    print(f"\n{'─'*40}")
    print("ROUTING PREDICTIONS (which branch each target needs)")
    print(f"{'─'*40}")
    for _, r in tseq.sort_values("len").iterrows():
        # Heuristic routing hint based on length/stoichiometry
        if r["len"] > 1000:
            branch = "de-novo (chunked)"
        elif r["is_complex"] and r["n_copies"] > 4:
            branch = "de-novo (large assembly)"
        elif r["len"] < 100:
            branch = "TBM (small, likely templated)"
        else:
            branch = "TBM or de-novo (router decides)"
        lig = f" +{r['ligand_ids']}" if r["has_ligand"] else ""
        print(f"  {r['target_id']:8s} {r['len']:5d}nt  {r['stoichiometry']:12s}{lig:15s} → {branch}")

    print(f"\n{'='*65}")
    print("DONE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ilan/kaggle/data",
                        help="Path to your local kaggle data directory")
    args = parser.parse_args()
    analyze(args.data_dir)
