"""
scripts/score_usalign.py — Score submission using actual USalign binary.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/score_usalign.py
"""
import subprocess, tempfile, os, sys, json
import pandas as pd, numpy as np

USALIGN = './USalign'
DATA_DIR = '/home/ilan/kaggle/data'
SENTINEL = -1e18

def write_pdb(coords, path, resnames=None):
    lines = []
    for i, (x,y,z) in enumerate(coords):
        rn = (resnames[i] if resnames else 'G').rjust(3)
        lines.append(
            f"ATOM  {i+1:5d}  C1' {rn} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

def usalign_tm(pred_coords, ref_coords):
    with tempfile.TemporaryDirectory() as d:
        write_pdb(pred_coords, f'{d}/pred.pdb')
        write_pdb(ref_coords,  f'{d}/ref.pdb')
        result = subprocess.run(
            [USALIGN, f'{d}/pred.pdb', f'{d}/ref.pdb', '-atom', ' C1\''],
            capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            if 'TM-score=' in line and 'Structure_2' in line:
                return float(line.split('TM-score=')[1].split()[0])
    return 0.0

# Load data
sub = pd.read_csv('outputs/submission_local.csv')
sub['target'] = sub['ID'].str.rsplit('_', n=1).str[0]
lbl = pd.read_csv(f'{DATA_DIR}/validation_labels.csv')
lbl['target'] = lbl['ID'].str.rsplit('_', n=1).str[0]
n_ref = sum(1 for i in range(1,41) if f'x_{i}' in lbl.columns)

print(f"{'Target':<12} {'L':>5}  {'TM-US':>7}  {'TM-RNA':>7}  {'delta':>7}")
print('-' * 45)

from src.utils.tm_score_rna import tm_score_rna

all_us, all_rna = [], []
for tid in sorted(sub['target'].unique()):
    ls    = lbl[lbl['target']==tid].sort_values('resid')
    # Get best ref
    refs = []
    for i in range(1, n_ref+1):
        if f'x_{i}' not in ls.columns: break
        if (ls[f'x_{i}']==SENTINEL).all(): continue
        mask = ls[f'x_{i}']!=SENTINEL
        refs.append(ls.loc[mask,[f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32))

    if not refs:
        continue

    # Get 5 predictions, take best
    rows = sub[sub['target']==tid].sort_values('resid')
    best_us, best_rna = 0.0, 0.0
    for k in range(1, 6):
        pred = rows[[f'x_{k}',f'y_{k}',f'z_{k}']].values.astype(np.float32)
        for ref in refs:
            n    = min(len(pred), len(ref))
            rna  = tm_score_rna(pred[:n], ref[:n])
            best_rna = max(best_rna, rna)
            us   = usalign_tm(pred[:n], ref[:n])
            best_us  = max(best_us, us)

    delta = best_us - best_rna
    flag  = '▲' if delta > 0.05 else ('▼' if delta < -0.05 else ' ')
    print(f"  {tid:<10} {len(rows):>5}  {best_us:>7.4f}  {best_rna:>7.4f}  {delta:>+6.3f} {flag}")
    all_us.append(best_us); all_rna.append(best_rna)

print('-' * 45)
print(f"  {'MEAN':<10} {'':>5}  {np.mean(all_us):>7.4f}  {np.mean(all_rna):>7.4f}  {np.mean(all_us)-np.mean(all_rna):>+6.3f}")
print(f"\nUSalign mean:  {np.mean(all_us):.4f}")
print(f"RNA formula:   {np.mean(all_rna):.4f}")
print(f"Kaggle public: 0.0980")
