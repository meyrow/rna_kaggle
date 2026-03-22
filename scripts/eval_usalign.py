"""
scripts/eval_usalign.py — Evaluate submission using real US-align binary.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/eval_usalign.py
"""
import subprocess, tempfile, os, sys, json, time
import pandas as pd, numpy as np

sys.path.insert(0, '.')

USALIGN  = './USalign'
DATA_DIR = '/home/ilan/kaggle/data'
SENTINEL = -1e18

def write_pdb(coords, path):
    lines = []
    for i, (x,y,z) in enumerate(coords):
        lines.append(
            f"ATOM  {i+1:5d}  C1' G   A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

def usalign_tm(pred, ref):
    """Run USalign, return TM-score normalized by ref length."""
    with tempfile.TemporaryDirectory() as d:
        write_pdb(pred, f'{d}/pred.pdb')
        write_pdb(ref,  f'{d}/ref.pdb')
        r = subprocess.run(
            [USALIGN, f'{d}/pred.pdb', f'{d}/ref.pdb', '-atom', " C1'"],
            capture_output=True, text=True
        )
        tms = [float(l.split('TM-score=')[1].split()[0])
               for l in r.stdout.split('\n')
               if l.strip().startswith('TM-score=')]
        return tms[1] if len(tms) >= 2 else (tms[0] if tms else 0.0)

# ── Load data ─────────────────────────────────────────────────────────────────
sub = pd.read_csv('outputs/submission_local.csv')
lbl = pd.read_csv(f'{DATA_DIR}/validation_labels.csv')
lbl['target'] = lbl['ID'].str.rsplit('_', n=1).str[0]
n_ref = sum(1 for i in range(1,41) if f'x_{i}' in lbl.columns)

# ── Score ─────────────────────────────────────────────────────────────────────
results = []
t0 = time.time()
targets = sorted(sub['ID'].str.rsplit('_',n=1).str[0].unique())

print(f"Scoring {len(targets)} targets with US-align...")
print(f"{'Target':<12} {'L':>5}  {'TM':>7}  Quality")
print('-' * 40)

for tid in targets:
    ls = lbl[lbl['target']==tid].sort_values('resid')

    # Collect all refs
    refs = []
    for i in range(1, n_ref+1):
        if f'x_{i}' not in ls.columns: break
        if (ls[f'x_{i}']==SENTINEL).all(): continue
        mask = ls[f'x_{i}'] != SENTINEL
        refs.append(ls.loc[mask,[f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32))

    rows = sub[sub['ID'].str.startswith(f'{tid}_')].sort_values('ID')
    L    = len(rows)

    best_tm = 0.0
    for k in range(1, 6):
        pred = rows[[f'x_{k}',f'y_{k}',f'z_{k}']].values.astype(np.float32)
        for ref in refs:
            n  = min(len(pred), len(ref))
            tm = usalign_tm(pred[:n], ref[:n])
            best_tm = max(best_tm, tm)

    quality = '✓ correct' if best_tm >= 0.45 else ('~ partial' if best_tm >= 0.25 else '✗ wrong')
    print(f"  {tid:<10} {L:>5}  {best_tm:>7.4f}  {quality}")
    results.append((tid, L, best_tm))

results.sort(key=lambda x: -x[2])
print('\n' + '='*40)
mean_tm = np.mean([r[2] for r in results])
correct = sum(1 for r in results if r[2] >= 0.45)
print(f"  MEAN TM-SCORE (US-align): {mean_tm:.4f}")
print(f"  Correct (≥0.45): {correct}/{len(results)}")
print(f"  Time: {time.time()-t0:.1f}s")

# Save per-target scores
df = pd.DataFrame(results, columns=['target','L','tm_usalign'])
df.to_csv('/tmp/usalign_eval.csv', index=False)
print(f"\nSaved: /tmp/usalign_eval.csv")
