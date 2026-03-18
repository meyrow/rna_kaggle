"""
scripts/local_eval.py — Local evaluation for TBM-direct pipeline.

Mirrors exactly what the Kaggle notebook does:
  1. Load templates from data/pdb_cache/template_predictions.json
  2. SW-align each test sequence to its template
  3. Build submission CSV (TBM or stub fallback)
  4. Score against validation_labels.csv

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/local_eval.py

    # Custom paths:
    python3 scripts/local_eval.py \
        --data    /home/ilan/kaggle/data \
        --output  outputs/my_submission.csv \
        --labels  /home/ilan/kaggle/data/validation_labels.csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.tm_score import _tm_approx

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR   = '/home/ilan/kaggle/data'
OUTPUT_DIR = 'outputs'
TEMPLATE_JSON = 'data/pdb_cache/template_predictions.json'
SENTINEL   = -1e18
SEEDS      = [42, 123, 456, 789, 1337]
RESNAME    = {'A':'A','C':'C','G':'G','U':'U','T':'U','a':'A','c':'C','g':'G','u':'U','N':'N'}
COLS       = ['ID','resname','resid'] + [f'{ax}_{i}' for i in range(1,6) for ax in ['x','y','z']]

# SW alignment
_M, _X, _G, _MIN_COV = 2, -1, -2, 0.65

def sw_align(a, b):
    m, n = len(a), len(b)
    H = np.zeros((m+1, n+1), dtype=np.int32)
    for i in range(1, m+1):
        for j in range(1, n+1):
            s = _M if a[i-1]==b[j-1] else _X
            H[i,j] = max(0, H[i-1,j-1]+s, H[i-1,j]+_G, H[i,j-1]+_G)
    i, j = divmod(int(H.argmax()), n+1)
    mapping = []
    while H[i,j] > 0 and i > 0 and j > 0:
        s = _M if a[i-1]==b[j-1] else _X
        if   H[i,j] == H[i-1,j-1]+s: mapping.append((i-1,j-1)); i-=1; j-=1
        elif H[i,j] == H[i-1,j]+_G:  i -= 1
        else:                          j -= 1
    mapping.reverse()
    return mapping, len(mapping)/len(a) if a else 0.0

def get_coords(tid, seq, templates):
    if tid not in templates:
        return None, 'stub (no template)'
    t     = templates[tid]
    c     = t['coords']
    t_seq = t['template_seq']
    L     = len(seq)
    if t_seq and len(seq) >= 10:
        mapping, cov = sw_align(seq, t_seq)
        if cov >= _MIN_COV and mapping:
            safe    = min(len(c), len(t_seq))
            aligned = np.array([c[j] for (_,j) in mapping if j < safe], dtype=np.float32)
            if len(aligned) < L:
                d     = aligned[-1]-aligned[-2] if len(aligned)>=2 else np.zeros(3)
                extra = np.array([aligned[-1]+d*(i+1) for i in range(L-len(aligned))], dtype=np.float32)
                aligned = np.vstack([aligned, extra])
            return aligned[:L], f"TBM {t['pident']:.0f}% (cov={cov:.2f})"
    # Legacy trim fallback
    base = c[:L] if len(c)>=L else np.vstack([c, np.zeros((L-len(c),3),dtype=np.float32)])
    return base, f"TBM {t['pident']:.0f}% (legacy-trim)"

def stub_coords(seq, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(seq)
    t   = np.linspace(0, n*0.6, n)
    c   = np.stack([9*np.cos(t), 9*np.sin(t), 2.8*np.arange(n)], axis=1).astype(np.float32)
    return c + rng.normal(0, 0.5, c.shape).astype(np.float32)

def build_submission(test, templates, output_csv, rhofold=None):
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    rows   = []
    n_tbm  = n_stub = n_rf = 0
    print(f"\n{'Target':<14} {'Len':>5}  {'Source'}")
    print('-' * 50)
    for _, row in test.iterrows():
        tid, seq = row['target_id'], row['sequence']
        L        = len(seq)
        base, src = get_coords(tid, seq, templates)
        if base is None and rhofold and tid in rhofold:
            base  = rhofold[tid]['coords']
            src   = f"RhoFold (pLDDT={rhofold[tid]['plddt']:.2f})"
            n_rf += 1
        elif base is None:
            base = stub_coords(seq)
            src  = 'stub'
            n_stub += 1
        else:
            n_tbm += 1
        noise = 0.05 if ('TBM' in src or 'RhoFold' in src) else 0.5
        print(f"  {tid:<12} {L:>5}  {src}")
        all_c = [base + np.random.default_rng(s).normal(0, noise, base.shape).astype(np.float32) for s in SEEDS]
        for j in range(L):
            r = {'ID': f'{tid}_{j+1}', 'resname': RESNAME.get(seq[j].upper(),'N'), 'resid': j+1}
            for k, c in enumerate(all_c):
                r[f'x_{k+1}'] = round(float(c[j,0]),3)
                r[f'y_{k+1}'] = round(float(c[j,1]),3)
                r[f'z_{k+1}'] = round(float(c[j,2]),3)
            rows.append(r)
    pd.DataFrame(rows)[COLS].to_csv(output_csv, index=False)
    print(f'\nTBM: {n_tbm}/28  RhoFold: {n_rf}/28  Stub: {n_stub}/28')
    print(f'Submission: {output_csv}  ({len(rows):,} rows)')
    return n_tbm, n_stub

def score(submission_csv, labels_csv):
    lbl = pd.read_csv(labels_csv)
    lbl['target'] = lbl['ID'].str.rsplit('_', n=1).str[0]
    sub = pd.read_csv(submission_csv)
    sub['target'] = sub['ID'].str.rsplit('_', n=1).str[0]
    n_ref = sum(1 for i in range(1,41) if f'x_{i}' in lbl.columns)

    results = []
    for tgt in sorted(lbl['target'].unique()):
        ls    = lbl[lbl['target']==tgt].sort_values('resid')
        ps    = sub[sub['target']==tgt].sort_values('resid')
        refs  = [ls.loc[ls[f'x_{i}']!=SENTINEL,[f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32)
                 for i in range(1,n_ref+1)
                 if f'x_{i}' in ls.columns and not (ls[f'x_{i}']==SENTINEL).all()]
        preds = [ps[[f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32) for i in range(1,6)]
        best  = 0.0
        for p in preds:
            for r in refs:
                n = min(len(p), len(r))
                best = max(best, _tm_approx(p[:n], r[:n]))
        results.append({'target': tgt, 'tm': best})

    df = pd.DataFrame(results).sort_values('tm', ascending=False)
    print(f"\n{'Target':<12} {'TM':>7}  {'Quality'}")
    print('-' * 40)
    for _, r in df.iterrows():
        q = '✓ correct' if r['tm']>=0.45 else ('~ partial' if r['tm']>=0.25 else '✗ wrong')
        print(f"  {r['target']:<10} {r['tm']:>7.4f}  {q}")
    mean = df['tm'].mean()
    print(f"\n{'MEAN TM-SCORE':>20} : {mean:.4f}")
    print(f"{'Correct (≥0.45)':>20} : {(df['tm']>=0.45).sum()}/28")
    print(f"{'Vfold human expert':>20} : ~0.55")
    print(f"{'Top Part 1 teams':>20} : ~0.59–0.64")
    return mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default=DATA_DIR)
    parser.add_argument('--templates', default=TEMPLATE_JSON)
    parser.add_argument('--output',    default=f'{OUTPUT_DIR}/submission_local.csv')
    parser.add_argument('--labels',    default=None)
    args = parser.parse_args()

    labels_csv = args.labels or f'{args.data}/validation_labels.csv'

    print('Loading templates...')
    with open(args.templates) as f:
        raw = json.load(f)
    templates = {k: {
        'coords':       np.array(v['coords'], dtype=np.float32),
        'pident':       v['pident'],
        'template_seq': v.get('template_seq', ''),
    } for k, v in raw.items()}
    print(f'  {len(templates)}/28 templates loaded')

    # Load RhoFold predictions
    rhofold_json = args.templates.replace('template_predictions.json', 'rhofold_predictions.json')
    rhofold = {}
    if os.path.exists(rhofold_json):
        with open(rhofold_json) as f:
            rraw = json.load(f)
        rhofold = {k: {'coords': np.array(v['coords'], dtype=np.float32), 'plddt': v['plddt']}
                   for k, v in rraw.items()}
        print(f'  {len(rhofold)} RhoFold predictions loaded')

    print('\nLoading test sequences...')
    test = pd.read_csv(f'{args.data}/test_sequences.csv')
    test['sequence'] = test['sequence'].str.upper().str.replace('T', 'U')

    t0 = time.time()
    build_submission(test, templates, args.output, rhofold)
    build_time = time.time() - t0
    print(f'Build time: {build_time:.1f}s')

    if os.path.exists(labels_csv):
        t0 = time.time()
        score(args.output, labels_csv)
        print(f'Score time: {time.time()-t0:.1f}s')
    else:
        print(f'\nNo labels at {labels_csv} — skipping score')

if __name__ == '__main__':
    main()
