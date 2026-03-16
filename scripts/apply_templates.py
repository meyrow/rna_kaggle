"""
scripts/apply_templates.py — Apply PDB templates directly to test sequences.

For sequences with high PDB identity (>90%), use the template C1' coords
directly as predictions. This is far more accurate than any de-novo model
for known structures.

Usage:
    cd ~/rna_kaggle
    python3 scripts/apply_templates.py

Creates: data/pdb_cache/template_predictions.pkl
  {target_id: {'coords': ndarray(L,3), 'pident': float, 'template': str}}
"""

import sys, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

HITS_TSV  = Path('data/pdb_cache/template_hits.tsv')
C1_CACHE  = Path('data/pdb_cache/pdb_c1_coords.pkl')
TEST_CSV  = Path('/home/ilan/kaggle/data/test_sequences.csv')
OUT_PKL   = Path('data/pdb_cache/template_predictions.pkl')

MIN_PIDENT = 85.0   # use template if >=85% sequence identity

def load_best_hits(hits_tsv, min_pident):
    """Load best template hit per query sequence."""
    hits = {}
    with open(hits_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            query, target, pident = parts[0], parts[1], float(parts[2])
            if pident >= min_pident:
                if query not in hits or pident > hits[query][1]:
                    hits[query] = (target, pident)
    return hits

def align_and_trim(ref_coords, query_len):
    """Trim or pad template coords to match query length."""
    ref_len = len(ref_coords)
    if ref_len >= query_len:
        return ref_coords[:query_len]
    else:
        # Pad with extrapolated coords
        pad = query_len - ref_len
        if ref_len >= 2:
            direction = ref_coords[-1] - ref_coords[-2]
            extra = np.array([ref_coords[-1] + direction * (i+1) 
                             for i in range(pad)], dtype=np.float32)
        else:
            extra = np.zeros((pad, 3), dtype=np.float32)
        return np.vstack([ref_coords, extra])

def main():
    print('Loading template hits...')
    best_hits = load_best_hits(HITS_TSV, MIN_PIDENT)
    print(f'  {len(best_hits)} sequences with pident >= {MIN_PIDENT}%')

    print('Loading C1\' coordinate cache...')
    with open(C1_CACHE, 'rb') as f:
        c1_cache = pickle.load(f)
    print(f'  {len(c1_cache)} chains in cache')

    print('Loading test sequences...')
    df = pd.read_csv(TEST_CSV)
    df['sequence'] = df['sequence'].str.upper().str.replace('T', 'U')

    results = {}
    print('\nApplying templates:')
    print(f'{"Target":<12} {"Template":<20} {"pident":>8} {"Q_len":>6} {"T_len":>6} {"Status"}')
    print('-' * 65)

    for _, row in df.iterrows():
        tid = row['target_id']
        seq = row['sequence']
        qlen = len(seq)

        if tid not in best_hits:
            continue

        template_chain, pident = best_hits[tid]

        # Try to find coords in cache
        coords = c1_cache.get(template_chain)
        if coords is None:
            # Try different chain naming conventions
            for key in c1_cache:
                if key.upper() == template_chain.upper():
                    coords = c1_cache[key]
                    break

        if coords is None:
            print(f'{tid:<12} {template_chain:<20} {pident:>7.1f}% {qlen:>6}   N/A   CACHE MISS')
            continue

        tlen = len(coords)

        # Skip only when template is LONGER than query by >20% (trimming loses structure)
        # When template is shorter, padding with A-form is acceptable
        if tlen > qlen * 1.20 and pident < 99.0:
            len_diff = abs(tlen - qlen) / max(tlen, qlen)
            print(f'{tid:<12} {template_chain:<20} {pident:>7.1f}% {qlen:>6} {tlen:>6}   SKIP (template too long {len_diff:.0%})')
            continue

        trimmed = align_and_trim(coords, qlen)
        exact = (pident == 100.0 and tlen == qlen)
        results[tid] = {
            'coords':   trimmed,
            'pident':   pident,
            'template': template_chain,
            'q_len':    qlen,
            't_len':    tlen,
        }
        status = '✓ EXACT' if exact else f'✓ {pident:.0f}%'
        print(f'{tid:<12} {template_chain:<20} {pident:>7.1f}% {qlen:>6} {tlen:>6}   {status}')

    print(f'\nTotal template predictions: {len(results)}/28')
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    print(f'Saved: {OUT_PKL}')

    # Quick sanity check
    if results:
        sample_tid = next(iter(results))
        r = results[sample_tid]
        coords = r['coords']
        d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        print(f'\nSanity check ({sample_tid}):')
        print(f'  C1-C1 mean = {d.mean():.2f} A  (ideal=5.4)')
        print(f'  C1[0]      = {coords[0].round(3)}')

if __name__ == '__main__':
    main()
