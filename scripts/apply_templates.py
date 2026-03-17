"""
scripts/apply_templates.py — Build template_predictions.json with alignment-based coords.

Implements all 5 TBM best practices:
  1. Always align before mapping   — SW alignment, never slice by index
  2. Handle offsets/gaps correctly — extract only aligned residue coordinates
  3. Fix multi-chain templates     — detect & skip bad coord/seq mismatches
  4. Trim by alignment, not length — extract template region from alignment
  5. Alignment confidence check   — fall back to denovo if coverage < MIN_COVERAGE

Usage:
    cd ~/kaggle/rna_folding
    python3 scripts/apply_templates.py

Creates: data/pdb_cache/template_predictions.json
  {target_id: {coords, pident, coverage, template_chain, template_seq}}
"""

import sys, os, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

HITS_TSV   = Path('data/pdb_cache/template_hits.tsv')
C1_CACHE   = Path('data/pdb_cache/pdb_c1_coords.pkl')
FASTA      = Path('data/pdb_cache/pdb_rna_seqs.fa')
TEST_CSV   = Path('/home/ilan/kaggle/data/test_sequences.csv')
OUT_JSON   = Path('data/pdb_cache/template_predictions.json')

MIN_PIDENT   = 85.0   # minimum % identity hit to consider
MIN_COVERAGE = 0.70   # minimum query coverage — below this, skip template
MIN_COORDS   = 10     # minimum aligned residues

# SW scoring
MATCH, MISMATCH, GAP = 2, -1, -2


def read_fasta(path):
    seqs = {}
    key  = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
            elif key:
                seqs[key] = seqs.get(key, '') + line
    return seqs


def load_best_hits(hits_tsv, min_pident):
    hits = {}
    with open(hits_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            query, target, pident, qlen = (
                parts[0], parts[1], float(parts[2]), int(parts[3])
            )
            if pident >= min_pident:
                if query not in hits or pident > hits[query]['pident']:
                    hits[query] = {
                        'target': target,
                        'pident': pident,
                        'qlen':   qlen,
                    }
    return hits


def sw_align(seq_a, seq_b):
    """
    Smith-Waterman local alignment.
    Returns list of (a_idx, b_idx) residue pairs and alignment stats.
    """
    m, n = len(seq_a), len(seq_b)
    H = np.zeros((m + 1, n + 1), dtype=np.int32)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s      = MATCH if seq_a[i-1] == seq_b[j-1] else MISMATCH
            match  = H[i-1, j-1] + s
            delete = H[i-1, j]   + GAP
            insert = H[i,   j-1] + GAP
            H[i, j] = max(0, match, delete, insert)

    # Traceback from max score
    i, j   = divmod(int(H.argmax()), n + 1)
    mapping = []

    while H[i, j] > 0 and i > 0 and j > 0:
        s = MATCH if seq_a[i-1] == seq_b[j-1] else MISMATCH
        if H[i, j] == H[i-1, j-1] + s:
            mapping.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif H[i, j] == H[i-1, j] + GAP:
            i -= 1
        else:
            j -= 1

    mapping.reverse()

    n_aligned = len(mapping)
    n_matches = sum(1 for a, b in mapping if seq_a[a] == seq_b[b])
    coverage  = n_aligned / len(seq_a) if seq_a else 0.0
    pident_sw = n_matches / n_aligned  if n_aligned else 0.0

    return mapping, coverage, pident_sw


def extract_aligned_coords(coords, template_seq, mapping):
    """
    Extract coords at the template positions from the alignment.

    The challenge: coords may have MORE residues than template_seq
    (multi-chain merge bug). We use only indices that are within
    the coord array AND correspond to aligned positions in template_seq.

    Since the coord array was built from the same chain as template_seq,
    valid coord indices are 0..min(len(coords), len(template_seq))-1.
    """
    safe_len = min(len(coords), len(template_seq))
    result   = []
    for (q_idx, t_idx) in mapping:
        if t_idx < safe_len:
            result.append(coords[t_idx])
    return np.array(result, dtype=np.float32) if result else None


def main():
    print('Loading test sequences...')
    df   = pd.read_csv(TEST_CSV)
    df['sequence'] = df['sequence'].str.upper().str.replace('T', 'U')
    test_seqs = dict(zip(df['target_id'], df['sequence']))
    print(f'  {len(test_seqs)} test sequences')

    print('Loading template hits...')
    best_hits = load_best_hits(HITS_TSV, MIN_PIDENT)
    print(f'  {len(best_hits)} sequences with pident >= {MIN_PIDENT}%')

    print('Loading C1\' coordinate cache...')
    with open(C1_CACHE, 'rb') as f:
        c1_cache = pickle.load(f)
    print(f'  {len(c1_cache)} chains in cache')

    print('Loading template sequences (FASTA)...')
    template_seqs = read_fasta(FASTA)
    print(f'  {len(template_seqs)} sequences')

    results = {}
    header  = (
        f"{'Target':<12} {'Template':<20} {'pident':>8} "
        f"{'Q_len':>6} {'Aligned':>8} {'Cov':>6}  {'Status'}"
    )
    print(f'\n{header}')
    print('-' * 75)

    for tid, query_seq in test_seqs.items():
        if tid not in best_hits:
            continue

        hit            = best_hits[tid]
        template_chain = hit['target']
        pident_mmseqs  = hit['pident']
        q_len          = len(query_seq)

        # ── 1. Get template sequence ──────────────────────────────────────
        template_seq = template_seqs.get(template_chain, '')
        if not template_seq:
            # Case-insensitive fallback
            lc = template_chain.lower()
            for k, v in template_seqs.items():
                if k.lower() == lc:
                    template_seq = v
                    break

        if not template_seq:
            print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
                  f'{q_len:>6}      N/A    N/A   SKIP (no seq)')
            continue

        # ── 2. Get coords ─────────────────────────────────────────────────
        coords = c1_cache.get(template_chain)
        if coords is None:
            for k in c1_cache:
                if k.upper() == template_chain.upper():
                    coords = c1_cache[k]
                    break

        if coords is None:
            print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
                  f'{q_len:>6}      N/A    N/A   SKIP (no coords)')
            continue

        # ── 3. SW alignment: query → template ────────────────────────────
        mapping, coverage, pident_sw = sw_align(query_seq, template_seq)

        if not mapping or len(mapping) < MIN_COORDS:
            print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
                  f'{q_len:>6} {len(mapping):>8} {coverage:>5.2f}   SKIP (no alignment)')
            continue

        # ── 4. Confidence check ───────────────────────────────────────────
        if coverage < MIN_COVERAGE:
            print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
                  f'{q_len:>6} {len(mapping):>8} {coverage:>5.2f}   SKIP (low coverage)')
            continue

        # ── 5. Extract coords at aligned template positions ───────────────
        aligned_coords = extract_aligned_coords(coords, template_seq, mapping)

        if aligned_coords is None or len(aligned_coords) < MIN_COORDS:
            print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
                  f'{q_len:>6} {len(mapping):>8} {coverage:>5.2f}   SKIP (coord extract failed)')
            continue

        # Sanity check: C1'-C1' bond distance (should be 4–8 Å for RNA)
        if len(aligned_coords) > 1:
            dists  = np.linalg.norm(np.diff(aligned_coords, axis=0), axis=1)
            median = float(np.median(dists))
            if not (3.0 < median < 10.0):
                print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
                      f'{q_len:>6} {len(mapping):>8} {coverage:>5.2f}   SKIP (bad geometry median={median:.1f}Å)')
                continue

        status = '✓ EXACT' if pident_mmseqs == 100.0 else f'✓ {pident_mmseqs:.0f}%'
        print(f'{tid:<12} {template_chain:<20} {pident_mmseqs:>7.1f}%  '
              f'{q_len:>6} {len(aligned_coords):>8} {coverage:>5.2f}   {status}')

        results[tid] = {
            'coords':         aligned_coords.tolist(),
            'pident':         pident_mmseqs,
            'coverage':       round(coverage, 4),
            'template_chain': template_chain,
            'template_seq':   template_seq[:500],  # store for Cell-17 runtime alignment
        }

    print(f'\nTotal template predictions: {len(results)}/28')

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {OUT_JSON}')

    # Sanity check on first entry
    if results:
        sample_tid = next(iter(results))
        r = results[sample_tid]
        coords = np.array(r['coords'])
        dists  = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        print(f'\nSanity check ({sample_tid}):')
        print(f'  Aligned residues : {len(coords)}')
        print(f'  Coverage         : {r["coverage"]:.2f}')
        print(f'  C1-C1 median     : {np.median(dists):.2f} Å  (ideal ≈ 5.4 Å)')
        print(f'  C1[0]            : {coords[0].round(3)}')


if __name__ == '__main__':
    main()
