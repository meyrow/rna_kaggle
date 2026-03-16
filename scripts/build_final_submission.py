"""
scripts/build_final_submission.py — Combine template + Protenix predictions.

Priority:
  1. EXACT templates (pident=100%, len match): use PDB coords directly × 5
  2. HIGH templates (pident>=85%, len mismatch): template coords for matched region,
     Protenix for padding
  3. No template: use Protenix predictions (from local_eval output)
  4. Fallback: stub

Usage:
    cd ~/rna_kaggle
    python3 scripts/build_final_submission.py

Reads:
    data/pdb_cache/template_predictions.pkl
    /tmp/protenix_out/ or /tmp/local_eval_out/submission.csv (if eval done)
    /home/ilan/kaggle/data/test_sequences.csv

Writes:
    outputs/submission_tbm.csv
"""

import sys, os, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

TEMPLATE_PKL = Path('data/pdb_cache/template_predictions.pkl')
TEST_CSV     = Path('/home/ilan/kaggle/data/test_sequences.csv')
PROTENIX_DIR = Path('/tmp/protenix_out')
PREV_SUB     = Path('/tmp/local_eval_out/submission.csv')
OUT_CSV      = Path('outputs/submission_tbm.csv')

Path('outputs').mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1337]

def stub_coords(n, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n * 0.6, n)
    c = np.stack([9.0*np.cos(t), 9.0*np.sin(t), 2.8*np.arange(n)], axis=1)
    return (c + rng.normal(0, 0.5, c.shape)).astype(np.float32)

def load_protenix_coords(target_id, seq_len):
    """Load best Protenix prediction from output dir."""
    # Look in protenix output dir for any CIF file for this target
    pattern = list(PROTENIX_DIR.glob(f"**/{target_id}*/**/*.cif"))
    if not pattern:
        return None
    
    # Parse best CIF (highest pLDDT)
    best_coords = None
    best_plddt = -1
    
    for cif_path in pattern[:5]:
        coords, plddt = [], []
        try:
            with open(cif_path) as f:
                for line in f:
                    if not line.startswith('ATOM'):
                        continue
                    parts = line.split()
                    if parts[2].strip('"') == "C1'":
                        plddt.append(float(parts[13]))
                        coords.append([float(parts[14]), float(parts[15]), float(parts[16])])
        except Exception:
            continue
        if coords and np.mean(plddt) > best_plddt:
            best_plddt = np.mean(plddt)
            best_coords = np.array(coords, dtype=np.float32)
    
    return best_coords

def load_prev_submission_coords(target_id, seq_len):
    """Load coords from previous submission CSV."""
    if not PREV_SUB.exists():
        return None
    try:
        df = pd.read_csv(PREV_SUB)
        sub = df[df['ID'].str.startswith(target_id + '_')].sort_values('resid')
        if len(sub) == 0:
            return None
        coords = sub[['x_1','y_1','z_1']].values.astype(np.float32)
        return coords[:seq_len]
    except Exception:
        return None

def main():
    print('Loading template predictions...')
    with open(TEMPLATE_PKL, 'rb') as f:
        templates = pickle.load(f)
    print(f'  {len(templates)} sequences with templates')

    print('Loading test sequences...')
    df = pd.read_csv(TEST_CSV)
    df['sequence'] = df['sequence'].str.upper().str.replace('T', 'U')
    print(f'  {len(df)} test sequences')

    rows = []
    stats = {'template_exact': 0, 'template_approx': 0, 'protenix': 0, 'stub': 0}

    for _, row in df.iterrows():
        tid = row['target_id']
        seq = row['sequence']
        L   = len(seq)
        nuc_map = {'A':'A','C':'C','G':'G','U':'U','T':'U'}

        # ── Determine prediction source ───────────────────────────────
        if tid in templates:
            tmpl = templates[tid]
            coords = tmpl['coords'][:L]
            pident = tmpl['pident']
            tlen   = tmpl['t_len']

            if pident == 100.0 and tlen == L:
                # Perfect match — use coords directly × 5
                all_coords = [coords] * 5
                stats['template_exact'] += 1
                source = 'TBM_EXACT'
            else:
                # Partial match — template for known region, Protenix/stub for rest
                # Add small noise between seeds for diversity
                all_coords = []
                for i, seed in enumerate(SEEDS):
                    rng = np.random.default_rng(seed)
                    noise = rng.normal(0, 0.3, coords.shape).astype(np.float32)
                    all_coords.append(coords + noise)
                stats['template_approx'] += 1
                source = f'TBM_{pident:.0f}'
        else:
            # Try Protenix output, then previous submission, then stub
            protenix = load_protenix_coords(tid, L)
            if protenix is not None and len(protenix) >= L * 0.9:
                coords = protenix[:L]
                all_coords = [coords] * 5
                stats['protenix'] += 1
                source = 'PROTENIX'
            else:
                prev = load_prev_submission_coords(tid, L)
                if prev is not None and len(prev) >= L * 0.9:
                    all_coords = [prev[:L]] * 5
                    stats['protenix'] += 1
                    source = 'PREV_SUB'
                else:
                    all_coords = [stub_coords(L, seed) for seed in SEEDS]
                    stats['stub'] += 1
                    source = 'STUB'

        # Pad all_coords to exactly L
        all_coords = [
            np.vstack([c, stub_coords(L - len(c))]) if len(c) < L else c[:L]
            for c in all_coords
        ]

        # ── Build submission rows ─────────────────────────────────────
        for res_idx, nuc in enumerate(seq):
            resid = res_idx + 1
            row_data = {
                'ID': f'{tid}_{resid}',
                'resname': nuc_map.get(nuc.upper(), 'A'),
                'resid': resid,
            }
            for s_idx, c in enumerate(all_coords):
                row_data[f'x_{s_idx+1}'] = round(float(c[res_idx, 0]), 3)
                row_data[f'y_{s_idx+1}'] = round(float(c[res_idx, 1]), 3)
                row_data[f'z_{s_idx+1}'] = round(float(c[res_idx, 2]), 3)
            rows.append(row_data)

        print(f'  {tid:<12} len={L:<5} source={source}')

    # ── Write submission ──────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    col_order = ['ID','resname','resid'] + \
                [f'{a}_{i}' for i in range(1,6) for a in ['x','y','z']]
    out_df = out_df[col_order]
    out_df.to_csv(OUT_CSV, index=False)

    print(f'\n{"="*50}')
    print(f'Submission: {OUT_CSV}')
    print(f'  Rows    : {len(out_df):,}')
    print(f'  Targets : {len(df)}')
    print(f'\nSources:')
    print(f'  TBM exact    : {stats["template_exact"]:2d}/28')
    print(f'  TBM approx   : {stats["template_approx"]:2d}/28')
    print(f'  Protenix     : {stats["protenix"]:2d}/28')
    print(f'  Stub         : {stats["stub"]:2d}/28')
    print(f'{"="*50}')

    # Score if labels available
    labels = Path('/home/ilan/kaggle/data/validation_labels.csv')
    if labels.exists():
        print('\nScoring...')
        os.system(f'python3 scripts/validate_submission.py --submission {OUT_CSV} --labels {labels}')

if __name__ == '__main__':
    main()
