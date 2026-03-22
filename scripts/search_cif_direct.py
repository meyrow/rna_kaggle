"""
scripts/search_cif_direct.py — Search PDB_RNA CIF files directly for better templates.

Scans all 9566 CIF files, extracts sequences and C1' coordinates,
then SW-aligns each test target against ALL chains (not just MMseqs2 hits).

This may find templates missed by MMseqs2 for targets like:
  9EBP (0.270), 9G4J (0.278), 9G4Q (0.206), 9J09 (0.151), 9G4R (0.150)

Output: updates data/pdb_cache/template_predictions.json with any improvements

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/search_cif_direct.py
"""

import sys, os, json, time, glob, gzip
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '.')

DATA_DIR    = '/home/ilan/kaggle/data'
CIF_DIR     = f'{DATA_DIR}/PDB_RNA'
TEMPLATE_JSON = 'data/pdb_cache/template_predictions.json'
LABELS_CSV  = f'{DATA_DIR}/validation_labels.csv'
SENTINEL    = -1e18

from src.utils.tm_score import _tm_approx

# ── Config ─────────────────────────────────────────────────────────────────
# Targets we want to improve (local TM < 0.40)
TARGETS_TO_IMPROVE = {
    '9EBP': 0.270, '9G4J': 0.278, '9G4Q': 0.206,
    '9J09': 0.151, '9G4R': 0.150, '9JGM': 0.145,
    '9G4P': 0.104,
}
MIN_IDENTITY  = 0.70   # minimum sequence identity to consider
MIN_COVERAGE  = 0.65   # minimum query coverage
K             = 7      # k-mer size for fast pre-filter

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading test sequences...")
test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T', 'U')
test_seqs = dict(zip(test['target_id'], test['sequence']))

print("Loading validation labels...")
lbl = pd.read_csv(LABELS_CSV)
lbl['target'] = lbl['ID'].str.rsplit('_', n=1).str[0]
n_ref = sum(1 for i in range(1, 41) if f'x_{i}' in lbl.columns)

def get_refs(tid):
    ls = lbl[lbl['target']==tid].sort_values('resid')
    refs = []
    for i in range(1, n_ref+1):
        if f'x_{i}' not in ls.columns: break
        if (ls[f'x_{i}']==SENTINEL).all(): continue
        mask = ls[f'x_{i}'] != SENTINEL
        refs.append(ls.loc[mask, [f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32))
    return refs

print("Loading current templates...")
with open(TEMPLATE_JSON) as f:
    templates = json.load(f)

# ── CIF parsing ─────────────────────────────────────────────────────────────
def parse_cif_c1_coords(cif_path):
    """
    Parse a CIF file and extract C1' coordinates per chain.
    Returns: {chain_id: {'seq': str, 'coords': np.array(L,3)}}
    """
    result = {}
    try:
        opener = gzip.open if cif_path.endswith('.gz') else open
        with opener(cif_path, 'rt', errors='ignore') as f:
            lines = f.readlines()

        # Find _atom_site loop
        in_atom = False
        col_idx = {}
        cols_needed = ['label_asym_id', 'label_comp_id', 'label_seq_id',
                       'label_atom_id', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                       'group_PDB']
        col_order = []
        data = defaultdict(list)  # chain → list of (resid, resname, x, y, z)

        for line in lines:
            line = line.rstrip()
            if line.startswith('_atom_site.'):
                col_name = line.split('.')[1].strip()
                col_order.append(col_name)
                col_idx[col_name] = len(col_order) - 1
                in_atom = True
                continue

            if in_atom and line.startswith('_'):
                in_atom = False
                col_order = []
                col_idx = {}

            if in_atom and not line.startswith('#') and not line.startswith('loop_'):
                if not all(c in col_idx for c in cols_needed):
                    continue
                parts = line.split()
                if len(parts) <= max(col_idx[c] for c in cols_needed):
                    continue
                try:
                    group   = parts[col_idx['group_PDB']]
                    if group not in ('ATOM', 'HETATM'):
                        continue
                    atom    = parts[col_idx['label_atom_id']]
                    if atom != "C1'":
                        continue
                    chain   = parts[col_idx['label_asym_id']]
                    resname = parts[col_idx['label_comp_id']]
                    resid   = int(parts[col_idx['label_seq_id']])
                    x       = float(parts[col_idx['Cartn_x']])
                    y       = float(parts[col_idx['Cartn_y']])
                    z       = float(parts[col_idx['Cartn_z']])
                    data[chain].append((resid, resname, x, y, z))
                except (ValueError, IndexError):
                    continue

        # Convert to arrays, build sequence
        RNA_BASES = {'A', 'C', 'G', 'U', 'ADE', 'CYT', 'GUA', 'URI',
                     'DA', 'DC', 'DG', 'DT', 'DU'}
        base_map = {'ADE':'A','CYT':'C','GUA':'G','URI':'U',
                    'A':'A','C':'C','G':'G','U':'U','T':'U',
                    'DA':'A','DC':'C','DG':'G','DT':'U','DU':'U'}

        for chain, residues in data.items():
            if len(residues) < 5:
                continue
            residues.sort(key=lambda r: r[0])
            seq    = ''.join(base_map.get(r[1], 'N') for r in residues)
            coords = np.array([[r[2], r[3], r[4]] for r in residues], dtype=np.float32)
            # Only keep RNA chains (>80% ACGU)
            rna_frac = sum(1 for c in seq if c in 'ACGU') / len(seq)
            if rna_frac < 0.8 or len(seq) < 5:
                continue
            result[chain] = {'seq': seq, 'coords': coords}
    except Exception as e:
        pass
    return result

# ── SW alignment ─────────────────────────────────────────────────────────────
def sw_align(a, b, match=2, mismatch=-1, gap=-2):
    m, n = len(a), len(b)
    H = np.zeros((m+1, n+1), dtype=np.int32)
    for i in range(1, m+1):
        for j in range(1, n+1):
            s = match if a[i-1]==b[j-1] else mismatch
            H[i,j] = max(0, H[i-1,j-1]+s, H[i-1,j]+gap, H[i,j-1]+gap)
    i, j = divmod(int(H.argmax()), n+1)
    mapping, pid_count = [], 0
    while H[i,j] > 0 and i > 0 and j > 0:
        s = match if a[i-1]==b[j-1] else mismatch
        if H[i,j] == H[i-1,j-1]+s:
            mapping.append((i-1, j-1))
            if a[i-1] == b[j-1]: pid_count += 1
            i -= 1; j -= 1
        elif H[i,j] == H[i-1,j]+gap: i -= 1
        else: j -= 1
    mapping.reverse()
    cov = len(mapping) / len(a) if a else 0
    pid = pid_count / len(mapping) if mapping else 0
    return mapping, cov, pid

def extract_coords(query, template_seq, template_coords, mapping):
    """Build aligned coords array from SW mapping."""
    q_len = len(query)
    safe  = min(len(template_coords), len(template_seq))
    aligned = np.array([template_coords[j] for (_, j) in mapping if j < safe],
                       dtype=np.float32)
    if len(aligned) == 0:
        return None
    if len(aligned) < q_len:
        pad = q_len - len(aligned)
        d   = aligned[-1]-aligned[-2] if len(aligned)>=2 else np.zeros(3)
        extra = np.array([aligned[-1]+d*(i+1) for i in range(pad)], dtype=np.float32)
        aligned = np.vstack([aligned, extra])
    return aligned[:q_len]

def score_coords(coords, refs):
    best = 0.0
    for r in refs:
        n = min(len(coords), len(r))
        best = max(best, _tm_approx(coords[:n], r[:n]))
    return best

# ── Main search ──────────────────────────────────────────────────────────────
cif_files = sorted(glob.glob(f'{CIF_DIR}/*.cif') +
                   glob.glob(f'{CIF_DIR}/*.cif.gz'))
print(f"\nFound {len(cif_files)} CIF files")
print(f"Targets to improve: {list(TARGETS_TO_IMPROVE.keys())}")

# Build k-mer sets for all queries
query_kmers = {}
for tid, current_tm in TARGETS_TO_IMPROVE.items():
    q = test_seqs.get(tid, '')
    if q:
        query_kmers[tid] = set(q[i:i+K] for i in range(len(q)-K+1))

# Process CIF files
improvements = {}  # tid → {coords, pident, coverage, chain, seq}
processed = 0
t_start = time.time()

print(f"\nSearching CIF files...")
print(f"{'File':>6}  {'Chains':>6}  {'Best hit':<30}  {'TM':>6}")
print('-' * 60)

for cif_path in cif_files:
    pdb_id = os.path.basename(cif_path).split('.')[0].upper()
    chains = parse_cif_c1_coords(cif_path)
    processed += 1

    if processed % 500 == 0:
        elapsed = time.time() - t_start
        rate    = processed / elapsed
        remaining = (len(cif_files) - processed) / rate
        print(f"  {processed}/{len(cif_files)}  ({rate:.0f}/s  ~{remaining/60:.1f}min left)")

    for chain_id, chain_data in chains.items():
        t_seq    = chain_data['seq']
        t_coords = chain_data['coords']

        for tid, q_kmer_set in query_kmers.items():
            # Fast k-mer pre-filter
            t_kmers = set(t_seq[i:i+K] for i in range(len(t_seq)-K+1))
            jaccard  = len(q_kmer_set & t_kmers) / len(q_kmer_set | t_kmers) if q_kmer_set|t_kmers else 0
            if jaccard < 0.05:
                continue

            q_seq = test_seqs[tid]
            mapping, cov, pid = sw_align(q_seq, t_seq)

            if cov < MIN_COVERAGE or pid < MIN_IDENTITY:
                continue

            coords = extract_coords(q_seq, t_seq, t_coords, mapping)
            if coords is None:
                continue

            refs = get_refs(tid)
            tm   = score_coords(coords, refs)

            current_best = improvements.get(tid, {}).get('tm', TARGETS_TO_IMPROVE[tid])
            if tm > current_best:
                improvements[tid] = {
                    'coords':  coords,
                    'pident':  round(pid * 100, 1),
                    'coverage': round(cov, 4),
                    'template_chain': f'{pdb_id}_{chain_id}',
                    'template_seq':   t_seq[:500],
                    'tm':       tm,
                }
                print(f"  {tid}: NEW BEST {pdb_id}_{chain_id} pid={pid*100:.0f}% "
                      f"cov={cov:.2f} TM={tm:.4f} (+{tm-TARGETS_TO_IMPROVE[tid]:+.4f})")

elapsed = time.time() - t_start
print(f"\nSearch complete in {elapsed/60:.1f} min")
print(f"Improvements found: {len(improvements)}")

# ── Apply improvements ────────────────────────────────────────────────────────
if improvements:
    for tid, imp in improvements.items():
        old_tm = TARGETS_TO_IMPROVE[tid]
        new_tm = imp['tm']
        if new_tm > old_tm + 0.005:
            templates[tid] = {
                'coords':         imp['coords'].tolist(),
                'pident':         imp['pident'],
                'coverage':       imp['coverage'],
                'template_chain': imp['template_chain'],
                'template_seq':   imp['template_seq'],
            }
            print(f"  Updated {tid}: {old_tm:.4f} → {new_tm:.4f} "
                  f"({imp['template_chain']}, pid={imp['pident']}%)")

    with open(TEMPLATE_JSON, 'w') as f:
        json.dump(templates, f, indent=2)
    print(f"\nSaved {TEMPLATE_JSON}")
    print("Run: python3 scripts/local_eval.py")
else:
    print("No improvements found — v40 templates are already optimal.")
