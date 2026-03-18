"""
scripts/test_rhofold.py — Test RhoFold inference on stub targets.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/test_rhofold.py

Tests RhoFold on the 8 stub targets and scores against validation labels.
"""

import sys, os, time, tempfile, json
import numpy as np
import pandas as pd

RHOFOLD_REPO = '/home/ilan/kaggle/data/models/rhofold'
CKPT         = '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt'
DATA_DIR     = '/home/ilan/kaggle/data'

sys.path.insert(0, RHOFOLD_REPO)
sys.path.insert(0, '.')

# ── 1. Load RhoFold ───────────────────────────────────────────────────────────
print("Loading RhoFold...")
import torch
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

model = RhoFold(rhofold_config)
ckpt  = torch.load(CKPT, map_location=DEVICE)
state = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state)
model = model.to(DEVICE).eval()
print("RhoFold loaded OK")

# ── 2. Stub targets ───────────────────────────────────────────────────────────
test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T', 'U')

with open('data/pdb_cache/template_predictions.json') as f:
    templates = json.load(f)

stub_targets = test[~test['target_id'].isin(templates)].copy()
print(f"\nStub targets: {len(stub_targets)}")
for _, r in stub_targets.iterrows():
    print(f"  {r['target_id']}: {len(r['sequence'])}nt")

# ── 3. Run inference ──────────────────────────────────────────────────────────
def run_rhofold(seq, device=DEVICE):
    """Run RhoFold on a single sequence. Returns C1' coords shape (L, 3)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fas_path = os.path.join(tmpdir, 'input.fasta')
        with open(fas_path, 'w') as f:
            f.write(f'>query\n{seq}\n')

        result        = get_features(fas_path, fas_path)
        tokens        = result['tokens'].to(device)
        rna_fm_tokens = result['rna_fm_tokens'].to(device)
        seq_out       = result['seq']

        with torch.no_grad():
            outputs = model(tokens, rna_fm_tokens, seq_out)

        output = outputs[-1]

        # Use the explicit C1' output key
        c1_coords = output["cords_c1'"][0].cpu().numpy()  # shape (L, 3)
        plddt     = output['plddt'][0].cpu().numpy()       # shape (L,)

        return c1_coords.astype(np.float32), float(plddt.mean())

print("\nRunning RhoFold inference...")
results = {}
for _, row in stub_targets.iterrows():
    tid = row['target_id']
    seq = row['sequence']
    if len(seq) > 500:
        print(f"  {tid}: SKIP (len={len(seq)}, too long for test)")
        continue
    print(f"  {tid} ({len(seq)}nt)...", end=' ', flush=True)
    t0 = time.time()
    try:
        coords, plddt = run_rhofold(seq)
        elapsed = time.time() - t0
        dists = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        print(f"OK in {elapsed:.1f}s  C1-C1 mean={dists.mean():.2f}Å  pLDDT={plddt:.1f}")
        results[tid] = coords
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nSuccessful: {len(results)}/{len(stub_targets)}")

# ── 4. Score against validation labels ───────────────────────────────────────
if results:
    from src.utils.tm_score import _tm_approx
    SENTINEL = -1e18
    lbl = pd.read_csv(f'{DATA_DIR}/validation_labels.csv')
    lbl['target'] = lbl['ID'].str.rsplit('_', n=1).str[0]
    n_ref = sum(1 for i in range(1,41) if f'x_{i}' in lbl.columns)

    print(f"\n{'Target':<12} {'TM':>7}  {'vs stub':>8}")
    stub_baseline = {'9G4Q':0.0776,'9WHV':0.0235,'9J09':0.0103,'9G4P':0.0089,
                     '9JFS':0.0084,'9RVP':0.0078,'9ZCC':0.0044,'9I9W':0.0025,
                     '9OD4':0.0021,'9QZJ':0.0017,'9MME':0.0008}

    for tid, coords in results.items():
        ls   = lbl[lbl['target']==tid].sort_values('resid')
        refs = [ls.loc[ls[f'x_{i}']!=SENTINEL,[f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32)
                for i in range(1,n_ref+1)
                if f'x_{i}' in ls.columns and not (ls[f'x_{i}']==SENTINEL).all()]
        best = 0.0
        for r in refs:
            n = min(len(coords), len(r))
            best = max(best, _tm_approx(coords[:n], r[:n]))
        delta = best - stub_baseline.get(tid, 0)
        flag  = '▲' if delta > 0.01 else ('▼' if delta < -0.01 else ' ')
        print(f"  {tid:<10} {best:>7.4f}  {delta:>+7.4f} {flag}")
