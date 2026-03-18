"""
scripts/build_rhofold_cache.py — Run RhoFold on stub targets and save to JSON.

Output: data/pdb_cache/rhofold_predictions.json
  {target_id: {'coords': [[x,y,z],...], 'plddt': float}}

Upload this file to the rna-templates Kaggle dataset alongside template_predictions.json.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/build_rhofold_cache.py
"""

import sys, os, time, tempfile, json
import numpy as np
import pandas as pd

RHOFOLD_REPO = '/home/ilan/kaggle/data/models/rhofold'
CKPT         = '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt'
DATA_DIR     = '/home/ilan/kaggle/data'
OUT_JSON     = 'data/pdb_cache/rhofold_predictions.json'
MAX_LEN      = 500  # skip sequences longer than this

sys.path.insert(0, RHOFOLD_REPO)
sys.path.insert(0, '.')

import torch
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

print("Loading RhoFold...")
model = RhoFold(rhofold_config)
ckpt  = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model = model.to(DEVICE).eval()
print("RhoFold loaded OK")

# Load test sequences
test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T', 'U')

# Load existing TBM templates — skip those targets
with open('data/pdb_cache/template_predictions.json') as f:
    templates = json.load(f)

stub_targets = test[~test['target_id'].isin(templates)].copy()
print(f"\nStub targets: {len(stub_targets)}")

def run_rhofold(seq):
    with tempfile.TemporaryDirectory() as tmpdir:
        fas_path = os.path.join(tmpdir, 'input.fasta')
        with open(fas_path, 'w') as f:
            f.write(f'>query\n{seq}\n')
        result        = get_features(fas_path, fas_path)
        tokens        = result['tokens'].to(DEVICE)
        rna_fm_tokens = result['rna_fm_tokens'].to(DEVICE)
        seq_out       = result['seq']
        with torch.no_grad():
            outputs = model(tokens, rna_fm_tokens, seq_out)
        out       = outputs[-1]
        c1_coords = out["cords_c1'"][0][0].cpu().numpy()   # (L, 3)
        plddt     = out['plddt'][0][0].cpu().numpy().mean() # scalar
        return c1_coords.astype(np.float32), float(plddt)

results = {}
print(f"\n{'Target':<12} {'Len':>5}  {'Time':>6}  {'C1-C1':>6}  {'pLDDT':>6}  Status")
print('-' * 55)

for _, row in stub_targets.iterrows():
    tid = row['target_id']
    seq = row['sequence']
    L   = len(seq)

    if L > MAX_LEN:
        print(f"{tid:<12} {L:>5}  {'':>6}  {'':>6}  {'':>6}  SKIP (>{MAX_LEN}nt)")
        continue

    print(f"{tid:<12} {L:>5}  ", end='', flush=True)
    t0 = time.time()
    try:
        coords, plddt = run_rhofold(seq)
        elapsed = time.time() - t0
        dists   = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        print(f"{elapsed:>5.1f}s  {dists.mean():>5.2f}Å  {plddt:>6.3f}  OK")
        results[tid] = {
            'coords': coords.tolist(),
            'plddt':  plddt,
            'method': 'rhofold',
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{elapsed:>5.1f}s  {'':>6}  {'':>6}  FAILED: {e}")

print(f"\nSuccessful: {len(results)}/{len(stub_targets)} targets")

# Save
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUT_JSON}")
print(f"\nNext step: upload {OUT_JSON} to Kaggle dataset 'rna-templates'")
print("  Go to: https://www.kaggle.com/datasets/ilanmeyrowitsch/rna-templates")
print("  Add rhofold_predictions.json as a new file in the dataset")
