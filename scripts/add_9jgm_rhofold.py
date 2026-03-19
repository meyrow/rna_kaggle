"""
scripts/add_9jgm_rhofold.py — Add 9JGM to RhoFold cache (better than TBM).

RhoFold scores 0.145 vs TBM 0.104 for 9JGM.
Also removes 9JGM from template_predictions.json so RhoFold is used.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/add_9jgm_rhofold.py
"""

import sys, os, json, tempfile, time
import numpy as np
import pandas as pd

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

RHOFOLD_REPO = '/home/ilan/kaggle/data/models/rhofold'
CKPT         = '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt'
DATA_DIR     = '/home/ilan/kaggle/data'
TEMPLATE_JSON = 'data/pdb_cache/template_predictions.json'
RHOFOLD_JSON  = 'data/pdb_cache/rhofold_predictions.json'
SEEDS = [42, 123, 456, 789, 1337]

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

# Get 9JGM sequence
test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T', 'U')
seq = test[test['target_id'] == '9JGM']['sequence'].iloc[0]
print(f"\n9JGM: {len(seq)}nt")

# Run 5 seeds
coords_list, plddts = [], []
print(f"Running {len(SEEDS)} seeds...")
t0 = time.time()

for seed in SEEDS:
    with tempfile.TemporaryDirectory() as d:
        fas = f'{d}/q.fasta'
        with open(fas, 'w') as f:
            f.write(f'>q\n{seq}\n')
        r = get_features(fas, fas)
        tokens = r['tokens'].to(DEVICE)
        rna_fm = r['rna_fm_tokens'].to(DEVICE)
        if seed != 42:
            noise = torch.randn_like(rna_fm.float()) * 0.01 * (seed / 42)
            rna_fm = (rna_fm.float() + noise).to(rna_fm.dtype)
        with torch.no_grad():
            out = model(tokens, rna_fm, r['seq'])
        coords = out[-1]["cords_c1'"][0][0].cpu().numpy().astype(np.float32)
        plddt  = float(out[-1]['plddt'][0][0].cpu().numpy().mean())
        coords_list.append(coords.tolist())
        plddts.append(plddt)
        del tokens, rna_fm, out
        torch.cuda.empty_cache()
    print(f"  seed {seed}: pLDDT={plddt:.3f}  ({time.time()-t0:.1f}s)")

mean_plddt = float(np.mean(plddts))
print(f"\nMean pLDDT: {mean_plddt:.3f}")

# Add to rhofold_predictions.json
with open(RHOFOLD_JSON) as f:
    rho = json.load(f)

rho['9JGM'] = {
    'coords_list': coords_list,
    'coords':      coords_list[0],
    'plddt':       mean_plddt,
    'plddt_list':  plddts,
    'method':      'rhofold_5seeds',
    'n_seeds':     5,
}
with open(RHOFOLD_JSON, 'w') as f:
    json.dump(rho, f, indent=2)
print(f"Saved {RHOFOLD_JSON} — {len(rho)} predictions total")

# Remove 9JGM from template_predictions.json (RhoFold is better)
with open(TEMPLATE_JSON) as f:
    templates = json.load(f)

if '9JGM' in templates:
    del templates['9JGM']
    with open(TEMPLATE_JSON, 'w') as f:
        json.dump(templates, f, indent=2)
    print(f"Removed 9JGM from TBM templates. Remaining: {len(templates)}")

print("\nNext steps:")
print("  python3 scripts/local_eval.py")
print("  cp data/pdb_cache/rhofold_predictions.json ~/kaggle/rna-templates/")
print("  cp data/pdb_cache/template_predictions.json ~/kaggle/rna-templates/")
print("  cd ~/kaggle/rna-templates && kaggle datasets version -m 'v6: 9JGM via RhoFold'")
print("  cd ~/kaggle/rna_kaggle && kaggle kernels push -p .")
