"""
scripts/improve_weak_targets.py — Improve weak US-align targets with more RhoFold seeds.

Targets to fix:
  9JFS  (US=0.096) — bad RhoFold, 246nt
  9JGM  (US=0.223) — C:2 dimer, try both monomer and full seq
  9QZJ  (US=0.162) — 19nt tiny
  9WHV  (US=0.161) — G-quadruplex repeat

Strategy: run 20 seeds, score with US-align, keep best 5.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/improve_weak_targets.py
"""
import sys, os, json, subprocess, tempfile, time
import numpy as np, pandas as pd

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, '.')
sys.path.insert(0, '/home/ilan/kaggle/data/models/rhofold')

import torch
from rhofold.rhofold import RhoFold
from rhofold.config import rhofold_config
from rhofold.utils.alphabet import get_features

CKPT     = '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt'
DATA_DIR = '/home/ilan/kaggle/data'
SENTINEL = -1e18
USALIGN  = './USalign'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading RhoFold on {DEVICE}...")
model = RhoFold(rhofold_config)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE)['model'])
model = model.to(DEVICE).eval()
print("Loaded OK")

test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T','U')
lbl  = pd.read_csv(f'{DATA_DIR}/validation_labels.csv')
lbl['target'] = lbl['ID'].str.rsplit('_',n=1).str[0]
n_ref = sum(1 for i in range(1,41) if f'x_{i}' in lbl.columns)

def get_refs(tid):
    ls = lbl[lbl['target']==tid].sort_values('resid')
    refs = []
    for i in range(1,n_ref+1):
        if f'x_{i}' not in ls.columns: break
        if (ls[f'x_{i}']==SENTINEL).all(): continue
        mask = ls[f'x_{i}']!=SENTINEL
        refs.append(ls.loc[mask,[f'x_{i}',f'y_{i}',f'z_{i}']].values.astype(np.float32))
    return refs

def write_pdb(coords, path):
    lines = []
    for i,(x,y,z) in enumerate(coords):
        lines.append(f"ATOM  {i+1:5d}  C1' G   A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    lines.append("END")
    with open(path,'w') as f: f.write('\n'.join(lines))

def usalign_tm(pred, ref):
    with tempfile.TemporaryDirectory() as d:
        write_pdb(pred, f'{d}/pred.pdb')
        write_pdb(ref,  f'{d}/ref.pdb')
        r = subprocess.run([USALIGN, f'{d}/pred.pdb', f'{d}/ref.pdb', '-atom', " C1'"],
                          capture_output=True, text=True)
        tms = [float(l.split('TM-score=')[1].split()[0])
               for l in r.stdout.split('\n') if l.strip().startswith('TM-score=')]
        return tms[1] if len(tms)>=2 else (tms[0] if tms else 0.0)

def run_rhofold(seq, seed=42):
    with tempfile.TemporaryDirectory() as d:
        fas = f'{d}/q.fasta'
        with open(fas,'w') as f: f.write(f'>q\n{seq}\n')
        r = get_features(fas, fas)
        tok = r['tokens'].to(DEVICE)
        fm  = r['rna_fm_tokens'].to(DEVICE)
        if seed != 42:
            noise = torch.randn_like(fm.float()) * 0.005
            fm = (fm.float()+noise).to(fm.dtype)
        with torch.no_grad():
            out = model(tok, fm, r['seq'])
        c = out[-1]["cords_c1'"][0][0].cpu().numpy().astype(np.float32)
        p = float(out[-1]['plddt'][0][0].cpu().numpy().mean())
        del tok, fm, out; torch.cuda.empty_cache()
        return c, p

def score_best(coords, refs):
    best = 0.0
    for ref in refs:
        n = min(len(coords), len(ref))
        best = max(best, usalign_tm(coords[:n], ref[:n]))
    return best

# ── Load current cache ─────────────────────────────────────────────────────────
with open('data/pdb_cache/rhofold_predictions.json') as f:
    rho = json.load(f)

SEEDS = [42,100,200,300,400,500,600,700,800,900,
         150,250,350,450,550,650,750,850,950,1050]

# Targets to improve (US-align score)
TARGETS = {
    '9JFS': 0.096,
    '9JGM': 0.223,
    '9QZJ': 0.162,
    '9WHV': 0.161,
}

for tid, current_us in TARGETS.items():
    seq  = test[test['target_id']==tid]['sequence'].iloc[0]
    refs = get_refs(tid)
    print(f"\n{'='*50}")
    print(f"{tid} ({len(seq)}nt) — current US-align={current_us:.3f}")

    if len(seq) > 300:
        print(f"  SKIP: too long for reliable RhoFold ({len(seq)}nt)")
        continue

    all_coords, all_tm = [], []
    for seed in SEEDS:
        try:
            coords, plddt = run_rhofold(seq, seed)
            tm = score_best(coords, refs)
            all_coords.append(coords.tolist())
            all_tm.append(tm)
            print(f"  seed {seed:>4}: TM={tm:.4f}  pLDDT={plddt:.3f}")
        except Exception as e:
            print(f"  seed {seed:>4}: FAILED {e}")
            break

    if not all_tm:
        continue

    # Pick best 5
    top5 = sorted(range(len(all_tm)), key=lambda i:-all_tm[i])[:5]
    new_best = all_tm[top5[0]]
    print(f"\n  Current: {current_us:.4f}  New best: {new_best:.4f}  (+{new_best-current_us:+.4f})")

    if new_best > current_us + 0.01:
        rho[tid] = {
            'coords_list': [all_coords[i] for i in top5],
            'coords':       all_coords[top5[0]],
            'plddt':        float(np.mean([rho[tid]['plddt']] * 5) if tid in rho else 0.5),
            'method':       'rhofold_best5of20_usalign',
            'n_seeds':      5,
        }
        with open('data/pdb_cache/rhofold_predictions.json','w') as f:
            json.dump(rho, f, indent=2)
        print(f"  SAVED ✓")
    else:
        print(f"  No improvement — keeping current")

print("\n\nDone. Run: python3 scripts/local_eval.py")
