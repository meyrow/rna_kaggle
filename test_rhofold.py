import sys, torch
sys.path.insert(0, '/home/ilan/kaggle/data/external/RhoFold')
from src.rhofold_predictor import RhoFoldPredictor
import numpy as np

p = RhoFoldPredictor({
    'rhofold_checkpoint': '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt',
    'rhofold_repo': '/home/ilan/kaggle/data/external/RhoFold'
})

seq = 'GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA'
seq_int = [{'A':0,'C':1,'G':2,'U':3}.get(c,0) for c in seq]
nuc_fm = {'A':5,'C':6,'G':7,'U':8,'T':8,'N':5}
raw_ids = [nuc_fm.get(c,5) for c in seq]
raw_tok = torch.tensor([raw_ids], dtype=torch.long, device=p._device)
tokens = raw_tok.unsqueeze(1)

with torch.inference_mode():
    output = p._model(tokens=tokens, rna_fm_tokens=raw_tok, seq=seq_int)

last = output[-1]

# Check cords_c1' key
key = "cords_c1'"
c1_raw = last[key]
if isinstance(c1_raw, list):
    c1_raw = c1_raw[-1]
print(f"cords_c1' shape : {c1_raw.shape}")
print(f"cords_c1' min   : {c1_raw.min():.3f}  max: {c1_raw.max():.3f}  nonzero: {c1_raw.abs().sum():.1f}")

# Inspect cord_tns_pred — shape (1, 1679, 3) = 73 residues * N atoms
ct = last['cord_tns_pred']
if isinstance(ct, list):
    ct = ct[-1]
ct = ct.squeeze(0)   # (1679, 3)
print(f"\ncord_tns_pred shape: {ct.shape}")
n_res = len(seq)
n_atoms = ct.shape[0] // n_res
c = ct.reshape(n_res, n_atoms, 3).cpu().numpy()
print(f"Reshaped to ({n_res}, {n_atoms}, 3)")
print("\nNon-zero atom indices for residue 0:")
for i in range(n_atoms):
    v = c[0, i]
    if not np.allclose(v, 0, atol=1e-3):
        print(f"  atom[{i:2d}]: {v.round(3)}")

# Try to get atom names from constants
print("\nChecking constants for atom order:")
try:
    from rhofold.utils import constants
    attrs = [a for a in dir(constants) if 'ATOM' in a.upper() or 'C1' in a]
    print("Relevant constants:", attrs)
    for a in attrs:
        print(f"  {a} = {getattr(constants, a)}")
except Exception as e:
    print(f"  constants error: {e}")
