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

# frames shape: (8, 1, 73, 7) — 7 = 4 (quaternion) + 3 (translation)
# C1' is the ORIGIN of each backbone frame, so frame translation = C1' position
frames = last['frames']
print(f"frames shape: {frames.shape}")
for i in range(frames.shape[0]):
    trans = frames[i, 0, :, 4:].cpu().numpy()   # (73, 3) translation
    nz = np.sum(np.any(np.abs(trans) > 0.01, axis=1))
    print(f"  Frame {i}: range=[{trans.min():.2f}, {trans.max():.2f}]  nonzero_res={nz}  res0={trans[0].round(3)}")

# Try converter
print("\nConverter approach:")
try:
    converter = p._model.structure_module.converter
    print(f"  type: {type(converter).__name__}")
    methods = [m for m in dir(converter) if not m.startswith('_')]
    print(f"  methods: {methods}")

    # Try build_cords or similar
    for method_name in methods:
        print(f"  {method_name}: {getattr(converter, method_name)}")
except Exception as e:
    print(f"  error: {e}")

# Try using structure_module directly with get_c1_coords
print("\nStructure module methods:")
sm = p._model.structure_module
sm_methods = [m for m in dir(sm) if not m.startswith('_') and 'cord' in m.lower()]
print(f"  {sm_methods}")
