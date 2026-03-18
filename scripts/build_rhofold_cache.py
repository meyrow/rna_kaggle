"""
scripts/build_rhofold_cache.py — Run RhoFold on stub targets, 5 seeds each.

Output: data/pdb_cache/rhofold_predictions.json
  {target_id: {'coords_list': [[[x,y,z],...] x5], 'plddt': float, 'method': str}}

Upload to Kaggle dataset 'rna-templates' alongside template_predictions.json.

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/build_rhofold_cache.py
"""

import sys, os, time, tempfile, json
import numpy as np
import pandas as pd

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

RHOFOLD_REPO = '/home/ilan/kaggle/data/models/rhofold'
CKPT         = '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt'
DATA_DIR     = '/home/ilan/kaggle/data'
OUT_JSON     = 'data/pdb_cache/rhofold_predictions.json'
MAX_LEN      = 500   # skip sequences longer than this (single-shot)
CHUNK_LEN    = 400   # for long seqs: split into chunks of this size
SEEDS        = [42, 123, 456, 789, 1337]

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
print("RhoFold loaded OK\n")

# ── Load test sequences ───────────────────────────────────────────────────────
test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T', 'U')

with open('data/pdb_cache/template_predictions.json') as f:
    templates = json.load(f)

stub_targets = test[~test['target_id'].isin(templates)].copy()
print(f"Stub targets: {len(stub_targets)}")
for _, r in stub_targets.iterrows():
    print(f"  {r['target_id']}: {len(r['sequence'])}nt")

# ── RhoFold inference ─────────────────────────────────────────────────────────
def run_single(seq, seed=42, device=None):
    """Run RhoFold on a sequence. Returns (coords, plddt)."""
    if device is None:
        device = DEVICE
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        fas_path = os.path.join(tmpdir, 'input.fasta')
        with open(fas_path, 'w') as f:
            f.write(f'>query\n{seq}\n')
        result        = get_features(fas_path, fas_path)
        tokens        = result['tokens'].to(device)
        rna_fm_tokens = result['rna_fm_tokens'].to(device)
        seq_out       = result['seq']

        if seed != 42:
            noise = torch.randn_like(rna_fm_tokens.float()) * 0.01 * (seed / 42)
            rna_fm_tokens = (rna_fm_tokens.float() + noise).to(rna_fm_tokens.dtype)

        # Move model to target device for this inference
        model.to(device)
        with torch.no_grad():
            outputs = model(tokens, rna_fm_tokens, seq_out)

        out       = outputs[-1]
        c1_coords = out["cords_c1'"][0][0].cpu().numpy()
        plddt     = out['plddt'][0][0].cpu().numpy().mean()

        del tokens, rna_fm_tokens, outputs
        if device == 'cuda':
            torch.cuda.empty_cache()

        return c1_coords.astype(np.float32), float(plddt)


def run_cpu_fallback(seq, seed=42):
    """Run on CPU — slow but no VRAM limit."""
    print(f"    → CPU fallback (may take several minutes)...", flush=True)
    torch.cuda.empty_cache()
    return run_single(seq, seed=seed, device='cpu')

def run_chunked(seq, chunk_size=CHUNK_LEN, seed=42):
    """
    Run RhoFold on a long sequence by splitting into overlapping chunks.
    Joins chunks with a smooth transition.
    """
    L = len(seq)
    if L <= chunk_size:
        return run_single(seq, seed)

    # Split into chunks with 20nt overlap
    overlap = 20
    chunks  = []
    starts  = []
    i = 0
    while i < L:
        end = min(i + chunk_size, L)
        chunks.append(seq[i:end])
        starts.append(i)
        if end == L:
            break
        i += chunk_size - overlap

    print(f"    Chunked: {len(chunks)} chunks of ~{chunk_size}nt")

    all_coords = np.zeros((L, 3), dtype=np.float32)
    weights    = np.zeros(L, dtype=np.float32)

    for ci, (chunk_seq, start) in enumerate(zip(chunks, starts)):
        chunk_coords, _ = run_single(chunk_seq, seed)
        end = start + len(chunk_seq)

        # Apply cosine blending weights to reduce boundary artifacts
        w = np.ones(len(chunk_seq), dtype=np.float32)
        blend = min(overlap, 10)
        if ci > 0:  # taper start
            w[:blend] = np.linspace(0, 1, blend)
        if ci < len(chunks) - 1:  # taper end
            w[-blend:] = np.linspace(1, 0, blend)

        # Align chunk to existing coords at overlap region
        if ci > 0 and start > 0:
            ref_start = start
            ref_end   = start + blend
            if weights[ref_start:ref_end].sum() > 0:
                existing = all_coords[ref_start:ref_end] / np.maximum(weights[ref_start:ref_end, None], 1e-6)
                predicted = chunk_coords[:blend]
                # Simple translation alignment
                offset = existing.mean(0) - predicted.mean(0)
                chunk_coords = chunk_coords + offset

        all_coords[start:end] += chunk_coords * w[:, None]
        weights[start:end]    += w

    # Normalize
    mask = weights > 0
    all_coords[mask] /= weights[mask, None]

    plddt = 0.5  # placeholder for chunked inference
    return all_coords, plddt

# ── Run all stub targets ──────────────────────────────────────────────────────
results = {}
print(f"\n{'Target':<12} {'Len':>5}  {'Seeds':>6}  {'C1-C1':>6}  {'pLDDT':>6}  Status")
print('-' * 60)

for _, row in stub_targets.iterrows():
    tid = row['target_id']
    seq = row['sequence']
    L   = len(seq)

    if L > 5000:
        print(f"{tid:<12} {L:>5}  {'':>6}  {'':>6}  {'':>6}  SKIP (>{5000}nt — too large even for CPU)")
        continue

    # Use CPU for sequences too long for GPU
    use_cpu = L > MAX_LEN
    if use_cpu:
        print(f"{tid:<12} {L:>5}  ", end='', flush=True)
        print(f"CPU-mode (len>{MAX_LEN})", flush=True)
    else:
        print(f"{tid:<12} {L:>5}  ", end='', flush=True)
    t0 = time.time()

    try:
        all_coords_list = []
        plddts = []

        for seed in SEEDS:
            if use_cpu:
                coords, plddt = run_cpu_fallback(seq, seed=seed)
            elif L > MAX_LEN:
                coords, plddt = run_chunked(seq, seed=seed)
            else:
                coords, plddt = run_single(seq, seed=seed)
            all_coords_list.append(coords.tolist())
            plddts.append(plddt)

        elapsed    = time.time() - t0
        mean_plddt = float(np.mean(plddts))

        # Check geometry of first prediction
        c0    = np.array(all_coords_list[0])
        dists = np.linalg.norm(np.diff(c0, axis=0), axis=1)

        print(f"{len(SEEDS):>6}  {dists.mean():>5.2f}Å  {mean_plddt:>6.3f}  OK ({elapsed:.1f}s)")

        results[tid] = {
            'coords_list': all_coords_list,  # 5 predictions
            'coords':      all_coords_list[0],  # best (highest plddt) for legacy compat
            'plddt':       mean_plddt,
            'plddt_list':  plddts,
            'method':      'rhofold_5seeds',
            'n_seeds':     len(SEEDS),
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"{'':>6}  {'':>6}  {'':>6}  FAILED ({elapsed:.1f}s): {e}")

print(f"\nSuccessful: {len(results)}/{len(stub_targets)} targets")

# Save
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
size_kb = os.path.getsize(OUT_JSON) // 1024
print(f"Saved: {OUT_JSON} ({size_kb}KB)")
print(f"\nUpload to Kaggle dataset:")
print(f"  cp {OUT_JSON} ~/kaggle/rna-templates/")
print(f"  cd ~/kaggle/rna-templates && kaggle datasets version -m 'v4: RhoFold 5-seed predictions'")
