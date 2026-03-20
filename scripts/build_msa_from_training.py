"""
scripts/build_msa_from_training.py — Build MSA files for stub targets.

Searches 204K training sequences for similar sequences to use as MSA
for RhoFold inference. Uses k-mer similarity (no external tools needed).

Output: data/msa/{target_id}.fasta  — MSA in FASTA format

Usage:
    cd ~/kaggle/rna_kaggle
    python3 scripts/build_msa_from_training.py
"""

import sys, os, json, time
import numpy as np
import pandas as pd

DATA_DIR = '/home/ilan/kaggle/data'
OUT_DIR  = 'data/msa'
K        = 6       # k-mer length
TOP_N    = 64      # max MSA depth (RhoFold uses up to 128)
MIN_SIM  = 0.05    # minimum k-mer Jaccard similarity

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load training sequences ───────────────────────────────────────────────────
print("Loading training sequences...")
train = pd.read_csv(f'{DATA_DIR}/train_sequences.csv')
train['sequence'] = train['sequence'].str.upper().str.replace('T','U')

# Extract all sequences from the all_sequences column (102K sequences)
all_seqs = []
for _, row in train.iterrows():
    # Add primary sequence
    all_seqs.append(row['sequence'])
    # Parse all_sequences FASTA field
    fasta_str = str(row.get('all_sequences', ''))
    if fasta_str and fasta_str != 'nan':
        seq = ''
        for line in fasta_str.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if seq: all_seqs.append(seq.upper().replace('T','U'))
                seq = ''
            else:
                seq += line
        if seq: all_seqs.append(seq.upper().replace('T','U'))

# Deduplicate
all_seqs = list(set(s for s in all_seqs if len(s) >= 10))
print(f"  {len(all_seqs)} unique sequences loaded")

# ── Load stub target sequences ────────────────────────────────────────────────
test = pd.read_csv(f'{DATA_DIR}/test_sequences.csv')
test['sequence'] = test['sequence'].str.upper().str.replace('T','U')

with open('data/pdb_cache/template_predictions.json') as f:
    templates = json.load(f)
with open('data/pdb_cache/rhofold_predictions.json') as f:
    rhofold = json.load(f)

# All non-TBM targets (use RhoFold)
stub_targets = test[~test['target_id'].isin(templates)].copy()
print(f"Targets needing MSA: {len(stub_targets)}")

def kmers(seq, k=K):
    return set(seq[i:i+k] for i in range(len(seq)-k+1))

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0

def build_msa(query_seq, train_seqs, top_n=TOP_N, min_sim=MIN_SIM):
    """Find top_n most similar sequences to query from training set."""
    q_kmers = kmers(query_seq)
    scores  = []
    for seq in train_seqs:
        if abs(len(seq) - len(query_seq)) > len(query_seq) * 0.5:
            continue  # skip very different lengths
        t_kmers = kmers(seq)
        sim     = jaccard(q_kmers, t_kmers)
        if sim >= min_sim:
            scores.append((sim, seq))
    scores.sort(reverse=True)
    # Return top_n unique sequences (excluding exact duplicates of query)
    seen = {query_seq}
    result = []
    for sim, seq in scores[:top_n*2]:
        if seq not in seen:
            result.append((sim, seq))
            seen.add(seq)
        if len(result) >= top_n:
            break
    return result

# Pre-build k-mer sets for training sequences (fast lookup)
print("Pre-computing training k-mers...")
t0 = time.time()
train_seqs = all_seqs
print(f"  {len(train_seqs)} sequences ready in {time.time()-t0:.1f}s")

# ── Build MSA for each stub target ───────────────────────────────────────────
print(f"\n{'Target':<12} {'Len':>5}  {'MSA_hits':>9}  {'Top_sim':>8}  Status")
print('-' * 50)

for _, row in stub_targets.iterrows():
    tid      = row['target_id']
    query    = row['sequence']
    q_len    = len(query)
    out_path = f'{OUT_DIR}/{tid}.fasta'

    if q_len > 600:
        print(f"  {tid:<10} {q_len:>5}  {'':>9}  {'':>8}  SKIP (too long)")
        continue

    t0   = time.time()
    hits = build_msa(query, train_seqs)
    elapsed = time.time() - t0

    top_sim = hits[0][0] if hits else 0.0

    # Write MSA FASTA: query first, then hits
    with open(out_path, 'w') as f:
        f.write(f'>{tid}_query\n{query}\n')
        for i, (sim, seq) in enumerate(hits):
            f.write(f'>hit_{i+1}_sim{sim:.3f}\n{seq}\n')

    total = 1 + len(hits)
    print(f"  {tid:<10} {q_len:>5}  {total:>9}  {top_sim:>8.3f}  OK ({elapsed:.1f}s) → {out_path}")

print(f"\nMSA files written to {OUT_DIR}/")
print("\nNext: rerun build_rhofold_cache.py with MSA paths")
