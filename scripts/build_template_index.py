"""
scripts/build_template_index.py — Build template search index from PDB C1' cache.

Uses simple k-mer hashing for fast sequence similarity search.
No MMseqs2 required. Runs in ~2-5 min after build_c1_cache_fast.py.

Creates: data/pdb_cache/template_index.pkl
  {kmer: [list of chain_ids]}

Usage:
    cd ~/rna_kaggle
    python3 scripts/build_template_index.py
"""

import os, sys, pickle
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '.')

PKL_IN   = Path('data/pdb_cache/pdb_c1_coords.pkl')
FASTA_IN = Path('data/pdb_cache/pdb_rna_seqs.fa')
IDX_OUT  = Path('data/pdb_cache/template_index.pkl')

KMER_LEN = 6   # 6-mer hashing — good balance speed vs specificity

def read_fasta(path):
    seqs = {}
    key = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:]
            elif key:
                seqs[key] = seqs.get(key, '') + line
    return seqs

def build_kmer_index(seqs, k=KMER_LEN):
    """Build inverted index: kmer -> list of chain_ids."""
    index = defaultdict(list)
    for chain_id, seq in seqs.items():
        kmers_seen = set()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer not in kmers_seen:
                index[kmer].append(chain_id)
                kmers_seen.add(kmer)
    return dict(index)

def search_templates(query_seq, index, seqs, coords_cache, top_k=5, min_id=0.3):
    """
    Find top-k template chains for a query sequence using k-mer voting.
    
    Returns list of (chain_id, seq_identity, c1_coords) sorted by identity.
    """
    k = KMER_LEN
    # Count k-mer hits per chain
    votes = defaultdict(int)
    query_kmers = set(query_seq[i:i+k] for i in range(len(query_seq)-k+1))
    for kmer in query_kmers:
        for chain_id in index.get(kmer, []):
            votes[chain_id] += 1

    if not votes:
        return []

    # Score by Jaccard-like similarity
    results = []
    for chain_id, hits in sorted(votes.items(), key=lambda x: -x[1])[:50]:
        ref_seq = seqs.get(chain_id, '')
        if not ref_seq or chain_id not in coords_cache:
            continue

        # Quick sequence identity estimate using shared kmers
        ref_kmers = set(ref_seq[i:i+k] for i in range(len(ref_seq)-k+1))
        shared = len(query_kmers & ref_kmers)
        union  = len(query_kmers | ref_kmers)
        jaccard = shared / union if union > 0 else 0

        # Length ratio penalty
        len_ratio = min(len(query_seq), len(ref_seq)) / max(len(query_seq), len(ref_seq))
        score = jaccard * len_ratio

        if score >= min_id * 0.3:  # loose threshold for k-mer phase
            results.append((chain_id, score, coords_cache[chain_id]))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def main():
    print(f'Loading sequences from {FASTA_IN}...')
    seqs = read_fasta(FASTA_IN)
    print(f'  {len(seqs)} sequences loaded')

    print(f'Loading coords cache from {PKL_IN}...')
    with open(PKL_IN, 'rb') as f:
        coords = pickle.load(f)
    print(f'  {len(coords)} chains loaded')

    print(f'Building {KMER_LEN}-mer index...')
    index = build_kmer_index(seqs, k=KMER_LEN)
    print(f'  {len(index)} unique {KMER_LEN}-mers')

    # Save everything together
    bundle = {
        'index': index,
        'seqs': seqs,
        'kmer_len': KMER_LEN,
    }
    with open(IDX_OUT, 'wb') as f:
        pickle.dump(bundle, f, protocol=4)
    size_mb = IDX_OUT.stat().st_size // 1024 // 1024
    print(f'Saved: {IDX_OUT}  ({size_mb} MB)')

    # Quick test: search for tRNA-Phe
    print('\nTest search: tRNA-Phe (73nt)')
    test_seq = 'GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA'
    hits = search_templates(test_seq, index, seqs, coords, top_k=5)
    for chain_id, score, c in hits:
        print(f'  {chain_id}: score={score:.3f}  len={len(c)}  coords[0]={c[0].round(2)}')


if __name__ == '__main__':
    main()
