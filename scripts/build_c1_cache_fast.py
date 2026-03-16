"""
scripts/build_c1_cache_fast.py — Build C1' coordinate cache from CIF files.

Reads all .cif files in PDB_RNA/ and extracts C1' atom coordinates.
Creates two outputs:
  1. data/pdb_cache/pdb_c1_coords.pkl  — C1' coords dict {pdbid_chain: ndarray}
  2. data/pdb_cache/pdb_rna_seqs.fa   — FASTA for MMseqs2 template search

Usage:
    cd ~/rna_kaggle
    python3 scripts/build_c1_cache_fast.py

Runtime: ~20-30 min for 9566 CIF files
"""

import os, sys, pickle, re
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')

PDB_DIR  = Path('/home/ilan/kaggle/data/PDB_RNA')
OUT_DIR  = Path('data/pdb_cache')
PKL_OUT  = OUT_DIR / 'pdb_c1_coords.pkl'
FASTA_OUT = OUT_DIR / 'pdb_rna_seqs.fa'

OUT_DIR.mkdir(parents=True, exist_ok=True)

NUC_MAP = {
    'A': 'A', 'ADE': 'A', 'A23': 'A', 'ATP': 'A', '2MA': 'A', '1MA': 'A',
    'C': 'C', 'CYT': 'C', 'CBR': 'C', '5MC': 'C', 'OMC': 'C',
    'G': 'G', 'GUA': 'G', 'GTP': 'G', '7MG': 'G', 'OMG': 'G', '2MG': 'G',
    'U': 'U', 'URA': 'U', 'URI': 'U', 'H2U': 'U', 'PSU': 'U', '5MU': 'U',
    'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'U', 'DU': 'U',
}

def parse_cif_c1(cif_path):
    """
    Extract C1' coords from mmCIF file. Returns dict {chain_id: (seq, coords)}.
    Handles both PDB-format ATOM lines in CIF and loop_ ATOM_SITE tables.
    """
    chains = {}

    with open(cif_path, errors='ignore') as f:
        content = f.read()

    # CIF files have ATOM records in loop_ _atom_site tables
    # Parse the column headers first
    col_map = {}
    in_atom_loop = False
    col_idx = 0
    lines = content.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect start of atom site loop
        if line == 'loop_':
            # Check if next lines define _atom_site columns
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
                cols.append(lines[j].strip())
                j += 1
            if cols:
                col_map = {c: idx for idx, c in enumerate(cols)}
                in_atom_loop = True
                i = j
                continue
            else:
                in_atom_loop = False

        if in_atom_loop and line and not line.startswith('_') and not line.startswith('#'):
            # Parse atom record
            parts = line.split()
            try:
                group     = parts[col_map.get('_atom_site.group_PDB', 0)]
                atom_name = parts[col_map.get('_atom_site.label_atom_id', 3)].strip('"')
                res_name  = parts[col_map.get('_atom_site.label_comp_id', 5)].upper()
                chain_id  = parts[col_map.get('_atom_site.label_asym_id',
                            col_map.get('_atom_site.auth_asym_id', 6))]
                res_seq   = parts[col_map.get('_atom_site.label_seq_id',
                            col_map.get('_atom_site.auth_seq_id', 8))]
                x = float(parts[col_map.get('_atom_site.Cartn_x', 10)])
                y = float(parts[col_map.get('_atom_site.Cartn_y', 11)])
                z = float(parts[col_map.get('_atom_site.Cartn_z', 12)])

                if group == 'ATOM' and atom_name == "C1'" and res_name in NUC_MAP:
                    nuc = NUC_MAP[res_name]
                    key = (chain_id, res_seq)
                    if chain_id not in chains:
                        chains[chain_id] = {}
                    chains[chain_id][key] = (nuc, [x, y, z])
            except (IndexError, ValueError, KeyError):
                pass

        elif in_atom_loop and (line.startswith('#') or line.startswith('_') or line == 'loop_'):
            in_atom_loop = False

        i += 1

    # Convert to arrays
    result = {}
    for chain_id, residues in chains.items():
        if len(residues) < 5:  # skip very short fragments
            continue
        # Sort by res_seq
        try:
            sorted_res = sorted(residues.items(), key=lambda x: int(x[0][1]))
        except ValueError:
            sorted_res = sorted(residues.items())
        seq    = ''.join(v[0] for _, v in sorted_res)
        coords = np.array([v[1] for _, v in sorted_res], dtype=np.float32)
        result[chain_id] = (seq, coords)
    return result


def main():
    cif_files = sorted(PDB_DIR.glob('*.cif'))
    print(f'Found {len(cif_files)} CIF files in {PDB_DIR}')
    print(f'Output: {PKL_OUT}')
    print()

    cache = {}    # {pdbid_chain: ndarray}
    fasta = []    # for MMseqs2
    n_ok = 0
    n_fail = 0

    for idx, cif_path in enumerate(cif_files):
        pdb_id = cif_path.stem.upper()
        try:
            chains = parse_cif_c1(cif_path)
            for chain_id, (seq, coords) in chains.items():
                key = f"{pdb_id}_{chain_id}"
                cache[key] = coords
                if len(seq) >= 10:
                    fasta.append(f">{key}\n{seq}")
            n_ok += 1
        except Exception as e:
            n_fail += 1

        if (idx + 1) % 500 == 0:
            pct = (idx + 1) / len(cif_files) * 100
            print(f'  [{idx+1}/{len(cif_files)}] {pct:.0f}%  '
                  f'chains={len(cache)}  failed={n_fail}')

    print(f'\nDone: {n_ok} parsed, {n_fail} failed')
    print(f'Total chains: {len(cache)}')

    # Save pickle
    with open(PKL_OUT, 'wb') as f:
        pickle.dump(cache, f, protocol=4)
    print(f'Saved: {PKL_OUT}  ({PKL_OUT.stat().st_size//1024//1024} MB)')

    # Save FASTA
    with open(FASTA_OUT, 'w') as f:
        f.write('\n'.join(fasta))
    print(f'Saved: {FASTA_OUT}  ({len(fasta)} sequences)')


if __name__ == '__main__':
    main()
