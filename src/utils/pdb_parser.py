"""utils/pdb_parser.py — PDB/mmCIF C1' coordinate extraction."""

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

RNA_RESNAMES = {"A", "C", "G", "U", "ADE", "CYT", "GUA", "URA", "DA", "DC", "DG", "DT"}


def extract_c1_coords(pdb_path: str, chain_id: str = None) -> tuple[np.ndarray, str]:
    """
    Extract C1' atom coordinates from a PDB file.

    Args:
        pdb_path: path to .pdb or .cif file
        chain_id: if given, extract only this chain

    Returns:
        (coords, sequence) where coords is shape (L, 3) and sequence is the RNA string
    """
    coords = []
    sequence = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            # resid = int(line[22:26].strip())

            if chain_id and chain != chain_id:
                continue
            if resname not in RNA_RESNAMES:
                continue
            if atom_name != "C1'":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

                # Map residue name to single letter
                res1 = _resname_to_1letter(resname)
                sequence.append(res1)
            except ValueError:
                continue

    if not coords:
        return np.zeros((0, 3)), ""

    return np.array(coords, dtype=np.float32), "".join(sequence)


def _resname_to_1letter(resname: str) -> str:
    mapping = {
        "A": "A", "ADE": "A",
        "C": "C", "CYT": "C",
        "G": "G", "GUA": "G",
        "U": "U", "URA": "U",
        "DA": "A", "DC": "C", "DG": "G", "DT": "U",
    }
    return mapping.get(resname.upper(), "N")


def build_pdb_c1_cache(pdb_dir: str, output_pkl: str):
    """
    Build a local cache of C1' coordinates for all RNA PDB structures.
    Run once via scripts/build_pdb_cache.sh.

    Saves a dict: { "PDBID_CHAIN": np.ndarray(L, 3) }
    """
    import pickle
    pdb_dir = Path(pdb_dir)
    cache = {}
    pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.ent"))

    logger.info(f"Building C1' cache from {len(pdb_files)} PDB files in {pdb_dir}")

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper().replace("PDB", "")
        coords, seq = extract_c1_coords(str(pdb_path))
        if len(coords) > 0:
            # Try to get chain from filename or default to A
            chain = "A"
            key = f"{pdb_id}_{chain}"
            cache[key] = {"coords": coords, "sequence": seq}

    with open(output_pkl, "wb") as f:
        pickle.dump(cache, f)

    logger.info(f"Cache built: {len(cache)} entries → {output_pkl}")
