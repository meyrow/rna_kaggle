"""utils/sequence_utils.py — RNA sequence utilities."""

import re

VALID_RNA_CHARS = set("ACGUacguNn")
IUPAC_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "U",
    "R": "AG", "Y": "CU", "S": "GC", "W": "AU",
    "K": "GU", "M": "AC", "B": "CGU", "D": "AGU",
    "H": "ACU", "V": "ACG", "N": "ACGU",
}


def validate_rna_sequence(sequence: str) -> bool:
    """Return True if sequence contains only valid RNA/IUPAC characters."""
    return bool(sequence) and all(c.upper() in IUPAC_MAP for c in sequence)


def normalize_sequence(sequence: str) -> str:
    """Uppercase and replace T→U."""
    return sequence.upper().replace("T", "U")


def gc_content(sequence: str) -> float:
    """Return GC fraction of the sequence."""
    seq = normalize_sequence(sequence)
    gc = sum(1 for c in seq if c in ("G", "C"))
    return gc / len(seq) if seq else 0.0


def split_into_chunks(sequence: str, chunk_size: int, overlap: int = 50) -> list[tuple[int, int]]:
    """
    Split a long sequence into overlapping chunks.
    Returns list of (start, end) tuples (0-indexed, end exclusive).
    """
    n = len(sequence)
    if n <= chunk_size:
        return [(0, n)]
    chunks = []
    pos = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        chunks.append((pos, end))
        if end == n:
            break
        pos += chunk_size - overlap
    return chunks


def format_fasta(target_id: str, sequence: str) -> str:
    """Format a sequence as FASTA."""
    return f">{target_id}\n{sequence}\n"
