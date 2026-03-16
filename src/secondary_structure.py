"""
secondary_structure.py — Secondary structure prediction wrapper.

Supports: ViennaRNA (RNAfold), EternaFold, CONTRAfold
Output: SecondaryStructure dataclass with dot-bracket, base pairs, stems, loops
"""

import subprocess
import re
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StemLoop:
    """A stem-loop (hairpin) structural element."""
    stem_start: int
    stem_end: int
    loop_start: int
    loop_end: int
    loop_sequence: str

    @property
    def loop_length(self) -> int:
        return self.loop_end - self.loop_start + 1


@dataclass
class SecondaryStructure:
    """Container for secondary structure prediction results."""
    sequence: str
    dot_bracket: str
    mfe: float                              # minimum free energy (kcal/mol)
    base_pairs: list[tuple[int, int]]       # (i, j) 0-indexed
    stems: list[tuple[int, int, int, int]]  # (i_start, i_end, j_start, j_end)
    hairpins: list[StemLoop]
    engine: str = "viennarna"

    # Detected motif positions (filled by MotifCorrector)
    gnra_positions: list[int] = field(default_factory=list)
    kturn_positions: list[int] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.base_pairs)

    @property
    def pair_fraction(self) -> float:
        return (2 * self.n_pairs) / len(self.sequence) if self.sequence else 0.0

    def has_hairpin_of_length(self, n: int) -> bool:
        return any(h.loop_length == n for h in self.hairpins)


class SecondaryStructurePredictor:
    """
    Wraps ViennaRNA (RNAfold) as the default engine.
    Falls back gracefully if not installed (returns minimal structure).
    """

    def __init__(self, cfg: dict):
        self.engine = cfg.get("engine", "viennarna")
        self.temperature = cfg.get("temperature", 37.0)
        self.use_pseudoknot = cfg.get("use_pseudoknot", False)
        self._viennarna_available = self._check_viennarna()

    def _check_viennarna(self) -> bool:
        try:
            result = subprocess.run(
                ["RNAfold", "--version"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("RNAfold not found. Using simple bracket fallback.")
            return False

    def predict(self, sequence: str) -> SecondaryStructure:
        """Predict secondary structure for a given RNA sequence."""
        if self.engine == "viennarna" and self._viennarna_available:
            return self._predict_viennarna(sequence)
        else:
            logger.warning(f"Engine '{self.engine}' unavailable. Using fallback.")
            return self._predict_fallback(sequence)

    def _predict_viennarna(self, sequence: str) -> SecondaryStructure:
        """Call RNAfold subprocess and parse output."""
        cmd = ["RNAfold", "--noPS", f"--temp={self.temperature}"]
        if self.use_pseudoknot:
            cmd.append("--gquad")  # G-quadruplex support

        result = subprocess.run(
            cmd,
            input=sequence,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            logger.error(f"RNAfold failed: {result.stderr}")
            return self._predict_fallback(sequence)

        lines = result.stdout.strip().split("\n")
        # ViennaRNA output: line0 = sequence, line1 = "dotbracket (MFE)"
        db_line = lines[-1]
        # Parse: "(((...))) (-12.34)"
        match = re.match(r'^([.()[\]{}]+)\s+\((-?[\d.]+)\)', db_line)
        if not match:
            logger.warning(f"Could not parse RNAfold output: {db_line}")
            return self._predict_fallback(sequence)

        dot_bracket = match.group(1)
        mfe = float(match.group(2))

        base_pairs = self._parse_base_pairs(dot_bracket)
        stems = self._extract_stems(base_pairs)
        hairpins = self._extract_hairpins(sequence, dot_bracket, base_pairs)

        return SecondaryStructure(
            sequence=sequence,
            dot_bracket=dot_bracket,
            mfe=mfe,
            base_pairs=base_pairs,
            stems=stems,
            hairpins=hairpins,
            engine="viennarna",
        )

    def _predict_fallback(self, sequence: str) -> SecondaryStructure:
        """Minimal fallback: return fully unstructured."""
        n = len(sequence)
        return SecondaryStructure(
            sequence=sequence,
            dot_bracket="." * n,
            mfe=0.0,
            base_pairs=[],
            stems=[],
            hairpins=[],
            engine="fallback",
        )

    def _parse_base_pairs(self, dot_bracket: str) -> list[tuple[int, int]]:
        """Parse dot-bracket notation into list of (i,j) base pairs (0-indexed)."""
        pairs = []
        stack = []
        for i, c in enumerate(dot_bracket):
            if c == "(":
                stack.append(i)
            elif c == ")":
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        return sorted(pairs)

    def _extract_stems(
        self, base_pairs: list[tuple[int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """
        Group consecutive base pairs into stems.
        A stem is a run of pairs (i, j), (i+1, j-1), (i+2, j-2), ...
        Returns list of (i_start, i_end, j_start, j_end).
        """
        if not base_pairs:
            return []
        stems = []
        current_stem = [base_pairs[0]]
        for prev, curr in zip(base_pairs, base_pairs[1:]):
            if curr[0] == prev[0] + 1 and curr[1] == prev[1] - 1:
                current_stem.append(curr)
            else:
                if len(current_stem) >= 2:
                    stems.append((
                        current_stem[0][0], current_stem[-1][0],
                        current_stem[-1][1], current_stem[0][1],
                    ))
                current_stem = [curr]
        if len(current_stem) >= 2:
            stems.append((
                current_stem[0][0], current_stem[-1][0],
                current_stem[-1][1], current_stem[0][1],
            ))
        return stems

    def _extract_hairpins(
        self,
        sequence: str,
        dot_bracket: str,
        base_pairs: list[tuple[int, int]],
    ) -> list[StemLoop]:
        """Extract hairpin loops (closing pair + unpaired loop region)."""
        hairpins = []
        pair_set = set(base_pairs)
        for i, j in base_pairs:
            # Check if all residues between i and j are unpaired
            inner = dot_bracket[i + 1:j]
            if all(c == "." for c in inner):
                loop_seq = sequence[i + 1:j]
                hairpins.append(StemLoop(
                    stem_start=i,
                    stem_end=j,
                    loop_start=i + 1,
                    loop_end=j - 1,
                    loop_sequence=loop_seq,
                ))
        return hairpins
