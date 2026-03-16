"""
candidate_sampler.py — 5-candidate ensemble sampling + ranking.

The competition requires 5 predicted structures per sequence.
Score = mean of best-of-5 TM-scores across all targets.

Strategy:
  - Use 5 different random seeds for diffusion sampling
  - Optionally mix TBM and de novo candidates (hybrid branch)
  - Rank by pLDDT confidence score
  - For de novo targets: add diversity weighting to avoid 5 near-identical structures
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.structure_predictor import StructurePredictor, PredictedStructure
from src.secondary_structure import SecondaryStructure
from src.template_router import TemplateRouter

logger = logging.getLogger(__name__)


class CandidateSampler:
    """
    Generates and ranks 5 candidate structures per sequence.
    """

    def __init__(self, cfg: dict):
        self.n_seeds = cfg.get("n_seeds", 5)
        self.seeds = cfg.get("seeds", [42, 123, 456, 789, 1337])
        self.ranking_metric = cfg.get("ranking_metric", "plddt")
        self.diversity_weighting = cfg.get("diversity_weighting", 0.2)
        self._router = None  # injected by pipeline

    def sample(
        self,
        sequence: str,
        sec_struct: SecondaryStructure,
        templates: list,
        predictor: StructurePredictor,
        branch: str,
        target_id: str = "unknown",
    ) -> list[PredictedStructure]:
        """
        Generate n_seeds candidate structures.

        For hybrid branch: first 3 seeds use TBM, last 2 use de novo.
        """
        structures = []
        seeds = self.seeds[:self.n_seeds]

        for i, seed in enumerate(seeds):
            # Determine which templates to use for this candidate slot
            if branch == "tbm":
                slot_templates = [templates[i % len(templates)]] if templates else []
            elif branch == "hybrid":
                # First 3: TBM, last 2: de novo
                slot_templates = [templates[i % len(templates)]] if (i < 3 and templates) else []
                effective_branch = "tbm" if slot_templates else "denovo"
            else:
                slot_templates = []
                effective_branch = "denovo"

            if branch != "hybrid":
                effective_branch = branch

            logger.debug(f"    Seed {seed} ({effective_branch}, {len(slot_templates)} templates)")

            struct = predictor.predict(
                sequence=sequence,
                target_id=target_id,
                seed=seed,
                templates=slot_templates,
                branch=effective_branch,
            )
            structures.append(struct)

        return structures

    def rank(self, structures: list[PredictedStructure]) -> list[PredictedStructure]:
        """
        Rank structures for submission.

        Ranking metric:
          - pLDDT: higher is better (confidence-based)
          - Optionally: subtract diversity penalty if structures are too similar

        The competition takes the best-of-5, so we want to maximize
        diversity while still putting the highest-confidence prediction first.
        """
        if not structures:
            return structures

        if self.ranking_metric == "plddt":
            # Primary: pLDDT descending
            # Secondary: diversity bonus (reward structures that are different from #1)
            scored = []
            for i, s in enumerate(structures):
                score = s.plddt
                if i > 0 and self.diversity_weighting > 0:
                    diversity = self._compute_diversity(s, structures[0])
                    score += self.diversity_weighting * diversity
                scored.append((score, s))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in scored]

        # Default: return as-is
        return structures

    def _compute_diversity(
        self, s: PredictedStructure, reference: PredictedStructure
    ) -> float:
        """
        Compute a diversity score between two structures.
        Uses mean C1' RMSD (after centroid alignment).
        Normalized to [0, 100].
        """
        try:
            c1 = s.c1_coords
            c2 = reference.c1_coords
            if c1.shape != c2.shape:
                return 0.0
            # Center both
            c1 = c1 - c1.mean(axis=0)
            c2 = c2 - c2.mean(axis=0)
            rmsd = float(np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=1))))
            # Cap at 20 Angstroms → normalize to 0-100
            return min(rmsd / 20.0 * 100.0, 100.0)
        except Exception:
            return 0.0

    def make_fallback(self, sequence: str) -> list[PredictedStructure]:
        """
        Emergency fallback: return 5 trivial linear-chain structures.
        Used when prediction fails completely.
        """
        from src.structure_predictor import PredictedStructure
        n = len(sequence)
        fallbacks = []
        for i, seed in enumerate(self.seeds[:5]):
            rng = np.random.default_rng(seed)
            # Simple chain: each residue 3.4 Angstroms apart (A-form rise)
            coords = np.zeros((n, 3), dtype=np.float32)
            coords[:, 2] = np.arange(n) * 3.4
            coords += rng.normal(0, 0.1, coords.shape)
            fallbacks.append(PredictedStructure(
                target_id="fallback",
                sequence=sequence,
                c1_coords=coords,
                plddt=30.0,
                plddt_per_residue=np.full(n, 30.0),
                seed=seed,
                branch="fallback",
            ))
        return fallbacks
