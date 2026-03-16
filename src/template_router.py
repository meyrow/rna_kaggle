"""
template_router.py — Routing logic: TBM branch vs de novo branch.

The core strategic decision in the pipeline.
Based on Part 1 analysis: different methods win on different targets.

Routing rules (in priority order):
  1. Force de novo if family is "unknown" or "new_to_nature"
  2. Use TBM if best template expected_tm >= tbm_threshold
  3. Use TBM if multiple moderate templates exist (ensemble approach)
  4. Fall back to de novo
"""

import logging
from typing import Literal

from src.template_search import Template
from src.family_classifier import FamilyResult

logger = logging.getLogger(__name__)

Branch = Literal["tbm", "denovo", "hybrid"]


class TemplateRouter:
    """
    Decides which prediction branch to use for each sequence.

    TBM  branch: template priors → Protenix with ca_precomputed templates
    Denovo branch: sequence + MSA only → RibonanzaNet2 + Protenix
    Hybrid branch: use TBM for some candidates, denovo for others
    """

    def __init__(self, cfg: dict):
        self.tbm_threshold = cfg.get("tbm_threshold", 0.45)
        self.force_denovo_families = set(cfg.get("force_denovo_families", ["unknown", "new_to_nature"]))

    def route(self, templates: list[Template], family: FamilyResult) -> Branch:
        """
        Decide branch for this sequence.

        Returns: "tbm", "denovo", or "hybrid"
        """
        # Rule 1: Force de novo for novel/unknown families
        if family.name in self.force_denovo_families:
            logger.debug(f"  → denovo (forced by family={family.name})")
            return "denovo"

        # Rule 2: No templates found
        if not templates:
            logger.debug("  → denovo (no templates found)")
            return "denovo"

        best_tm = templates[0].expected_tm

        # Rule 3: Strong single template → pure TBM
        if best_tm >= self.tbm_threshold:
            logger.debug(f"  → tbm (best_tm={best_tm:.3f} >= threshold={self.tbm_threshold})")
            return "tbm"

        # Rule 4: Moderate templates + known family → hybrid
        n_moderate = sum(1 for t in templates if t.expected_tm >= 0.35)
        if n_moderate >= 2 and family.is_known:
            logger.debug(f"  → hybrid ({n_moderate} moderate templates, family={family.name})")
            return "hybrid"

        # Rule 5: Weak templates → de novo
        logger.debug(f"  → denovo (best_tm={best_tm:.3f} < threshold, n_moderate={n_moderate})")
        return "denovo"

    def get_templates_for_branch(
        self, branch: Branch, templates: list[Template], n_candidates: int
    ) -> list[list[Template]]:
        """
        Distribute templates across the 5 candidate slots.

        For TBM: use top templates, cycling through for diversity.
        For hybrid: split slots between TBM and de novo.
        For denovo: empty template list for all candidates.

        Returns: list of template lists, one per candidate.
        """
        if branch == "denovo":
            return [[] for _ in range(n_candidates)]

        if branch == "tbm":
            # Assign templates in round-robin for diversity
            result = []
            for i in range(n_candidates):
                t = templates[i % len(templates)] if templates else None
                result.append([t] if t else [])
            return result

        if branch == "hybrid":
            # First 3 candidates: TBM with different templates
            # Last 2 candidates: de novo
            result = []
            n_tbm = min(3, n_candidates)
            for i in range(n_tbm):
                t = templates[i % len(templates)] if templates else None
                result.append([t] if t else [])
            for _ in range(n_candidates - n_tbm):
                result.append([])
            return result

        return [[] for _ in range(n_candidates)]
