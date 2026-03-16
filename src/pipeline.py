"""
pipeline.py — Main orchestrator for the RNA 3D folding pipeline.

Implements the hybrid family-aware routing strategy:
  RNA sequence
    → secondary structure
    → family classification
    → template search
    → routing (TBM vs de novo)
    → Protenix structure prediction
    → motif correction (GNRA, K-turn)
    → 5-candidate sampling + pLDDT ranking
    → submission.csv
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from src.secondary_structure import SecondaryStructurePredictor
from src.family_classifier import FamilyClassifier
from src.template_search import TemplateSearcher
from src.template_router import TemplateRouter
from src.structure_predictor import StructurePredictor
from src.motif_corrector import MotifCorrector
from src.candidate_sampler import CandidateSampler
from src.submission import SubmissionBuilder
from src.utils.sequence_utils import validate_rna_sequence


def setup_logging(cfg: dict) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("log_file", "outputs/pipeline.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("pipeline")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sequences(input_csv: str) -> pd.DataFrame:
    """
    Load test_sequences.csv in competition format.

    Part 2 columns:
      target_id, sequence, temporal_cutoff, description,
      stoichiometry, all_sequences, ligand_ids, ligand_SMILES
    """
    df = pd.read_csv(input_csv)
    required = {"target_id", "sequence"}
    if not required.issubset(df.columns):
        raise ValueError(f"input CSV must have columns: {required}. Got: {list(df.columns)}")

    # Normalize: uppercase, T→U
    df["sequence"] = df["sequence"].str.upper().str.replace("T", "U")

    # Parse stoichiometry into chain_id and n_copies
    if "stoichiometry" not in df.columns:
        df["stoichiometry"] = "A:1"
    df["chain_id"]       = df["stoichiometry"].str.extract(r"^([A-Z]+):")
    df["n_copies"]       = df["stoichiometry"].str.extract(r":(\d+)$").astype(float).fillna(1).astype(int)
    df["is_multi_chain"] = df["stoichiometry"].str.contains(";") | (df["n_copies"] > 1)

    # Flag presence of ligands (Mg2+, K+, organic small molecules)
    df["has_ligands"] = (
        df.get("ligand_ids", pd.Series([""] * len(df)))
        .fillna("").str.len() > 0
    )

    # Precompute sequence length for routing decisions
    df["seq_len"] = df["sequence"].str.len()

    return df.reset_index(drop=True)


def run_pipeline(config_path: str, input_csv: str, output_csv: str):
    cfg = load_config(config_path)
    logger = setup_logging(cfg)

    # Allow CLI overrides
    if input_csv:
        cfg["pipeline"]["input_csv"] = input_csv
    if output_csv:
        cfg["pipeline"]["output_csv"] = output_csv

    logger.info("=" * 60)
    logger.info("RNA 3D Folding Pipeline — Stanford Kaggle Part 2")
    logger.info("=" * 60)
    logger.info(f"Config:  {config_path}")
    logger.info(f"Input:   {cfg['pipeline']['input_csv']}")
    logger.info(f"Output:  {cfg['pipeline']['output_csv']}")

    # ── Load sequences ────────────────────────────────────────────
    df = load_sequences(cfg["pipeline"]["input_csv"])
    logger.info(f"Loaded {len(df)} sequences")
    logger.info(f"Length range: {df['sequence'].str.len().min()}–{df['sequence'].str.len().max()} nt")

    # ── Initialize all modules ────────────────────────────────────
    ss_predictor = SecondaryStructurePredictor(cfg["secondary_structure"])
    family_clf = FamilyClassifier(cfg["family_classifier"])
    template_searcher = TemplateSearcher(cfg["template_search"])
    router = TemplateRouter(cfg["routing"])
    structure_predictor = StructurePredictor(cfg)
    motif_corrector = MotifCorrector(cfg["motif_correction"])
    candidate_sampler = CandidateSampler(cfg["candidate_sampling"])
    submission_builder = SubmissionBuilder()

    all_predictions = []
    t0 = time.time()

    for idx, row in df.iterrows():
        target_id     = row["target_id"]
        sequence      = row["sequence"]
        seq_len       = int(row.get("seq_len", len(sequence)))
        stoichiometry = str(row.get("stoichiometry", "A:1"))
        n_copies      = int(row.get("n_copies", 1))
        is_multi      = bool(row.get("is_multi_chain", False))
        has_ligands   = bool(row.get("has_ligands", False))
        ligand_ids    = str(row.get("ligand_ids", "")) if has_ligands else ""

        logger.info(
            f"[{idx+1}/{len(df)}] {target_id}  len={seq_len}  "
            f"stoich={stoichiometry}  ligands={ligand_ids or 'none'}"
        )
        if is_multi:
            logger.info(f"  ⚠ Multi-chain assembly: {stoichiometry} ({n_copies} copies)")
        if has_ligands:
            logger.info(f"  ⚠ Ligand-bound: {ligand_ids}")

        # Validate
        if not validate_rna_sequence(sequence):
            logger.warning(f"  Skipping {target_id}: invalid RNA characters")
            continue

        try:
            t_seq = time.time()

            # Stage 1: Secondary structure
            logger.debug(f"  Stage 1: secondary structure")
            sec_struct = ss_predictor.predict(sequence)

            # Stage 2: Family classification
            logger.debug(f"  Stage 2: family classification")
            family = family_clf.classify(sequence, sec_struct)
            logger.info(f"  Family: {family.name} (score={family.score:.3f})")

            # Stage 3: Template search
            logger.debug(f"  Stage 3: template search")
            templates = template_searcher.search(sequence, family)
            best_tmscore = templates[0].expected_tm if templates else 0.0
            logger.info(f"  Templates found: {len(templates)}, best expected TM={best_tmscore:.3f}")

            # Stage 4: Route
            branch = router.route(templates, family)
            logger.info(f"  Branch: {branch}")

            # Stage 5: Structure prediction (5 seeds)
            logger.debug(f"  Stage 5: structure prediction ({branch})")
            raw_structures = candidate_sampler.sample(
                sequence=sequence,
                sec_struct=sec_struct,
                templates=templates if branch == "tbm" else [],
                predictor=structure_predictor,
                branch=branch,
            )

            # Stage 6: Motif correction
            logger.debug(f"  Stage 6: motif correction")
            corrected_structures = [
                motif_corrector.correct(s, sec_struct)
                for s in raw_structures
            ]

            # Stage 7: Rank and pick top 5
            ranked = candidate_sampler.rank(corrected_structures)

            elapsed = time.time() - t_seq
            logger.info(f"  Done in {elapsed:.1f}s  pLDDT_best={ranked[0].plddt:.3f}")

            all_predictions.append({
                "target_id": target_id,
                "sequence": sequence,
                "stoichiometry": stoichiometry,
                "n_copies": n_copies,
                "is_multi_chain": is_multi,
                "has_ligands": has_ligands,
                "ligand_ids": ligand_ids,
                "family": family.name,
                "branch": branch,
                "n_templates": len(templates),
                "structures": ranked,
            })

        except Exception as e:
            logger.error(f"  FAILED {target_id}: {e}", exc_info=True)
            # Produce a fallback zero-coordinate prediction so we don't skip the target
            all_predictions.append({
                "target_id": target_id,
                "sequence": sequence,
                "family": "error",
                "branch": "fallback",
                "n_templates": 0,
                "structures": candidate_sampler.make_fallback(sequence),
            })

    # Build and save submission
    output_path = cfg["pipeline"]["output_csv"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_builder.build(all_predictions, output_path)

    total = time.time() - t0
    logger.info(f"\nDone. {len(all_predictions)} sequences processed in {total/60:.1f} min")
    logger.info(f"Submission saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RNA 3D Folding Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default=None, help="Path to test_sequences.csv")
    parser.add_argument("--output", default=None, help="Path to output submission.csv")
    args = parser.parse_args()
    run_pipeline(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
