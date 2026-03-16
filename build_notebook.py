"""
build_notebook.py — Regenerate the Kaggle submission notebook from src/ files.

Usage:
    python build_notebook.py               # rebuild notebook_submission.ipynb
    python build_notebook.py --run-local   # rebuild and run locally with test data
    python build_notebook.py --validate    # rebuild + run + score against validation labels

Workflow:
    1. Edit src/ modules locally
    2. Run: python build_notebook.py
    3. Test locally: jupyter nbconvert --to notebook --execute notebook_submission.ipynb
    4. git push → Kaggle notebook syncs from this repo
"""

import json
import sys
import textwrap
from pathlib import Path

# ── Which src files become notebook cells (in order) ──────────────────────────
SRC_CELLS = [
    ("src/utils/sequence_utils.py", "Utils — sequence helpers"),
    ("src/secondary_structure.py",  "Stage 1 — Secondary structure prediction"),
    ("src/family_classifier.py",    "Stage 2 — Family classification (Rfam heuristic)"),
    ("src/template_search.py",      "Stage 3 — Template search (MMseqs2 / PDB cache)"),
    ("src/template_router.py",      "Stage 4 — Template router (TBM vs de novo)"),
    ("src/structure_predictor.py",  "Stage 5 — Structure predictor (Protenix + RibonanzaNet2)"),
    ("src/motif_corrector.py",      "Stage 6 — Motif correction (GNRA tetraloop, K-turn)"),
    ("src/candidate_sampler.py",    "Stage 7 — Candidate sampling (5 seeds, pLDDT ranking)"),
    ("src/submission.py",           "Output — Submission CSV builder"),
]

OUTPUT_NOTEBOOK = "notebook_submission.ipynb"


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text, "outputs": []}


def code_cell(source: str, label: str = "") -> dict:
    lines = source.splitlines(keepends=True)
    if label:
        lines = [f"# {'─'*60}\n", f"# {label}\n", f"# {'─'*60}\n"] + lines
    return {
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    }



def strip_src_imports(source: str) -> str:
    """
    Remove 'from src.xxx import yyy' and 'import src.xxx' lines.
    When src/ files are inlined as notebook cells, all classes are
    already defined in the notebook scope — cross-imports crash on Kaggle.
    Also remove 'from src.utils.xxx import yyy' patterns.
    """
    lines = source.splitlines(keepends=True)
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove any from src. or import src. lines
        if stripped.startswith("from src.") or stripped.startswith("import src."):
            # Keep as comment so the intent is visible
            cleaned.append("# " + line.rstrip() + "  # inlined above\n")
        else:
            cleaned.append(line)
    return "".join(cleaned)


def build_notebook():
    cells = []

    # ── Title ──────────────────────────────────────────────────────────────────
    cells.append(md_cell(textwrap.dedent("""\
        # RNA 3D Folding Part 2 — Submission Notebook
        **Hybrid family-aware pipeline:**
        RNA sequence → secondary structure → family classification → template search
        → TBM/de-novo routing → Protenix inference → motif correction (GNRA, K-turn)
        → 5-candidate ensemble → submission.csv

        Strategy based on:
        - **Approach A** (sigmaborov, LB 0.438): Protenix + RibonanzaNet2 baseline
        - **Approach B** (gourabr0y555): Protenix + TBM templates
        - **Approach C** (artemevstafyev): High-score without hash tricks (pure DL)
    """)))

    # ── Cell 1: Environment detection ──────────────────────────────────────────
    env_cell = textwrap.dedent("""\
        import os
        import sys
        import warnings
        warnings.filterwarnings("ignore")

        # ── Environment detection ─────────────────────────────────────────────
        if os.path.exists("/kaggle/input"):
            KAGGLE_ENV = True
            OUTPUT_DIR = "/kaggle/working"
            PIPELINE_DIR = "."  # notebook runs from working dir; src cells already defined

            # Auto-detect competition data path — try known slugs
            _candidates = [
                "/kaggle/input/stanford-rna-3d-folding-2",
                "/kaggle/input/stanford-rna-3d-folding",
            ]
            DATA_DIR = None
            for _p in _candidates:
                if os.path.exists(_p) and os.path.exists(f"{_p}/test_sequences.csv"):
                    DATA_DIR = _p
                    break

            # Diagnostics — always print so logs show what was mounted
            print("=== /kaggle/input/ contents ===")
            try:
                for _item in sorted(os.listdir("/kaggle/input")):
                    _full = f"/kaggle/input/{_item}"
                    _files = os.listdir(_full)[:5] if os.path.isdir(_full) else []
                    print(f"  {_item}/  {_files}")
            except Exception as _e:
                print(f"  (error listing: {_e})")

            if DATA_DIR is None:
                # Last resort: search for test_sequences.csv anywhere under /kaggle/input
                import glob
                _found = glob.glob("/kaggle/input/**/test_sequences.csv", recursive=True)
                if _found:
                    DATA_DIR = os.path.dirname(_found[0])
                    print(f"  Found test_sequences.csv via glob: {DATA_DIR}")
                else:
                    print("  ERROR: test_sequences.csv not found!")
                    print("  ACTION NEEDED: Add competition to this notebook in Kaggle UI")
                    print("  Go to notebook settings (right panel) -> Data -> Add competition")
                    DATA_DIR = "/kaggle/input/stanford-rna-3d-folding-2"  # will fail clearly
        else:
            KAGGLE_ENV   = False
            DATA_DIR     = "/home/ilan/kaggle/data"
            OUTPUT_DIR   = "."
            PIPELINE_DIR = "."

        sys.path.insert(0, PIPELINE_DIR)

        print(f"Environment : {'KAGGLE' if KAGGLE_ENV else 'LOCAL'}")
        print(f"DATA_DIR    : {DATA_DIR}")
        print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
        print(f"Python      : {sys.version.split()[0]}")
    """)
    cells.append(code_cell(env_cell, "Environment Setup"))

    # ── Cell 2: Install dependencies ───────────────────────────────────────────
    deps_cell = textwrap.dedent("""\
        # Install bioinformatics tools needed by the pipeline
        # (These are pre-installed in many Kaggle environments; safe to run regardless)
        import subprocess

        def install_if_missing(cmd_check: str, install_cmd: list):
            result = subprocess.run(["which", cmd_check], capture_output=True)
            if result.returncode != 0:
                print(f"Installing {cmd_check}...")
                subprocess.run(install_cmd, check=False, capture_output=True)
            else:
                print(f"  {cmd_check}: already available")

        # ViennaRNA (secondary structure)
        try:
            import RNA
            print("  ViennaRNA Python: available")
        except ImportError:
            print("  ViennaRNA Python: not available, using subprocess RNAfold")

        # Check command-line tools
        for tool, apt_pkg in [
            ("RNAfold",  "vienna-rna"),
            ("mmseqs",   ""),   # install from bioconda via setup.sh
            ("cmscan",   "infernal"),
            ("USalign",  ""),   # compiled in setup.sh
        ]:
            result = subprocess.run(["which", tool], capture_output=True)
            status = "✓ available" if result.returncode == 0 else "✗ not found (fallback active)"
            print(f"  {tool:12s}: {status}")

        print("\\nNote: missing tools trigger graceful fallbacks in the pipeline.")
    """)
    cells.append(code_cell(deps_cell, "Dependency Check"))

    # ── Cell 3: Load competition data ──────────────────────────────────────────
    load_cell = textwrap.dedent("""\
        import pandas as pd
        import numpy as np
        from pathlib import Path

        # Load test sequences
        test_sequences = pd.read_csv(f"{DATA_DIR}/test_sequences.csv")
        test_sequences["sequence"] = test_sequences["sequence"].str.upper().str.replace("T", "U")

        # Parse stoichiometry metadata
        test_sequences["n_copies"]   = test_sequences["stoichiometry"].str.extract(r":(\\d+)$").astype(float).fillna(1).astype(int)
        test_sequences["has_ligands"] = test_sequences["ligand_ids"].notna() & (test_sequences["ligand_ids"].astype(str).str.len() > 1)
        test_sequences["is_complex"] = test_sequences["stoichiometry"].str.contains(";") | (test_sequences["n_copies"] > 1)
        test_sequences["seq_len"]    = test_sequences["sequence"].str.len()

        print(f"Test sequences loaded : {len(test_sequences)}")
        print(f"Length range          : {test_sequences['seq_len'].min()} – {test_sequences['seq_len'].max()} nt")
        print(f"Has ligands           : {test_sequences['has_ligands'].sum()} / {len(test_sequences)}")
        print(f"Multi-chain           : {test_sequences['is_complex'].sum()} / {len(test_sequences)}")
        print()
        print(test_sequences[["target_id","seq_len","stoichiometry","has_ligands"]].to_string())
    """)
    cells.append(code_cell(load_cell, "Load Competition Data"))

    # ── Cells 4–12: Inline src/ modules ───────────────────────────────────────
    for src_path, label in SRC_CELLS:
        p = Path(src_path)
        if not p.exists():
            print(f"  WARNING: {src_path} not found, skipping cell")
            cells.append(code_cell(f"# {src_path} not found — cell skipped", label))
            continue
        source = p.read_text()
        # Strip cross-imports (from src.xxx import yyy) — classes are already
        # defined in earlier notebook cells, these imports crash on Kaggle
        source = strip_src_imports(source)
        cells.append(code_cell(source, label))
        print(f"  Added cell: {label} ({len(source)} chars from {src_path})")

    # ── Cell: Instantiate pipeline modules ────────────────────────────────────
    init_cell = textwrap.dedent("""\
        import yaml

        # Load config
        config_path = Path(PIPELINE_DIR) / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            # Minimal inline config for Kaggle (no yaml file present)
            cfg = {
                "pipeline":      {"n_candidates": 5, "device": "cuda", "chunk_length": 400,
                                   "max_sequence_length": 6000},
                "secondary_structure": {"engine": "viennarna", "temperature": 37.0,
                                        "use_pseudoknot": False},
                "family_classifier":   {"rfam_db": "", "evalue_threshold": 1e-5,
                                        "known_families": ["riboswitch","tRNA","ribosomal"]},
                "template_search":     {"enabled": True, "mmseqs2_db": "",
                                        "pdb_c1_cache": f"{DATA_DIR}/pdb_c1_coords.pkl",
                                        "max_templates": 10, "min_seq_identity": 0.25,
                                        "min_coverage": 0.5},
                "routing":        {"tbm_threshold": 0.45,
                                   "force_denovo_families": ["unknown","large_ncrna"]},
                "protenix":       {"checkpoint": f"{DATA_DIR}/models/protenix_base_default_v0.5.0.pt",
                                   "use_template": "ca_precomputed", "n_cycle": 10,
                                   "n_step": 200, "use_msa": True,
                                   "msa_dir": f"{DATA_DIR}/MSA_v2",
                                   "n_template_blocks": 2, "dtype": "bf16",
                                   "gradient_checkpointing": True},
                "ribonanzanet2":  {"checkpoint": f"{DATA_DIR}/models/ribonanzanet2/pytorch_model_fsdp.bin",
                                   "network_config": f"{DATA_DIR}/models/ribonanzanet2/pairwise.yaml",
                                   "freeze_encoder": True},
                "motif_correction":    {"enabled": True, "gnra_tetraloop": True, "kturn": True,
                                        "motif_detection_rmsd": 2.0, "correction_weight": 0.85},
                "candidate_sampling":  {"n_seeds": 5, "seeds": [42,123,456,789,1337],
                                        "ranking_metric": "plddt", "diversity_weighting": 0.2},
            }

        # Override device paths to DATA_DIR for Kaggle
        if KAGGLE_ENV:
            cfg["pipeline"]["device"] = "cuda"

        # Instantiate modules
        ss_predictor       = SecondaryStructurePredictor(cfg["secondary_structure"])
        family_clf         = FamilyClassifier(cfg["family_classifier"])
        template_searcher  = TemplateSearcher(cfg["template_search"])
        router             = TemplateRouter(cfg["routing"])
        structure_pred     = StructurePredictor(cfg)
        motif_corrector    = MotifCorrector(cfg["motif_correction"])
        candidate_sampler  = CandidateSampler(cfg["candidate_sampling"])
        submission_builder = SubmissionBuilder()

        print("Pipeline modules initialised ✓")
    """)
    cells.append(code_cell(init_cell, "Initialise Pipeline Modules"))

    # ── Cell: Main prediction loop ────────────────────────────────────────────
    loop_cell = textwrap.dedent("""\
        import time
        import logging
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        logger = logging.getLogger("notebook")

        all_predictions = []
        t_total = time.time()

        for idx, row in test_sequences.iterrows():
            target_id     = row["target_id"]
            sequence      = row["sequence"]
            seq_len       = int(row["seq_len"])
            stoichiometry = str(row.get("stoichiometry", "A:1"))
            has_ligands   = bool(row.get("has_ligands", False))

            logger.info(f"[{idx+1}/{len(test_sequences)}] {target_id}  "
                        f"len={seq_len}  stoich={stoichiometry}")

            # Validate sequence
            if not sequence or not all(c in "ACGUN" for c in sequence.upper()):
                logger.warning(f"  Skipping {target_id}: invalid characters")
                continue

            try:
                t_seq = time.time()

                # Stage 1: Secondary structure
                sec_struct = ss_predictor.predict(sequence)

                # Stage 2: Family classification
                family = family_clf.classify(sequence, sec_struct)
                logger.info(f"  Family: {family.name}")

                # Stage 3: Template search
                templates = template_searcher.search(sequence, family)
                logger.info(f"  Templates: {len(templates)}, "
                            f"best TM≈{templates[0].expected_tm:.3f}" if templates else "  Templates: 0")

                # Stage 4: Route
                branch = router.route(templates, family)
                logger.info(f"  Branch: {branch}")

                # Stage 5+6+7: Predict → correct → rank
                raw_structs = candidate_sampler.sample(
                    sequence=sequence, sec_struct=sec_struct,
                    templates=templates if branch == "tbm" else [],
                    predictor=structure_pred, branch=branch,
                    target_id=target_id,
                )
                corrected = [motif_corrector.correct(s, sec_struct) for s in raw_structs]
                ranked    = candidate_sampler.rank(corrected)

                logger.info(f"  Done in {time.time()-t_seq:.1f}s  "
                            f"pLDDT={ranked[0].plddt:.1f}")

                all_predictions.append({
                    "target_id":     target_id,
                    "sequence":      sequence,
                    "stoichiometry": stoichiometry,
                    "has_ligands":   has_ligands,
                    "family":        family.name,
                    "branch":        branch,
                    "n_templates":   len(templates),
                    "structures":    ranked,
                })

            except Exception as e:
                logger.error(f"  FAILED {target_id}: {e}", exc_info=True)
                all_predictions.append({
                    "target_id":  target_id,
                    "sequence":   sequence,
                    "stoichiometry": stoichiometry,
                    "has_ligands":  False,
                    "family":     "error",
                    "branch":     "fallback",
                    "n_templates": 0,
                    "structures": candidate_sampler.make_fallback(sequence),
                })

        logger.info(f"\\nAll {len(all_predictions)} sequences done in "
                    f"{(time.time()-t_total)/60:.1f} min")
    """)
    cells.append(code_cell(loop_cell, "Main Prediction Loop"))

    # ── Cell: Build and save submission ───────────────────────────────────────
    save_cell = textwrap.dedent("""\
        output_path = f"{OUTPUT_DIR}/submission.csv"
        submission_builder.build(all_predictions, output_path)

        # Validate format
        import pandas as pd
        df = pd.read_csv(output_path)
        print(f"\\nsubmission.csv written: {output_path}")
        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {list(df.columns)}")
        print(f"  Targets : {df['ID'].str.rsplit('_',n=1).str[0].nunique()}")
        print()
        print(df.head(3).to_string())
    """)
    cells.append(code_cell(save_cell, "Build submission.csv"))

    # ── Cell: Quick local validation (skipped on Kaggle) ──────────────────────
    val_cell = textwrap.dedent("""\
        # Run only locally to score against validation_labels.csv
        if not KAGGLE_ENV:
            import subprocess, sys
            labels_path = f"{DATA_DIR}/validation_labels.csv"
            if Path(labels_path).exists():
                result = subprocess.run(
                    [sys.executable, "scripts/validate_submission.py",
                     "--submission", output_path,
                     "--labels",     labels_path],
                    capture_output=False
                )
            else:
                print(f"Labels not found at {labels_path} — skipping validation")
        else:
            print("On Kaggle: validation scoring not available (no labels)")
    """)
    cells.append(code_cell(val_cell, "Local Validation (skipped on Kaggle)"))

    # ── Assemble notebook ─────────────────────────────────────────────────────
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [
                    {
                        "sourceType": "competition",
                        "datasetId": "stanford-rna-3d-folding-2",
                    }
                ],
                "isGpuEnabled": True,
                "isInternetEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path(OUTPUT_NOTEBOOK)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"\n{'='*55}")
    print(f"Notebook written: {OUTPUT_NOTEBOOK}")
    print(f"  Cells   : {len(cells)}")
    print(f"  Size    : {out.stat().st_size // 1024} KB")
    print(f"{'='*55}")
    print("\nNext steps:")
    print("  Local test : jupyter nbconvert --to notebook --execute notebook_submission.ipynb")
    print("  Push       : git add notebook_submission.ipynb && git commit -m 'update notebook' && git push")
    print("  Kaggle     : Open notebook, pull latest, click Submit")


if __name__ == "__main__":
    build_notebook()
