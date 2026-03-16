# RNA 3D Folding — Stanford Kaggle Part 2
[![CI](https://github.com/meyrow/rna-3d-folding/actions/workflows/ci.yml/badge.svg)](https://github.com/meyrow/rna-3d-folding/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%204060%208GB-green)
![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)

**Hybrid family-aware 3D RNA structure prediction pipeline.**
Local Ubuntu + RTX 4060 (8GB VRAM) · i9-13980HX · 32GB RAM.

---

## What This Is

A complete prediction pipeline for the
[Stanford RNA 3D Folding Part 2 Kaggle competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2).

The strategy synthesizes three public notebooks (sigmaborov LB 0.438,
Protenix+TBM, High-Score-Without-Hash) into a single routing pipeline:

```
RNA sequence (test_sequences.csv)
        │
        ▼
Secondary structure (ViennaRNA)
        │
        ▼
Family classification (Rfam / heuristic)
        │
        ▼
Template search (MMseqs2 vs PDB_RNA)
        │
        ┌────────────────────────┐
        │ Router: TBM threshold? │
        └──────┬─────────────────┘
         YES   │    NO
         ▼         ▼
    TBM branch   De novo branch
    (Protenix    (RibonanzaNet2
    +templates)  embeddings)
         │               │
         └───────┬───────┘
                 ▼
        Protenix inference
                 ▼
        Motif correction
        (GNRA tetraloop, K-turn)
                 ▼
        5-candidate ensemble
        (ranked by pLDDT)
                 ▼
        submission.csv
```

## Key Data Facts (28 Public Targets)

| Stat | Value |
|------|-------|
| Sequence length | 19 – 4640 nt |
| Targets with ligands | 14 / 28 (Mg²⁺, K⁺, organic) |
| Multi-chain assemblies | 7 / 28 |
| Hardest target | 9MME — 4640 nt, U:8 octamer |
| Label format | up to 40 reference conformations per target |
| Submission format | 5 predicted structures (x_1..z_5) |
| Scoring | mean best-of-5 TM-score over all targets |

## Project Structure

```
rna_folding/
├── config/config.yaml            # all tunable parameters
├── src/
│   ├── pipeline.py               # main orchestrator (7-stage loop)
│   ├── secondary_structure.py    # ViennaRNA wrapper
│   ├── family_classifier.py      # Rfam/cmscan + heuristic fallback
│   ├── template_search.py        # MMseqs2 PDB search
│   ├── template_router.py        # TBM vs de novo routing
│   ├── structure_predictor.py    # Protenix + RibonanzaNet2
│   ├── motif_corrector.py        # GNRA tetraloop + K-turn correction
│   ├── candidate_sampler.py      # 5-seed ensemble + pLDDT ranking
│   ├── submission.py             # competition CSV builder
│   └── utils/
│       ├── sequence_utils.py
│       ├── tm_score.py           # US-align wrapper + numpy approx
│       └── pdb_parser.py
├── scripts/
│   ├── setup_data_links.sh       # symlink kaggle/data/ into project
│   ├── download_models.sh        # RibonanzaNet2 + Protenix checkpoints
│   ├── download_rfam.sh          # Rfam.cm database
│   ├── build_pdb_cache.sh        # MMseqs2 DB + C1' coordinate cache
│   ├── run_pipeline.sh           # end-to-end runner
│   ├── analyze_data.py           # EDA on real competition data
│   └── validate_submission.py    # score against validation_labels.csv
├── notebooks/01_eda.ipynb        # sequence + motif analysis
└── tests/                        # 17 pytest tests
```

## Quick Start

```bash
# 1. Clone & setup environment (conda + apt tools + pip)
git clone https://github.com/meyrow/rna-3d-folding
cd rna-3d-folding
bash setup.sh

# 2. Link your existing Kaggle data (no re-download needed)
bash scripts/setup_data_links.sh /home/ilan/kaggle/data

# 3. Build PDB template search index from your PDB_RNA/ folder (~30 min)
bash scripts/build_pdb_cache.sh

# 4. Download ML model checkpoints (~2GB, needs Kaggle API key)
bash scripts/download_models.sh

# 5. Activate env and run tests
conda activate rna_folding
pytest

# 6. Analyze the competition data
python scripts/analyze_data.py --data_dir /home/ilan/kaggle/data

# 7. Run the full pipeline
python src/pipeline.py --input data/raw/test_sequences.csv

# 8. Score your submission locally
python scripts/validate_submission.py \
    --submission outputs/submission.csv \
    --labels     /home/ilan/kaggle/data/validation_labels.csv
```

## Scoring Targets

| Milestone | TM-score | Status |
|-----------|----------|--------|
| Sample submission (all zeros) | ~0.04 | baseline |
| Approach A — sigmaborov | 0.438 | LB public |
| Human expert (Vfold) | ~0.55 | to beat |
| Top Part 1 teams | ~0.59–0.64 | stretch goal |
| TM ≥ 0.45 | correct global fold | per-target threshold |

## Hardware Notes

- **RTX 4060 (8GB VRAM)**: sequences >400 nt are chunked automatically
- **9MME (4640 nt)**: ~30–60 min per prediction slot; uses monomer tiling strategy
- **bf16 precision**: enabled by default (RTX 4060 Ampere supports it natively)
- **Gradient checkpointing**: enabled for sequences >300 nt

## External Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| RNAfold (ViennaRNA) | Secondary structure | `conda install -c bioconda viennarna` |
| MMseqs2 | Template search | `conda install -c bioconda mmseqs2` |
| cmscan (Infernal) | Rfam family classification | `conda install -c bioconda infernal` |
| US-align | TM-score computation | compiled in `setup.sh` |
| Protenix | AlphaFold3 backbone | `pip install -e external/protenix/` |
| RibonanzaNet2 | RNA foundation model | downloaded via `download_models.sh` |

## License
MIT
