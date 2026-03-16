"""
scripts/local_eval.py — Run full pipeline locally with RhoFold and score.

Usage:
    cd ~/rna_kaggle
    python3 scripts/local_eval.py

Runs all 28 test sequences through RhoFold and scores against validation_labels.csv.
"""

import sys, os, logging, time, yaml
from pathlib import Path

# Add RhoFold repo to path BEFORE any imports
RHOFOLD_REPO = '/home/ilan/kaggle/data/external/RhoFold'
if os.path.exists(RHOFOLD_REPO):
    sys.path.insert(0, RHOFOLD_REPO)

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

import numpy as np
import pandas as pd

DATA_DIR    = '/home/ilan/kaggle/data'
OUTPUT_DIR  = '/tmp/local_eval_out'
OUT_CSV     = f'{OUTPUT_DIR}/submission.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
cfg = {
    'pipeline':   {'n_candidates': 5, 'device': 'cuda', 'chunk_length': 400,
                   'max_sequence_length': 6000,
                   'input_csv':  f'{DATA_DIR}/test_sequences.csv',
                   'output_csv': OUT_CSV},
    'secondary_structure': {'engine': 'viennarna', 'temperature': 37.0, 'use_pseudoknot': False},
    'family_classifier':   {'rfam_db': '', 'evalue_threshold': 1e-5,
                             'known_families': ['riboswitch','tRNA','ribosomal']},
    'template_search':     {'enabled': False, 'mmseqs2_db': '',
                             'pdb_c1_cache': f'{DATA_DIR}/pdb_c1_coords.pkl',
                             'max_templates': 0, 'min_seq_identity': 0.25, 'min_coverage': 0.5},
    'routing':             {'tbm_threshold': 0.45,
                             'force_denovo_families': ['unknown','large_ncrna']},
    'protenix':            {'checkpoint': '',
                             'rhofold_checkpoint': '/home/ilan/kaggle/data/models/rhofold/rhofold_pretrained_params.pt',
                             'rhofold_repo': RHOFOLD_REPO,
                             'use_template': 'ca_precomputed', 'n_cycle': 10, 'n_step': 200,
                             'use_msa': False, 'msa_dir': f'{DATA_DIR}/MSA_v2',
                             'n_template_blocks': 2, 'dtype': 'bf16',
                             'gradient_checkpointing': True},
    'ribonanzanet2':       {'checkpoint': f'{DATA_DIR}/models/ribonanzanet2/pytorch_model_fsdp.bin',
                             'network_config': f'{DATA_DIR}/models/ribonanzanet2/pairwise.yaml',
                             'freeze_encoder': True},
    'motif_correction':    {'enabled': True, 'gnra_tetraloop': True, 'kturn': True,
                             'motif_detection_rmsd': 2.0, 'correction_weight': 0.85},
    'candidate_sampling':  {'n_seeds': 5, 'seeds': [42,123,456,789,1337],
                             'ranking_metric': 'plddt', 'diversity_weighting': 0.2},
}

# Write config to yaml so run_pipeline can read it
cfg_path = f'{OUTPUT_DIR}/config.yaml'
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f)

# ── Run pipeline ──────────────────────────────────────────────────────────────
from src.pipeline import run_pipeline

t0 = time.time()
run_pipeline(cfg_path, cfg['pipeline']['input_csv'], OUT_CSV)
elapsed = time.time() - t0
print(f'\nTotal time: {elapsed/60:.1f} min  ({elapsed/28:.1f}s per sequence)')

# ── Score ─────────────────────────────────────────────────────────────────────
for labels_csv in [
    f'{DATA_DIR}/validation_labels.csv',
    f'{DATA_DIR}/competitions/stanford-rna-3d-folding-2/validation_labels.csv',
]:
    if os.path.exists(labels_csv):
        print(f'\nScoring against: {labels_csv}')
        os.system(f'python3 scripts/validate_submission.py --submission {OUT_CSV} --labels {labels_csv}')
        break
else:
    print(f'\nNo validation_labels.csv found. Run manually:')
    print(f'  python3 scripts/validate_submission.py --submission {OUT_CSV} --labels <path>')

