"""tests/test_motif_corrector.py"""
import sys
sys.path.insert(0, '.')

import pytest
import numpy as np

from src.motif_corrector import MotifCorrector, MotifHit, GNRA_CANONICAL_OFFSETS
from src.secondary_structure import SecondaryStructurePredictor
from src.structure_predictor import PredictedStructure


@pytest.fixture
def cfg():
    return {
        "enabled": True,
        "gnra_tetraloop": True,
        "kturn": True,
        "motif_detection_rmsd": 2.0,
        "correction_weight": 0.85,
    }


@pytest.fixture
def corrector(cfg):
    return MotifCorrector(cfg)


def make_struct(seq, seed=42):
    """Create a dummy PredictedStructure with helical coords."""
    n = len(seq)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n * 0.6, n)
    coords = np.stack([9.0 * np.cos(t), 9.0 * np.sin(t), 2.8 * np.arange(n)], axis=1)
    coords = coords.astype(np.float32)
    return PredictedStructure(
        target_id="test",
        sequence=seq,
        c1_coords=coords,
        plddt=70.0,
        plddt_per_residue=np.full(n, 70.0),
        seed=seed,
        branch="denovo",
    )


def test_gnra_detection_basic(corrector):
    """Sequence with known GNRA tetraloop in a 4-nt hairpin."""
    ss_pred = SecondaryStructurePredictor({"engine": "fallback"})
    # Manually build a secondary structure with a GNRA hairpin
    seq = "GCGGAAAGCGC"
    db  = "((((....)))"
    ss = ss_pred._predict_fallback(seq)
    # Inject a 4-nt loop hairpin
    from src.secondary_structure import StemLoop
    ss.hairpins = [StemLoop(
        stem_start=3, stem_end=8,
        loop_start=4, loop_end=7,
        loop_sequence="GAAA",
    )]
    hits = corrector._detect_gnra(seq, ss)
    assert len(hits) == 1
    assert hits[0].motif_type == "gnra"
    assert hits[0].sequence_match == "GAAA"


def test_gnra_detection_non_gnra(corrector):
    """4-nt loop that is NOT GNRA (e.g. CCCC) should not be detected."""
    ss_pred = SecondaryStructurePredictor({"engine": "fallback"})
    seq = "GCCCCCGCGC"
    ss = ss_pred._predict_fallback(seq)
    from src.secondary_structure import StemLoop
    ss.hairpins = [StemLoop(
        stem_start=1, stem_end=6,
        loop_start=2, loop_end=5,
        loop_sequence="CCCC",
    )]
    hits = corrector._detect_gnra(seq, ss)
    assert len(hits) == 0


def test_gnra_correction_changes_coords(corrector):
    """
    Applying GNRA correction should change C1' coordinates for loop residues.
    We disable K-turn in this fixture (default cfg has kturn=True which would
    also modify residues near the sequence pattern 'GA').
    """
    # Use a corrector with K-turn disabled to isolate GNRA correction
    gnra_only_corrector = MotifCorrector({
        "enabled": True,
        "gnra_tetraloop": True,
        "kturn": False,
        "motif_detection_rmsd": 2.0,
        "correction_weight": 0.85,
    })
    ss_pred = SecondaryStructurePredictor({"engine": "fallback"})
    seq = "GCGGAAAGCGC"
    ss = ss_pred._predict_fallback(seq)
    from src.secondary_structure import StemLoop
    ss.hairpins = [StemLoop(
        stem_start=3, stem_end=8,
        loop_start=4, loop_end=7,
        loop_sequence="GAAA",
    )]
    struct = make_struct(seq)
    original_coords = struct.c1_coords.copy()

    corrected = gnra_only_corrector.correct(struct, ss)

    # Loop residues 4,5,6,7 should have changed
    loop_idx = [4, 5, 6, 7]
    changed = np.any(corrected.c1_coords[loop_idx] != original_coords[loop_idx])
    assert changed, "GNRA correction should modify loop residue coordinates"

    # Non-loop residues should be unchanged (K-turn disabled)
    non_loop = [0, 1, 2, 3, 8, 9, 10]
    unchanged = np.allclose(
        corrected.c1_coords[non_loop], original_coords[non_loop], atol=1e-6
    )
    assert unchanged, "GNRA correction should NOT modify non-loop residues"


def test_correction_weight_zero(cfg):
    """correction_weight=0 should produce no change."""
    cfg["correction_weight"] = 0.0
    corrector = MotifCorrector(cfg)
    ss_pred = SecondaryStructurePredictor({"engine": "fallback"})
    seq = "GCGGAAAGCGC"
    ss = ss_pred._predict_fallback(seq)
    from src.secondary_structure import StemLoop
    ss.hairpins = [StemLoop(4, 7, 4, 7, "GAAA")]
    struct = make_struct(seq)
    corrected = corrector.correct(struct, ss)
    # With weight=0, loop coords should be unchanged
    # (correction_weight * delta = 0)
    assert np.allclose(corrected.c1_coords, struct.c1_coords, atol=1e-5)


def test_disabled_corrector():
    """Disabled corrector returns original structure unchanged."""
    corrector = MotifCorrector({"enabled": False})
    ss_pred = SecondaryStructurePredictor({"engine": "fallback"})
    seq = "GCGGAAAGCGC"
    ss = ss_pred._predict_fallback(seq)
    struct = make_struct(seq)
    result = corrector.correct(struct, ss)
    assert np.allclose(result.c1_coords, struct.c1_coords)
