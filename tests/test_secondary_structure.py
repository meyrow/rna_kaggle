"""tests/test_secondary_structure.py"""
import sys
sys.path.insert(0, '.')

import pytest
from src.secondary_structure import SecondaryStructurePredictor, SecondaryStructure


@pytest.fixture
def cfg():
    return {"engine": "viennarna", "temperature": 37.0, "use_pseudoknot": False}


@pytest.fixture
def predictor(cfg):
    return SecondaryStructurePredictor(cfg)


def test_parse_base_pairs():
    pred = SecondaryStructurePredictor({"engine": "fallback"})
    db = "(((...)))"
    pairs = pred._parse_base_pairs(db)
    assert (0, 8) in pairs
    assert (1, 7) in pairs
    assert (2, 6) in pairs
    assert len(pairs) == 3


def test_parse_base_pairs_empty():
    pred = SecondaryStructurePredictor({"engine": "fallback"})
    assert pred._parse_base_pairs("....") == []


def test_extract_hairpins():
    pred = SecondaryStructurePredictor({"engine": "fallback"})
    seq = "GCGAAAGCG"
    db  = "(((...)))"
    pairs = pred._parse_base_pairs(db)
    hairpins = pred._extract_hairpins(seq, db, pairs)
    assert len(hairpins) == 1
    assert hairpins[0].loop_sequence == "AAA"
    assert hairpins[0].loop_length == 3


def test_gnra_detection_in_hairpin():
    """
    GNRA tetraloop: G-N-R-A pattern in a 4-nt hairpin loop.
    GAAA = G-A(N)-A(R, purine)-A — valid GNRA.

    Structure:    5'-AC[GAAA]CG-3'
    dot-bracket:  "((.....)))" — no: use "((....))" for 8-nt seq.

    Pairs (0,7) and (1,6); innermost (1,6) → loop = seq[2:6] = "GAAA".
    """
    pred = SecondaryStructurePredictor({"engine": "fallback"})

    # 3-nt loop — NOT GNRA (too short)
    seq2 = "GGGAAACCC"   # 9 chars
    db2  = "(((...)))"   # 9 chars; innermost pair (2,6), loop = seq2[3:6] = "AAA"
    pairs2 = pred._parse_base_pairs(db2)
    hp2 = pred._extract_hairpins(seq2, db2, pairs2)
    assert len(hp2) == 1
    assert hp2[0].loop_length == 3          # 3-nt, not a tetraloop

    # 4-nt GNRA loop ✓
    # "((....))" = 8 chars; pairs (0,7),(1,6); innermost (1,6) → loop = seq[2:6]
    seq3 = "ACGAAACG"    # 8 chars: pos 2-5 = "GAAA"
    db3  = "((....))"    # 8 chars
    pairs3 = pred._parse_base_pairs(db3)
    hp3 = pred._extract_hairpins(seq3, db3, pairs3)
    assert len(hp3) == 1
    assert hp3[0].loop_sequence == "GAAA"   # 4-nt GNRA ✓
    assert hp3[0].loop_length == 4


def test_fallback_structure():
    pred = SecondaryStructurePredictor({"engine": "nonexistent"})
    struct = pred._predict_fallback("ACGU")
    assert struct.dot_bracket == "...."
    assert struct.mfe == 0.0
    assert struct.base_pairs == []


def test_structure_fields():
    pred = SecondaryStructurePredictor({"engine": "fallback"})
    struct = pred.predict("ACGUACGU")
    assert isinstance(struct, SecondaryStructure)
    assert struct.sequence == "ACGUACGU"
    assert len(struct.dot_bracket) == 8
