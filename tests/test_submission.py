"""tests/test_submission.py"""
import sys
sys.path.insert(0, '.')

import pytest
import tempfile
import numpy as np
import pandas as pd

from src.submission import SubmissionBuilder
from src.structure_predictor import PredictedStructure


def make_prediction(target_id="R0001", sequence="ACGUGCAU", n_structs=5):
    structs = []
    for i in range(n_structs):
        n = len(sequence)
        coords = np.random.randn(n, 3).astype(np.float32)
        structs.append(PredictedStructure(
            target_id=target_id,
            sequence=sequence,
            c1_coords=coords,
            plddt=70.0 + i,
            plddt_per_residue=np.full(n, 70.0),
            seed=42 + i,
            branch="denovo",
        ))
    return {"target_id": target_id, "sequence": sequence, "structures": structs}


def test_submission_shape():
    """Output CSV should have L rows per sequence (one per residue)."""
    builder = SubmissionBuilder()
    seq = "ACGUGCAU"
    pred = make_prediction(sequence=seq)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    builder.build([pred], path)
    df = pd.read_csv(path)

    assert len(df) == len(seq), f"Expected {len(seq)} rows, got {len(df)}"


def test_submission_columns():
    """All required columns must be present."""
    builder = SubmissionBuilder()
    pred = make_prediction()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    builder.build([pred], path)
    df = pd.read_csv(path)

    required = ["ID", "resname", "resid"]
    for i in range(1, 6):
        required += [f"x_{i}", f"y_{i}", f"z_{i}"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_submission_id_format():
    """ID should be target_id_residueindex."""
    builder = SubmissionBuilder()
    pred = make_prediction(target_id="R1107", sequence="GCA")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    builder.build([pred], path)
    df = pd.read_csv(path)
    assert list(df["ID"]) == ["R1107_1", "R1107_2", "R1107_3"]


def test_submission_no_nan():
    """No NaN values in coordinate columns."""
    builder = SubmissionBuilder()
    pred = make_prediction()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    builder.build([pred], path)
    df = pd.read_csv(path)
    coord_cols = [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]]
    assert df[coord_cols].isna().sum().sum() == 0


def test_fewer_than_5_structures_padded():
    """If fewer than 5 structures are provided, they should be padded."""
    builder = SubmissionBuilder()
    pred = make_prediction(n_structs=2)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    builder.build([pred], path)
    df = pd.read_csv(path)
    # Should still have all 15 coordinate columns (x_1..z_5)
    assert "x_5" in df.columns
    assert "z_5" in df.columns


def test_validate_valid_submission():
    builder = SubmissionBuilder()
    pred = make_prediction()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    builder.build([pred], path)
    assert builder.validate(path) is True
