from pathlib import Path
import pandas as pd
import pytest

from app.data_loader import DataLoader


def _write_good_matrix(path: Path):
    df = pd.DataFrame(
        [
            {"diseases": "A", "fever": 1, "cough": 0},
            {"diseases": "B", "fever": 0, "cough": 1},
        ]
    )
    df.to_csv(path, index=False)
    return path


def test_load_matrix_ok(tmp_path):
    csv = tmp_path / "symptom_matrix.csv"
    _write_good_matrix(csv)
    loader = DataLoader(csv)
    df, cols = loader.load_matrix()
    assert len(df) == 2
    assert "diseases" in df.columns
    assert "fever" in cols and "cough" in cols


def test_load_matrix_missing_file(tmp_path):
    csv = tmp_path / "no_such_file.csv"
    loader = DataLoader(csv)
    with pytest.raises(FileNotFoundError):
        loader.load_matrix()


def test_load_matrix_missing_diseases(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame([{"x": 1}]).to_csv(csv, index=False)
    loader = DataLoader(csv)
    with pytest.raises(ValueError):
        loader.load_matrix()
