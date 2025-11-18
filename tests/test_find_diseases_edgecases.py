import pandas as pd

from app.data_loader import DataLoader


def _write_matrix(tmp_path):
    df = pd.DataFrame(
        [
            {"diseases": "Flu", "fever": 1, "cough": 1},
            {"diseases": "Cold", "fever": 0, "cough": 1},
        ]
    )
    path = tmp_path / "symptom_matrix.csv"
    df.to_csv(path, index=False)
    return path


def test_empty_input_returns_empty(tmp_path):
    csv = _write_matrix(tmp_path)
    loader = DataLoader(csv)
    res = loader.find_diseases_by_symptoms([], min_hits=1.0)
    assert res == []


def test_unmatched_symptoms(tmp_path):
    csv = _write_matrix(tmp_path)
    loader = DataLoader(csv)
    res = loader.find_diseases_by_symptoms(["unknownsymptom"], min_hits=1.0)
    assert res == []
