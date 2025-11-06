import pandas as pd
from pathlib import Path
from app.data_loader import DataLoader

def _write_matrix(tmp_path):
    df = pd.DataFrame([
        {"diseases": "Flu", "fever": 1, "cough": 1, "headache": 1, "fatigue": 1},
        {"diseases": "Cold", "fever": 0, "cough": 1, "headache": 1, "fatigue": 0},
        {"diseases": "Migraine", "fever": 0, "cough": 0, "headache": 1, "fatigue": 0},
        {"diseases": "FoodPoisoning", "fever": 1, "cough": 0, "headache": 0, "fatigue": 1},
    ])
    path = tmp_path / "symptom_matrix.csv"
    df.to_csv(path, index=False)
    return path

def test_exact_match(tmp_path):
    path = _write_matrix(tmp_path)
    loader = DataLoader(Path(path))
    results = loader.find_diseases_by_symptoms(["fever", "cough"], min_hits=1.0, top_k=3)
    assert results, "expected some results"
    assert results[0]["disease"] == "Flu"

def test_fuzzy_match(tmp_path):
    path = _write_matrix(tmp_path)
    loader = DataLoader(Path(path))
    results = loader.find_diseases_by_symptoms(["feever", "head ache"], min_hits=1.0, top_k=3)
    assert results and results[0]["disease"] == "Flu"

def test_negation_changes_ranking(tmp_path):
    path = _write_matrix(tmp_path)
    loader = DataLoader(Path(path))
    results = loader.find_diseases_by_symptoms(["no cough", "fever"], min_hits=1.0, top_k=3)
    assert results
    assert results[0]["disease"] == "FoodPoisoning"

def test_weights_and_min_hits(tmp_path):
    path = _write_matrix(tmp_path)
    loader = DataLoader(Path(path))
    results = loader.find_diseases_by_symptoms(["fever"], min_hits=2.0, top_k=5, symptom_weights={"fever": 2.0})
    assert results
    diseases = [r["disease"] for r in results]
    assert "Flu" in diseases and "FoodPoisoning" in diseases