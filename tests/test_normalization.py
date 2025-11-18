import pandas as pd

from app.data_loader import DataLoader


def test_normalize_and_index_build(tmp_path):
    csv = tmp_path / "symptom_matrix.csv"
    pd.DataFrame([{"diseases": "X", "Head-Ache": 1, "  Fever  ": 0}]).to_csv(
        csv, index=False
    )
    loader = DataLoader(csv)
    df, cols = loader.load_matrix()
    idx = loader._build_col_index(cols)
    assert "head ache" in idx
    assert "fever" in idx
    assert idx["head ache"] in cols


def test_fuzzy_match_basic(tmp_path):
    csv = tmp_path / "symptom_matrix.csv"
    pd.DataFrame([{"diseases": "X", "headache": 1}]).to_csv(csv, index=False)
    loader = DataLoader(csv)
    df, cols = loader.load_matrix()
    idx = loader._build_col_index(cols)
    norm_candidates = list(idx.keys())
    m = loader._fuzzy_match("head ache", norm_candidates, cutoff=0.6)
    assert m is None or m in norm_candidates
