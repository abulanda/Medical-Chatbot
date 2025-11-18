from fastapi.testclient import TestClient
from pathlib import Path
import pandas as pd

import main
from app.data_loader import DataLoader

client = TestClient(main.app)


def _write_small_matrix(path: Path):
    df = pd.DataFrame(
        [
            {"diseases": "Flu", "fever": 1, "cough": 1, "headache": 1, "fatigue": 1},
            {"diseases": "Cold", "fever": 0, "cough": 1, "headache": 1, "fatigue": 0},
            {
                "diseases": "Migraine",
                "fever": 0,
                "cough": 0,
                "headache": 1,
                "fatigue": 0,
            },
        ]
    )
    df.to_csv(path, index=False)


def test_get_test_endpoint(tmp_path):
    csv = tmp_path / "symptom_matrix.csv"
    _write_small_matrix(csv)
    main.data_loader = DataLoader(csv)

    r = client.get("/test")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "OK"
    assert j["diseases_count"] == 3


def test_post_find_diseases_basic(tmp_path):
    csv = tmp_path / "symptom_matrix.csv"
    _write_small_matrix(csv)
    main.data_loader = DataLoader(csv)

    payload = {"symptoms": ["fever", "cough"], "top_k": 3}
    r = client.post("/find-diseases", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert "diseases" in j
    assert len(j["diseases"]) > 0
