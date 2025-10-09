"""wczytywanie danych i wyszukiwanie chorób po objawach"""
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataLoader:
    def __init__(self, data_source: Path):
        self.data_source = data_source

    def load_matrix(self) -> tuple[pd.DataFrame, list[str]]:
        if not self.data_source.exists():
            raise FileNotFoundError(f"Brak pliku {self.data_source}.")
        df = pd.read_csv(self.data_source)
        df.columns = [c.strip().lower() for c in df.columns]
        if "diseases" not in df.columns:
            raise ValueError("Brak kolumny 'diseases'.")
        symptom_cols = [c for c in df.columns if c != "diseases"]
        return df, symptom_cols

    def find_diseases_by_symptoms(
        self, user_symptoms: list[str], min_hits: int = 1, top_k: int = 5
    ) -> list[dict]:
        df, symptom_cols = self.load_matrix()
        wanted = [s.strip().lower() for s in user_symptoms if s.strip().lower() in symptom_cols]
        if not wanted:
            return []

        hits = df[["diseases"] + wanted].copy()
        hits["score"] = hits[wanted].sum(axis=1)
        hits = hits[hits["score"] >= min_hits].sort_values("score", ascending=False)

        results = []
        seen_diseases = set()
        for _, row in hits.iterrows():
            disease = row["diseases"]
            if disease not in seen_diseases:
                seen_diseases.add(disease)
                results.append({
                    "disease": disease,
                    "score": int(row["score"]),
                    "matched_symptoms": [w for w in wanted if row[w] == 1]
                })
            if len(results) >= top_k:
                break

        logging.info(f"Dopasowano {len(results)} chorób dla objawów: {user_symptoms}")
        return results
