"""loading data and searching diseases by symptoms"""
from pathlib import Path
import pandas as pd
import logging
import re
from typing import Optional

try:
    from rapidfuzz import process as rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    from difflib import get_close_matches
    _HAS_RAPIDFUZZ = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataLoader:
    def __init__(self, data_source: Path):
        self.data_source = data_source

    def _normalize_text(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _build_col_index(self, symptom_cols: list[str]) -> dict[str, str]:
        """map normalized_name -> original_column_name"""
        idx = {}
        for c in symptom_cols:
            idx[self._normalize_text(c)] = c
        return idx

    def _fuzzy_match(self, token: str, candidates: list[str], cutoff: float = 0.7) -> Optional[str]:
        """match token to candidates (candidates are normalized strings)"""
        if not token:
            return None
        if _HAS_RAPIDFUZZ:
            match = rf_process.extractOne(token, candidates, score_cutoff=int(cutoff * 100))
            return match[0] if match else None
        else:
            matches = get_close_matches(token, candidates, n=1, cutoff=cutoff)
            return matches[0] if matches else None

    def load_matrix(self) -> tuple[pd.DataFrame, list[str]]:
        if not self.data_source.exists():
            raise FileNotFoundError(f"file not found: {self.data_source}")
        df = pd.read_csv(self.data_source)
        df.columns = [c.strip().lower() for c in df.columns]
        if "diseases" not in df.columns:
            raise ValueError("missing 'diseases' column")
        symptom_cols = [c for c in df.columns if c != "diseases"]
        return df, symptom_cols
    
    def find_diseases_by_symptoms(
        self,
        user_symptoms: list[str],
        min_hits: float = 1.0,
        top_k: int = 5,
        symptom_weights: Optional[dict[str, float]] = None,
        fuzzy_cutoff: float = 0.65
    ) -> list[dict]:
        """
        Simple rule-based matcher:
        - normalize and fuzzy-match input symptoms to dataset columns
        - handle English negation
        - optional symptom_weights
        """
        df, symptom_cols = self.load_matrix()

        col_index = self._build_col_index(symptom_cols)
        normalized_cols = list(col_index.keys())

        parsed = []
        unmatched = []
        negation_re = re.compile(r"\b(no|not|without|none|never)\b", flags=re.I)

        for raw in user_symptoms:
            txt = (raw or "").strip()
            if not txt:
                continue
            lower = txt.lower()
            negated = bool(negation_re.search(lower))
            clean = negation_re.sub(" ", lower)
            clean = self._normalize_text(clean)

            mapped_norm = None
            if clean in col_index:
                mapped_norm = clean
            else:
                mapped_norm = self._fuzzy_match(clean, normalized_cols, cutoff=fuzzy_cutoff)

            if mapped_norm:
                mapped_col = col_index[mapped_norm]
                parsed.append({"input": raw, "col": mapped_col, "negated": negated, "norm": mapped_norm})
            else:
                unmatched.append(raw)

        logging.info(f"parsed input -> mapped: {parsed}")
        if unmatched:
            logging.info(f"unmatched symptoms: {unmatched}")
        if not parsed:
            logging.info("no input symptoms matched to known symptom columns")
            return []

        weights = {}
        for p in parsed:
            key_raw = p["input"]
            col = p["col"]
            w = 1.0
            if symptom_weights:
                if key_raw in symptom_weights:
                    w = float(symptom_weights[key_raw])
                elif col in symptom_weights:
                    w = float(symptom_weights[col])
            weights[col] = w

        score_series = pd.Series(0.0, index=df.index)
        for p in parsed:
            col = p["col"]
            w = weights.get(col, 1.0)
            if p["negated"]:
                score_series -= w * df[col].fillna(0)
            else:
                score_series += w * df[col].fillna(0)

        df_scores = df[["diseases"]].copy()
        df_scores["score"] = score_series

        filtered = df_scores[df_scores["score"] >= float(min_hits)].sort_values("score", ascending=False)
        if filtered.empty:
            logging.info("no diseases passed the min_hits threshold")
            return []

        results = []
        seen = set()
        for idx, row in filtered.iterrows():
            disease = row["diseases"]
            if disease in seen:
                continue
            seen.add(disease)
            matched = []
            for p in parsed:
                c = p["col"]
                val = df.at[idx, c]
                if p["negated"]:
                    if val == 1:
                        matched.append(f"NOT {c}")
                else:
                    if val == 1:
                        matched.append(c)
            results.append({
                "disease": disease,
                "score": float(row["score"]),
                "matched_symptoms": matched
            })
            if len(results) >= top_k:
                break

        logging.info(f"matched {len(results)} diseases for symptoms: {user_symptoms}")
        return results
