"""Pobiera dane z Kaggle i zapisuje macierz choroba - objawy w CSV"""

from pathlib import Path
import pandas as pd
import logging
import kagglehub

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASET = "dhivyeshrk/diseases-and-symptoms-dataset"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_matrix_csv(dataset_dir: Path) -> Path:
    candidates = list(dataset_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Brak plików CSV")
    for csv_path in candidates:
        try:
            df = pd.read_csv(csv_path, nrows=10)
            cols = [c.strip().lower() for c in df.columns]
            if "diseases" in cols and len(cols) > 10:
                logging.info(f"Znaleziono plik CSV: {csv_path.name}")
                return csv_path
        except Exception as e:
            logging.warning(f"Nie udało się wczytać pliku {csv_path.name}: {e}")
            continue
    raise RuntimeError("Brak odpowiedniego pliku CSV")

def main() -> None:
    logging.info("Rozpoczęto pobieranie zbioru danych z Kaggle")
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    logging.info(f"Pobrano dane do: {dataset_path}")

    csv_path = find_matrix_csv(dataset_path)
    df = pd.read_csv(csv_path)

    unique_diseases = df["diseases"].nunique()
    logging.info(f"Liczba chorób: {unique_diseases}")
    logging.info(f"Liczba symptomów: {len(df.columns) - 1}")

    out_file = OUT_DIR / "symptom_matrix.csv"
    df.to_csv(out_file, index=False)
    logging.info(f"Zapisano przetworzony plik do: {out_file.resolve()}")

if __name__ == "__main__":
    main()