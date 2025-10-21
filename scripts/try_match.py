"""simple test script for disease-symptom matching"""

import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main() -> None:
    data_loader = DataLoader(Path("data/symptom_matrix.csv"))
    user_symptoms = ["fever", "cough", "headache"]

    try:
        matches = data_loader.find_diseases_by_symptoms(user_symptoms, min_hits=1, top_k=5)
        if matches:
            logging.info(f"found matches for symptoms {user_symptoms}:")
            for match in matches:
                logging.info(f"- {match['disease']} (score: {match['score']})")
        else:
            logging.info("no matches found")
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"unexpected error: {e}")

if __name__ == "__main__":
    main()
