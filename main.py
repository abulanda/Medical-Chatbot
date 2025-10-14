from fastapi import FastAPI, HTTPException
from pathlib import Path
from app.data_loader import DataLoader
from pydantic import BaseModel

app = FastAPI(title="Medical Chatbot API")

data_loader = DataLoader(Path("data/symptom_matrix.csv"))

class SymptomsRequest(BaseModel):
    symptoms: list[str]
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "Medical Chatbot API"}

@app.get("/test")
def test_connection():
    """checks if the data file is accessible"""
    try:
        df, symptom_cols = data_loader.load_matrix()
        return {
            "status": "OK",
            "diseases_count": len(df),
            "symptoms_count": len(symptom_cols),
            "sample_symptoms": symptom_cols[:5]
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="There is no data file. Run: python scripts/fetch_kaggle_data.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/find-diseases")
async def find_diseases(request: SymptomsRequest):
    """searches for diseases based on the provided symptoms"""
    try:
        results = data_loader.find_diseases_by_symptoms(
            request.symptoms, min_hits=1, top_k=request.top_k
        )
        return {
            "query_symptoms": request.symptoms,
            "found_diseases": len(results),
            "diseases": results
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="There is no data file. Run: python scripts/fetch_kaggle_data.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))