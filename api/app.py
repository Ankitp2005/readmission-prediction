import pickle
import numpy as np
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Hospital Readmission Predictor")

# Allow CORS for local HTML testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model using absolute path relative to this script
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

class PatientData(BaseModel):
    season: int
    age: int
    region: int
    primary_diagnosis: int
    comorbidities_count: int
    length_of_stay: int
    treatment_type: int
    medications_count: int
    followup_visits_last_year: int
    prev_readmissions: int
    insurance_type: int
    discharge_disposition: int
    readmission_risk_score: float

from fastapi.responses import PlainTextResponse, FileResponse
import traceback

@app.get("/")
def home():
    # Serve the frontend index.html from the root path
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    return FileResponse(frontend_path)

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to numpy array in right order
        input_features = np.array([[
            data.season,
            data.age,
            data.region,
            data.primary_diagnosis,
            data.comorbidities_count,
            data.length_of_stay,
            data.treatment_type,
            data.medications_count,
            data.followup_visits_last_year,
            data.prev_readmissions,
            data.insurance_type,
            data.discharge_disposition,
            data.readmission_risk_score
        ]])
        
        pred = model.predict(input_features)
        
        # predict_proba can crash due to scikit-learn version mismatch when pickling
        try:
            probability = model.predict_proba(input_features)[0][1] if hasattr(model, 'predict_proba') else float(pred[0])
        except AttributeError:
            # Fallback if attribute 'multi_class' is missing across sklearn versions
            probability = 1.0 if pred[0] == 1 else 0.0
        
        return {
            "prediction": int(pred[0]),
            "probability_of_readmission": float(probability),
            "status": "Readmission" if pred[0] == 1 else "No Readmission"
        }
    except Exception as e:
        return PlainTextResponse(str(traceback.format_exc()), status_code=500)