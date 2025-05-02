# ml_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib

# Define model class
class FairNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Load model + scaler
scaler = joblib.load("models/scaler.joblib")
model = FairNN(10)
model.load_state_dict(torch.load("models/fair_nn.pt", map_location=torch.device("cpu")))
model.eval()

# FastAPI app
app = FastAPI(title="FairScore ML API")

# Input format
class Candidate(BaseModel):
    age_group: int
    education_level: int
    professional_developer: int
    years_code: float
    pct_female_highered: float
    pct_male_highered: float
    pct_female_mided: float
    pct_male_mided: float
    pct_female_lowed: float
    pct_male_lowed: float

@app.post("/predict")
def predict(c: Candidate):
    try:
        features = np.array([[
            c.age_group,
            c.education_level,
            c.professional_developer,
            c.years_code,
            c.pct_female_highered,
            c.pct_male_highered,
            c.pct_female_mided,
            c.pct_male_mided,
            c.pct_female_lowed,
            c.pct_male_lowed
        ]])

        # Scale the country-level features
        features[:, 4:] = scaler.transform(features[:, 4:])
        input_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            prob = model(input_tensor).item()

        return {
            "qualification_score": round(prob, 3),
            "ai_decision": "hired" if prob > 0.75 else "interview" if prob > 0.5 else "rejected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
