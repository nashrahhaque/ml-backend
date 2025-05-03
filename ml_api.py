# ml_api.py  – FastAPI micro‑service for FairScore model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib

# -------------------------------------------------------
# 1) Model definition & loading
# -------------------------------------------------------
class FairNN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load scaler & model state (CPU safe)
scaler = joblib.load("models/scaler.joblib")
model  = FairNN(10)
model.load_state_dict(torch.load("models/fair_nn.pt",
                                 map_location=torch.device("cpu")))
model.eval()

# -------------------------------------------------------
# 2) FastAPI app + CORS so React can call us
# -------------------------------------------------------
app = FastAPI(title="FairScore ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # in prod restrict to your site
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# 3) Input schema
# -------------------------------------------------------
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

# -------------------------------------------------------
# 4) Prediction endpoint
# -------------------------------------------------------
@app.post("/predict")
def predict(c: Candidate):
    try:
        # Assemble feature vector
        feats = np.array([[
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
        ]], dtype=np.float32)

        # Scale the 6 country‑level columns (cols 4:)
        feats[:, 4:] = scaler.transform(feats[:, 4:])

        # Torch inference
        with torch.no_grad():
            prob = model(torch.from_numpy(feats)).item()

        return {
            "qualification_score": round(prob, 3),
            "ai_decision": (
                "hired" if prob > 0.75
                else "interview" if prob > 0.5
                else "rejected"
            )
        }

    except Exception as e:
        # Bubble up errors with 500 status
        raise HTTPException(status_code=500, detail=str(e))
