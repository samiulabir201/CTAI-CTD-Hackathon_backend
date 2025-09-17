# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from pipeline import predict_pipeline

app = FastAPI(title="CTAI - CTD Hackathon Backend")

# --- CORS (this is the one Render actually runs) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ctai-ctd-hackathon-frontend.vercel.app",  # your Vercel domain
        "http://localhost:3000",                            # local dev
    ],
    allow_credentials=False,             # set True only if you use cookies/auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[],
    max_age=600,                         # cache preflight for 10 minutes
)

class InputData(BaseModel):
    description: str
    uom: Optional[str] = None
    core_market: Optional[str] = None

@app.get("/")
def root():
    return {"message": "CTAI - CTD Hackathon Backend is running!"}

@app.post("/predict")
def predict(data: InputData):
    master_item, qty = predict_pipeline(
        data.description, data.uom, data.core_market
    )
    return {
        "MasterItemNo": master_item,
        "QtyShipped": qty,
        "uom": data.uom,
        "core_market": data.core_market,
    }

# For local runs
if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
