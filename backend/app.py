from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pipeline import predict_pipeline

app = FastAPI(title="CTAI - CTD Hackathon Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ctai-ctd-hackathon-frontend.vercel.app",  # replace with your real Vercel domain
        "http://localhost:3000",             # for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    description: str
    uom: Optional[str] = None
    core_market: Optional[str] = None

@app.post("/predict")
def predict(data: InputData):
    master_item, qty = predict_pipeline(data.description, data.uom, data.core_market)
    # Echo inputs so the frontend can show them
    return {
        "MasterItemNo": master_item,
        "QtyShipped": qty,
        "uom": data.uom,
        "core_market": data.core_market,
    }

# Health check endpoint
@app.get("/")
def root():
    return {"message": "CTAI - CTD Hackathon Backend is running!"}