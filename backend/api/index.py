from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ..pipeline import predict_pipeline   # import from parent dir

app = FastAPI(title="CTAI - CTD Hackathon Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend domain later
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
    return {
        "MasterItemNo": master_item,
        "QtyShipped": qty,
        "uom": data.uom,
        "core_market": data.core_market,
    }

@app.get("/")
def root():
    return {"message": "CTAI - CTD Hackathon Backend is running!"}