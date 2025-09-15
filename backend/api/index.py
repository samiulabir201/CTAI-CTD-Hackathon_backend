# api/index.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os

# Import pipeline from root of /app
from pipeline import predict_pipeline

app = FastAPI(title="CTAI - CTD Hackathon Backend")

# -----------------------------
# Request schema
# -----------------------------
class InputData(BaseModel):
    description: str
    uom: Optional[str] = None
    core_market: Optional[str] = None

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "CTAI - CTD Hackathon Backend is running!"}

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    master_item, qty = predict_pipeline(
        data.description,
        data.uom,
        data.core_market,
    )
    return {
        "MasterItemNo": master_item,
        "QtyShipped": qty,
        "uom": data.uom,
        "core_market": data.core_market,
    }


# -----------------------------
# Run locally (optional)
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.index:app", host="0.0.0.0", port=port, reload=False)