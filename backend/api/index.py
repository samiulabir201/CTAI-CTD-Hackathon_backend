# api/index.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

from pipeline import predict_pipeline  # your GPU-backed inference

app = FastAPI(title="CTAI - CTD Hackathon Backend")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ctai-ctd-hackathon-frontend.vercel.app",  # Vercel frontend
        "http://localhost:3000",                           # local dev
    ],
    allow_credentials=False,             # True only if using cookies/auth
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[],
    max_age=600,                         # cache preflight for 10 minutes
)

# -----------------------------
# Schemas
# -----------------------------
class InputData(BaseModel):
    description: str
    uom: Optional[str] = None
    core_market: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    # Pass the last prediction result from the frontend verbatim
    context: Optional[Dict[str, Any]] = None
    # Optional: include the original inputs (description/uom/core_market)
    inputs: Optional[Dict[str, Any]] = None

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
        data.description, data.uom, data.core_market
    )
    return {
        "MasterItemNo": master_item,
        "QtyShipped": qty,
        "uom": data.uom,
        "core_market": data.core_market,
    }

# -----------------------------
# Chatbot endpoint (explains prediction)
# -----------------------------
_openai_client = None  # lazy init so container can start without key

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Explainer bot grounded by the latest prediction.
    If OPENAI_API_KEY is missing, returns a rule-based fallback answer.
    """
    ctx = req.context or {}
    inp = req.inputs or {}

    system = (
        "You are a helpful procurement assistant for a construction/material forecasting app. "
        "Explain predictions clearly and concisely. If the user asks for sourcing, timelines, "
        "or confidence, give practical guidance. If you don't know, say so briefly.\n\n"
        "Context fields:\n"
        "- MasterItemNo: Predicted catalog ID for the item.\n"
        "- QtyShipped: Forecasted quantity shipped for the upcoming need.\n"
        "- uom/core_market: Optional inputs provided by the user.\n"
    )

    grounding = (
        f"Latest prediction:\n"
        f"- MasterItemNo: {ctx.get('MasterItemNo', '—')}\n"
        f"- QtyShipped: {ctx.get('QtyShipped', '—')}\n"
        f"- uom: {ctx.get('uom', '—')}\n"
        f"- core_market: {ctx.get('core_market', '—')}\n"
        f"Original input:\n"
        f"- description: {inp.get('description', '—')}\n"
    )

    # No API key? Provide a concise rule-based explanation.
    if not os.getenv("OPENAI_API_KEY"):
        base = (
            "Here’s what I can tell from the current prediction.\n\n"
            f"{grounding}\n"
            "Interpretation:\n"
            "- The MasterItemNo is the best-matching catalog item for your description.\n"
            "- QtyShipped is the forecasted quantity based on historical patterns and features.\n"
            "Tip: Provide UOM and Core Market to refine classification when available."
        )
        return {"answer": base}

    # With OpenAI (server-side, key stays private)
    try:
        client = _get_openai_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": grounding + "\nUser question: " + req.message},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        return {
            "answer": "I hit an error talking to the language model. "
                      "Basic interpretation:\n\n" + grounding,
            "error": str(e),
        }

# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.index:app", host="0.0.0.0", port=port)
