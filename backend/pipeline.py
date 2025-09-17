import os
import joblib

# Path where Dockerfile downloads Hugging Face repo
MODEL_DIR = os.environ.get("HF_CACHE_DIR", "/app/models_cache")

# ----------------------------
# Load memorizer + priors
# ----------------------------
ratio_item_uom  = joblib.load(os.path.join(MODEL_DIR, "ratio_item_uom.pkl"))
ratio_item      = joblib.load(os.path.join(MODEL_DIR, "ratio_item.pkl"))
ratio_global    = joblib.load(os.path.join(MODEL_DIR, "ratio_global.pkl"))
item_qty_median = joblib.load(os.path.join(MODEL_DIR, "item_qty_median.pkl"))
global_mode_int = joblib.load(os.path.join(MODEL_DIR, "global_mode_int.pkl"))

mem_duc = joblib.load(os.path.join(MODEL_DIR, "mem_duc.pkl"))
mem_du  = joblib.load(os.path.join(MODEL_DIR, "mem_du.pkl"))
mem_d   = joblib.load(os.path.join(MODEL_DIR, "mem_d.pkl"))

# ----------------------------
# Prediction pipeline
# ----------------------------
def predict_pipeline(description: str, uom: str = None, core_market: str = None):
    """
    Lightweight inference using memorizer + priors.
    This avoids TF-IDF/LR/NB (GPU-only) so it runs on CPU hosts like Render.
    """

    desc_key = str(description or "").lower().strip()
    uom_key  = str(uom or "").upper()
    cm_key   = str(core_market or "").upper()

    # Try memorizer tiers (most specific → least)
    master_item = (
        mem_duc.get((desc_key, uom_key, cm_key))
        or mem_du.get((desc_key, uom_key))
        or mem_d.get(desc_key)
    )
    if master_item is None:
        master_item = int(global_mode_int)

    # Estimate quantity using priors
    qty = None
    if (str(master_item), uom_key) in ratio_item_uom:
        qty = ratio_item_uom[(str(master_item), uom_key)]
    elif str(master_item) in ratio_item:
        qty = ratio_item[str(master_item)]
    else:
        qty = ratio_global

    # Fallback to median qty if priors missing
    if qty is None:
        qty = float(item_qty_median.get(master_item, 1.0))

    return int(master_item), float(qty)
