# backend/pipeline.py
import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack  # CPU sparse hstack

# ------------------------------------------------------------
# Version guard (prevents InconsistentVersionWarning at import)
# ------------------------------------------------------------
try:
    import sklearn
    if sklearn.__version__ != "1.7.2":
        warnings.warn(
            f"[pipeline] scikit-learn {sklearn.__version__} detected; "
            "artifacts were saved with 1.7.2. Consider `pip install scikit-learn==1.7.2` "
            "to avoid compatibility issues."
        )
except Exception:
    pass

# ------------------- Paths -------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ------------------- Helpers (match training) -------------------
_ws_rx = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").lower().strip()
    return _ws_rx.sub(" ", s)

def _tok(x: str) -> str:
    x = "" if x is None else str(x).strip().lower()
    return _ws_rx.sub("_", x)

def _build_itemdesc_meta(description: str, uom: str | None, core_market: str | None) -> str:
    """
    EXACTLY the same token shape used in training:
      normalized base + optional tokens + __memo_tier=none
    """
    base = _normalize_text(description or "")
    parts = [base]
    if uom:
        parts.append(f" __uom={_tok(uom)}")
    if core_market:
        parts.append(f" __core_market={_tok(core_market)}")
    parts.append(" __memo_tier=none")  # train had "none"
    return "".join(parts)

# ------------------- Load artifacts (CPU) -------------------
# TF-IDF + classifiers + label encoder (all sklearn)
tfidf_word = joblib.load(os.path.join(MODEL_DIR, "tfidf_word_sklearn.pkl"))
tfidf_char = joblib.load(os.path.join(MODEL_DIR, "tfidf_char_sklearn.pkl"))
lr         = joblib.load(os.path.join(MODEL_DIR, "lr_model_sklearn.pkl"))
nb         = joblib.load(os.path.join(MODEL_DIR, "nb_model_sklearn.pkl"))
le         = joblib.load(os.path.join(MODEL_DIR, "label_encoder_sklearn.pkl"))

# Priors / maps
ratio_item_uom  = joblib.load(os.path.join(MODEL_DIR, "ratio_item_uom.pkl"))
ratio_item      = joblib.load(os.path.join(MODEL_DIR, "ratio_item.pkl"))
ratio_global    = joblib.load(os.path.join(MODEL_DIR, "ratio_global.pkl"))
item_qty_median = joblib.load(os.path.join(MODEL_DIR, "item_qty_median.pkl"))
global_mode_int = joblib.load(os.path.join(MODEL_DIR, "global_mode_int.pkl"))

# Optional: cat_maps exist but aren’t needed for the current “priors-only” qty path
# cat_maps = joblib.load(os.path.join(MODEL_DIR, "cat_maps.pkl"))

# NOTE: We don’t import LightGBM/XGBoost here to keep inference light & portable.
# If you later want full regression features, you can lazy-load and wire them.

# ------------------- Inference -------------------
def predict_pipeline(description: str,
                     uom: str | None = None,
                     core_market: str | None = None):
    """
    Returns (MasterItemNo:int, QtyShipped:float)
    - Classification: TF-IDF -> (0.7 * LR + 0.3 * NB)
    - Quantity: priors (item+uom -> item -> global) — fast & robust
    """
    # 1) Build meta-augmented text exactly as in training
    meta_text = _build_itemdesc_meta(description, uom, core_market)

    # 2) Vectorize
    Xw = tfidf_word.transform([meta_text])
    Xc = tfidf_char.transform([meta_text])
    Xt = hstack([Xw, Xc], format="csr")

    # 3) Classification ensemble
    proba_lr = lr.predict_proba(Xt)
    proba_nb = nb.predict_proba(Xt)
    proba = 0.7 * proba_lr + 0.3 * proba_nb
    pred_idx = int(np.argmax(proba, axis=1)[0])

    # The label encoder was fit on MasterItemNo strings during training
    pred_master_str = le.inverse_transform([pred_idx])[0]
    try:
        master_item = int(float(pred_master_str))
    except Exception:
        # Fallback: use global mode if the label is somehow non-numeric
        master_item = int(global_mode_int)

    # 4) Quantity via priors (fast)
    # Prefer (item, uom) prior if UOM is present, otherwise item, else global.
    ratio = None
    if uom:
        ratio = ratio_item_uom.get((str(master_item), str(uom)))
    if ratio is None or not np.isfinite(ratio):
        ratio = ratio_item.get(str(master_item))
    if ratio is None or not np.isfinite(ratio):
        ratio = float(ratio_global)

    qty = float(ratio) * 1.0  # If you add ExtendedQuantity in the request, multiply here.

    if not np.isfinite(qty) or qty <= 0:
        qty = float(item_qty_median.get(str(master_item), 1.0))

    return master_item, qty