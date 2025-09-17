import os
import numpy as np
import joblib

# Path to HF-downloaded models (Dockerfile sets this up)
MODEL_DIR = os.environ.get("HF_CACHE_DIR", "/app/models_cache")

# ----------------------------
# Load TF-IDF vocab + idf (GPU-exported)
# ----------------------------
def load_tfidf_npz(path):
    """Load TF-IDF vocab/idf exported from cuML."""
    npz = np.load(path, allow_pickle=True)
    vocab = dict(zip(npz["toks"], npz["ids"]))
    idf = npz["idf"] if "idf" in npz.files else None
    return vocab, idf

tfidf_word_vocab, tfidf_word_idf = load_tfidf_npz(
    os.path.join(MODEL_DIR, "tfidf_word_cuml_v1.npz")
)
tfidf_char_vocab, tfidf_char_idf = load_tfidf_npz(
    os.path.join(MODEL_DIR, "tfidf_char_cuml_v1.npz")
)

# ----------------------------
# Load cuML LR/NB params
# ----------------------------
lr_params = np.load(os.path.join(MODEL_DIR, "cuml_lr_params_v1.npz"), allow_pickle=True)
nb_params = np.load(os.path.join(MODEL_DIR, "cuml_nb_params_v1.npz"), allow_pickle=True)

# ----------------------------
# Load regressors (LightGBM + XGBoost)
# ----------------------------
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.txt")
XGB_MODEL_PATH  = os.path.join(MODEL_DIR, "xgb_model.json")

HAS_LGB = os.path.exists(LGBM_MODEL_PATH)
HAS_XGB = os.path.exists(XGB_MODEL_PATH)

if HAS_LGB:
    import lightgbm as lgb
    lgbm = lgb.Booster(model_file=LGBM_MODEL_PATH)
else:
    lgbm = None

if HAS_XGB:
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(XGB_MODEL_PATH)
else:
    bst = None

# ----------------------------
# Load priors / memorizer maps
# ----------------------------
ratio_item_uom = joblib.load(os.path.join(MODEL_DIR, "ratio_item_uom.pkl"))
ratio_item     = joblib.load(os.path.join(MODEL_DIR, "ratio_item.pkl"))
ratio_global   = joblib.load(os.path.join(MODEL_DIR, "ratio_global.pkl"))
item_qty_median= joblib.load(os.path.join(MODEL_DIR, "item_qty_median.pkl"))
global_mode_int= joblib.load(os.path.join(MODEL_DIR, "global_mode_int.pkl"))

mem_duc = joblib.load(os.path.join(MODEL_DIR, "mem_duc.pkl"))
mem_du  = joblib.load(os.path.join(MODEL_DIR, "mem_du.pkl"))
mem_d   = joblib.load(os.path.join(MODEL_DIR, "mem_d.pkl"))

# ----------------------------
# Simple predict pipeline stub
# ----------------------------
def predict_pipeline(description: str, uom: str = None, core_market: str = None):
    """
    This is a lightweight stub that demonstrates
    how you'd call your trained models with the GPU artifacts.
    Replace this with your full preprocessing + inference.
    """

    # NOTE: At the moment we just return dummy values
    # because the full cuML vectorizer / LR/NB rebuild
    # is GPU-only. On Render (CPU-only), you’d normally
    # fallback to a CPU-mirror model.
    #
    # Since you uploaded *only GPU artifacts*, here we show
    # how to use priors as a placeholder.

    desc_key = description.lower().strip()
    uom_key  = str(uom or "").upper()
    cm_key   = str(core_market or "").upper()

    # Try memorizer maps
    master_item = (
        mem_duc.get((desc_key, uom_key, cm_key))
        or mem_du.get((desc_key, uom_key))
        or mem_d.get(desc_key)
    )
    if master_item is None:
        master_item = int(global_mode_int)

    # Use priors for qty
    qty = ratio_item.get(str(master_item), ratio_global)
    qty = float(qty) if qty is not None else float(item_qty_median.get(master_item, 1.0))

    return int(master_item), qty
