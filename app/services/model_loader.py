
# import joblib
# from pathlib import Path
# from typing import Literal
# import keras
# import json
# import tensorflow as tf
# import keras
# import numpy as np

# PRICE_TYPE = Literal["retail", "wholesale"]

# BASE_DIR = Path(__file__).resolve().parent.parent
# ML_MODELS_DIR = BASE_DIR / "ml_models"


# # Normalize price type

# def normalize_price_type(price_type: PRICE_TYPE | str | None) -> PRICE_TYPE:
#     value = (price_type or "retail").strip().lower()
#     if value not in {"retail", "wholesale"}:
#         raise ValueError("price_type must be 'retail' or 'wholesale'")
#     return value  # type: ignore



# # Get model paths (Retail / Wholesale)

# def get_model_paths(price_type: PRICE_TYPE | str):
#     price_type = normalize_price_type(price_type)

#     model_dir = ML_MODELS_DIR / price_type / "deployed"

#     ann_path = model_dir / "ann_xgb_hybrid_ann_model.keras"
#     xgb_path = model_dir / "ann_xgb_hybrid_xgb_model.pkl"
#     meta_path = model_dir / "ann_xgb_hybrid_metadata.json"

#     return ann_path, xgb_path, meta_path



# # Safe ANN loader

# def load_ann_model_safe(model_path: Path):
#     try:
#         return keras.models.load_model(model_path, compile=False)
#     except Exception:
#         print("Standard load failed, trying compatibility fix...")

#         import sys
#         sys.modules["keras.src.models.functional"] = keras.models

#         return keras.models.load_model(model_path, compile=False)



# # Main loader

# def load_ann_xgb_hybrid(price_type: PRICE_TYPE = "retail"):
#     price_type = normalize_price_type(price_type)

#     ann_path, xgb_path, meta_path = get_model_paths(price_type)

#     if not ann_path.exists():
#         raise FileNotFoundError(f"ANN model not found: {ann_path}")

#     if not xgb_path.exists():
#         raise FileNotFoundError(f"XGB model not found: {xgb_path}")

#     if not meta_path.exists():
#         raise FileNotFoundError(f"Metadata not found: {meta_path}")

#     print(f"Loading {price_type} model...")
#     print(f"ANN: {ann_path}")
#     print(f"XGB: {xgb_path}")

#     ann_model = load_ann_model_safe(ann_path)
#     xgb_model = joblib.load(xgb_path)

#     with open(meta_path, "r", encoding="utf-8") as f:
#         metadata = json.load(f)

#     print(" Model loaded successfully")

#     return ann_model, xgb_model, metadata


# # Debug helper

# def print_model_versions():
    

#     print("TensorFlow:", tf.__version__)
#     print("Keras:", keras.__version__)
#     print("NumPy:", np.__version__)

import json
import joblib
from pathlib import Path
from typing import Literal, cast

import tensorflow as tf
import numpy as np

PRICE_TYPE = Literal["retail", "wholesale"]

BASE_DIR = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = BASE_DIR / "ml_models"


# =========================================================
# Normalize price type
# =========================================================
def normalize_price_type(price_type: str | None) -> PRICE_TYPE:
    value = (price_type or "retail").strip().lower()
    if value not in {"retail", "wholesale"}:
        raise ValueError("price_type must be 'retail' or 'wholesale'")
    return cast(PRICE_TYPE, value)


# =========================================================
# Get model paths
# =========================================================
def get_model_paths(price_type: str | None):
    normalized_price_type = normalize_price_type(price_type)
    model_dir = ML_MODELS_DIR / normalized_price_type / "deployed"

    ann_path = model_dir / "ann_xgb_hybrid_ann_model.keras"
    xgb_path = model_dir / "ann_xgb_hybrid_xgb_model.pkl"
    meta_path = model_dir / "ann_xgb_hybrid_metadata.json"

    return normalized_price_type, model_dir, ann_path, xgb_path, meta_path


# =========================================================
# Safe ANN loader
# =========================================================
def load_ann_model_safe(model_path: Path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ANN model from {model_path}\n"
            f"Error: {e}"
        ) from e


# =========================================================
# Main loader
# =========================================================
def load_ann_xgb_hybrid(price_type: PRICE_TYPE = "retail"):
    normalized_price_type, model_dir, ann_path, xgb_path, meta_path = get_model_paths(price_type)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not ann_path.exists():
        raise FileNotFoundError(f"ANN model not found: {ann_path}")

    if not xgb_path.exists():
        raise FileNotFoundError(f"XGB model not found: {xgb_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    print(f"Loading {normalized_price_type} model...")
    print(f"ANN: {ann_path}")
    print(f"XGB: {xgb_path}")
    print(f"META: {meta_path}")

    ann_model = load_ann_model_safe(ann_path)

    try:
        xgb_model = joblib.load(xgb_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load XGBoost model from {xgb_path}\n"
            f"Error: {e}"
        ) from e

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load metadata from {meta_path}\n"
            f"Error: {e}"
        ) from e

    print("Model loaded successfully")

    return ann_model, xgb_model, metadata


# =========================================================
# Debug helper
# =========================================================
def print_model_versions():
    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("NumPy:", np.__version__)