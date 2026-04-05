

from typing import Any, List, Literal
import math
import os
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import get_db

# Retail tables
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.fish_training_price import FishTrainingPrice
from app.models.prediction_result import PredictionResult

# Wholesale tables
from app.models.wholesale_actual_price import WholesaleActualPrice
from app.models.wholesale_training_price import WholesaleTrainingPrice
from app.models.wholesale_prediction_result import WholesalePredictionResult

from app.services.system_status import (
    read_status,
)

from app.services.prediction_service import generate_next_week_predictions_with_saved_hybrid

router = APIRouter(prefix="/admin/pipeline", tags=["admin-pipeline"])

UPLOAD_DIR = "uploads"
MODEL_DIR = "models_store"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def normalize_price_type(price_type: str | None) -> str:
    value = (price_type or "retail").strip().lower()
    if value not in {"retail", "wholesale"}:
        raise HTTPException(status_code=400, detail="priceType must be 'retail' or 'wholesale'")
    return value


def get_status_key(base: str, price_type: str) -> str:
    return f"{price_type}_{base}"


def get_table_classes(price_type: str):
    if price_type == "retail":
        return {
            "actual": FishWeeklyPrice,
            "training": FishTrainingPrice,
            "predicted": PredictionResult,
        }

    return {
        "actual": WholesaleActualPrice,
        "training": WholesaleTrainingPrice,
        "predicted": WholesalePredictionResult,
        
    }


@router.get("/summary")
def get_pipeline_summary(
    priceType: str = Query("retail"),
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    price_type = normalize_price_type(priceType)
    status = read_status()

    tables = get_table_classes(price_type)
    ActualTable = tables["actual"]
    PredictedTable = tables["predicted"]

    latest_actual = (
        db.query(ActualTable)
        .order_by(desc(ActualTable.year), desc(ActualTable.month), desc(ActualTable.week))
        .first()
    )

    latest_pred = (
        db.query(PredictedTable)
        .order_by(desc(PredictedTable.created_at))
        .first()
    )

    latest_5 = (
        db.query(PredictedTable)
        .order_by(desc(PredictedTable.created_at), desc(PredictedTable.id))
        .limit(5)
        .all()
    )

    latest_actual_week = "—"
    if latest_actual:
        latest_actual_week = f"{latest_actual.month} {latest_actual.year} - Week {latest_actual.week}"

    latest_week = status.get(get_status_key("latestPredictionWeek", price_type)) or (
        latest_pred.week_label if latest_pred else "—"
    )

    latest_published = bool(latest_pred.is_published) if latest_pred else False
    last_upload_date = status.get(get_status_key("lastUploadDate", price_type), "—")

    return {
        "priceType": price_type,
        "latestWeek": latest_week,
        "latestPublished": latest_published,
        "latestActualWeek": latest_actual_week,
        "lastUploadDate": last_upload_date,
        "latest5Predictions": [
            {
                "id": row.id,
                "commonName": row.common_name,
                "week": row.week_label,
                "predictedPrice": float(row.predicted_price) if row.predicted_price is not None else None,
                "modelName": row.model_name,
                "isPublished": row.is_published,
                "status": "Published" if row.is_published else "Not Published",
            }
            for row in latest_5
        ],
    }
