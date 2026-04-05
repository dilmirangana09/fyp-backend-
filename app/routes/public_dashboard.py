from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.wholesale_actual_price import WholesaleActualPrice
from app.models.prediction_result import PredictionResult
from app.models.wholesale_prediction_result import WholesalePredictionResult

router = APIRouter(prefix="/public", tags=["public-dashboard"])

MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def month_name_to_number(month_value):
    if month_value is None:
        return None
    try:
        cleaned = str(month_value).strip().lower()
        if cleaned.isdigit():
            return int(cleaned)
        return MONTH_NAME_TO_NUM.get(cleaned)
    except Exception:
        return None


def build_week_label(year, month, week):
    if year is None or month is None or week is None:
        return None

    suffix = "th"
    if week == 1:
        suffix = "st"
    elif week == 2:
        suffix = "nd"
    elif week == 3:
        suffix = "rd"

    return f"{week}{suffix} week of {month} {year}"


def sort_actual_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            r.year if r.year is not None else -1,
            month_name_to_number(r.month) or -1,
            r.week if r.week is not None else -1,
            r.id if getattr(r, "id", None) is not None else -1,
        ),
        reverse=True,
    )


def sort_prediction_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            r.year if r.year is not None else -1,
            month_name_to_number(r.month) or -1,
            r.week if r.week is not None else -1,
            r.published_at.timestamp() if getattr(r, "published_at", None) else 0,
            r.id if getattr(r, "id", None) is not None else -1,
        ),
        reverse=True,
    )


def build_dashboard_overview(actual_model, prediction_model, db: Session):
    actual_rows_all = db.query(actual_model).all()
    actual_rows_all = sort_actual_rows(actual_rows_all)

    latest_actual_week = None
    actual_rows = []
    if actual_rows_all:
        latest_actual_week = (
            actual_rows_all[0].year,
            actual_rows_all[0].month,
            actual_rows_all[0].week,
        )
        actual_rows = [
            row for row in actual_rows_all
            if (row.year, row.month, row.week) == latest_actual_week
        ]

    published_prediction_rows_all = (
        db.query(prediction_model)
        .filter(prediction_model.is_published == True)
        .all()
    )
    published_prediction_rows_all = sort_prediction_rows(published_prediction_rows_all)

    latest_prediction_week = None
    latest_published_at = None
    predicted_rows = []
    if published_prediction_rows_all:
        latest_prediction_week = (
            published_prediction_rows_all[0].year,
            published_prediction_rows_all[0].month,
            published_prediction_rows_all[0].week,
        )
        latest_published_at = published_prediction_rows_all[0].published_at

        predicted_rows = [
            row for row in published_prediction_rows_all
            if (row.year, row.month, row.week) == latest_prediction_week
        ]

    actual_week_label = (
        build_week_label(*latest_actual_week) if latest_actual_week else None
    )
    predicted_week_label = (
        build_week_label(*latest_prediction_week) if latest_prediction_week else None
    )

    actual_preview = [
        {
            "id": row.id,
            "sinhalaName": row.sinhala_name,
            "commonName": row.common_name,
            "year": row.year,
            "month": row.month,
            "week": row.week,
            "actualPrice": float(row.price) if row.price is not None else None,
        }
        for row in actual_rows[:5]
    ]

    predicted_preview = [
        {
            "id": row.id,
            "sinhalaName": row.sinhala_name,
            "commonName": row.common_name,
            "year": row.year,
            "month": row.month,
            "week": row.week,
            "predictedPrice": float(row.predicted_price) if row.predicted_price is not None else None,
        }
        for row in predicted_rows[:5]
    ]

    return {
        "actualWeekLabel": actual_week_label,
        "predictedWeekLabel": predicted_week_label,
        "actualCount": len(actual_rows),
        "predictedCount": len(predicted_rows),
        "publishedAt": latest_published_at.strftime("%Y-%m-%d %H:%M:%S")
        if latest_published_at
        else None,
        "actualRows": actual_preview,
        "predictedRows": predicted_preview,
    }


@router.get("/dashboard-overview/retail")
def get_retail_dashboard_overview(db: Session = Depends(get_db)):
    return build_dashboard_overview(
        actual_model=FishWeeklyPrice,
        prediction_model=PredictionResult,
        db=db,
    )


@router.get("/dashboard-overview/wholesale")
def get_wholesale_dashboard_overview(db: Session = Depends(get_db)):
    return build_dashboard_overview(
        actual_model=WholesaleActualPrice,
        prediction_model=WholesalePredictionResult,
        db=db,
    )