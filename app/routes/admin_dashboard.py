from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.core.security import get_current_admin
from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.prediction_result import PredictionResult
from app.models.feedback import Feedback
from app.models.wholesale_actual_price import WholesaleActualPrice
from app.models.wholesale_prediction_result import WholesalePredictionResult
from app.services.system_status import read_status

router = APIRouter(tags=["admin-dashboard"])


def build_market_stats(
    db: Session,
    actual_model,
    prediction_model,
    upload_date_value: str = "—",
):
    fish_count = (
        db.query(func.count(func.distinct(actual_model.common_name))).scalar() or 0
    )

    published_prediction_count = (
        db.query(func.count(prediction_model.id))
        .filter(prediction_model.is_published == True)
        .scalar()
        or 0
    )

    latest_actual = (
        db.query(actual_model)
        .order_by(
            desc(actual_model.year),
            desc(actual_model.month),
            desc(actual_model.week),
        )
        .first()
    )

    latest_prediction = (
        db.query(prediction_model)
        .filter(prediction_model.is_published == True)
        .order_by(desc(prediction_model.published_at))
        .first()
    )

    latest_actual_week = "—"
    if latest_actual:
        latest_actual_week = (
            f"{latest_actual.month} {latest_actual.year} - Week {latest_actual.week}"
        )

    latest_prediction_week = "—"
    if latest_prediction:
        latest_prediction_week = latest_prediction.week_label or "—"

    return {
        "fishCount": int(fish_count),
        "publishedPredictionCount": int(published_prediction_count),
        "latestActualWeek": latest_actual_week,
        "latestPredictionWeek": latest_prediction_week,
        "lastUploadDate": upload_date_value or "—",
    }


@router.get("/admin/dashboard/stats")
def get_dashboard_stats(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()

    total_feedback = db.query(func.count(Feedback.id)).scalar() or 0

    retail_stats = build_market_stats(
        db=db,
        actual_model=FishWeeklyPrice,
        prediction_model=PredictionResult,
        upload_date_value=status.get("lastUploadDate", "—"),
    )

    wholesale_stats = build_market_stats(
        db=db,
        actual_model=WholesaleActualPrice,
        prediction_model=WholesalePredictionResult,
        upload_date_value=status.get("wholesaleLastUploadDate", "—"),
    )

    return {
        "totalFeedback": int(total_feedback),
        "retail": retail_stats,
        "wholesale": wholesale_stats,
    }


@router.get("/admin/pipeline/activity-logs")
def get_pipeline_activity_logs(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()

    rows = []

    retail_upload_date = status.get("lastUploadDate")
    if retail_upload_date:
        rows.append(
            {
                "id": 1,
                "date": retail_upload_date,
                "action": "Retail dataset upload",
                "status": "Completed",
                "notes": status.get("lastLongFilename", "Retail data uploaded"),
            }
        )

    wholesale_upload_date = status.get("wholesaleLastUploadDate")
    if wholesale_upload_date:
        rows.append(
            {
                "id": 2,
                "date": wholesale_upload_date,
                "action": "Wholesale dataset upload",
                "status": "Completed",
                "notes": status.get("wholesale_lastLongFilename", "Wholesale data uploaded"),
            }
        )

    latest_retail_prediction = (
        db.query(PredictionResult)
        .filter(PredictionResult.batch_id.isnot(None))
        .order_by(desc(PredictionResult.created_at))
        .first()
    )

    if latest_retail_prediction:
        rows.append(
            {
                "id": 3,
                "date": latest_retail_prediction.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if latest_retail_prediction.created_at
                else "—",
                "action": "Retail prediction batch generated",
                "status": "Completed",
                "notes": latest_retail_prediction.batch_id or "Retail prediction batch created",
            }
        )

    latest_wholesale_prediction = (
        db.query(WholesalePredictionResult)
        .filter(WholesalePredictionResult.batch_id.isnot(None))
        .order_by(desc(WholesalePredictionResult.created_at))
        .first()
    )

    if latest_wholesale_prediction:
        rows.append(
            {
                "id": 4,
                "date": latest_wholesale_prediction.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if latest_wholesale_prediction.created_at
                else "—",
                "action": "Wholesale prediction batch generated",
                "status": "Completed",
                "notes": latest_wholesale_prediction.batch_id or "Wholesale prediction batch created",
            }
        )

    latest_feedback = (
        db.query(Feedback)
        .order_by(desc(Feedback.id))
        .first()
    )

    if latest_feedback:
        rows.append(
            {
                "id": 5,
                "date": getattr(latest_feedback, "created_at", None).strftime("%Y-%m-%d %H:%M:%S")
                if getattr(latest_feedback, "created_at", None)
                else "—",
                "action": "User feedback received",
                "status": "Completed",
                "notes": "New feedback record added",
            }
        )

    def parse_date(x):
        try:
            return x["date"] or ""
        except Exception:
            return ""

    rows = sorted(rows, key=parse_date, reverse=True)

    return {
        "rowCount": len(rows),
        "rows": rows,
    }