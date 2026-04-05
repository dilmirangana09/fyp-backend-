from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.core.security import get_current_admin
from app.models.prediction_result import PredictionResult
from app.models.wholesale_prediction_result import WholesalePredictionResult

router = APIRouter(prefix="/admin/prediction-history", tags=["admin-prediction-history"])


def get_model_by_market(market: str):
    market = market.strip().lower()
    if market == "retail":
        return PredictionResult
    if market == "wholesale":
        return WholesalePredictionResult
    raise HTTPException(status_code=400, detail="Invalid market. Use 'retail' or 'wholesale'.")


@router.get("/{market}")
def get_prediction_history(
    market: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    rows = (
        db.query(
            model.batch_id.label("batch_id"),
            func.max(model.model_name).label("model_name"),
            func.max(model.week_label).label("week_label"),
            func.max(model.year).label("year"),
            func.max(model.month).label("month"),
            func.max(model.week).label("week"),
            func.count(model.id).label("row_count"),
            func.max(model.is_published).label("is_published"),
            func.max(model.source_prediction_file).label("source_prediction_file"),
            func.max(model.created_at).label("created_at"),
        )
        .filter(model.batch_id.isnot(None))
        .group_by(model.batch_id)
        .order_by(func.max(model.created_at).desc())
        .all()
    )

    return {
        "market": market,
        "rowCount": len(rows),
        "rows": [
            {
                "batchId": row.batch_id,
                "modelName": row.model_name,
                "weekLabel": row.week_label,
                "year": row.year,
                "month": row.month,
                "week": row.week,
                "rowCount": int(row.row_count or 0),
                "isPublished": bool(row.is_published),
                "sourcePredictionFile": row.source_prediction_file,
                "createdAt": row.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if row.created_at
                else None,
            }
            for row in rows
        ],
    }


@router.get("/{market}/{batch_id}")
def get_prediction_history_details(
    market: str,
    batch_id: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    rows = (
        db.query(model)
        .filter(model.batch_id == batch_id)
        .order_by(model.common_name.asc())
        .all()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Prediction batch not found")

    first = rows[0]

    return {
        "market": market,
        "batchId": batch_id,
        "modelName": first.model_name,
        "weekLabel": first.week_label,
        "isPublished": first.is_published,
        "createdAt": first.created_at.strftime("%Y-%m-%d %H:%M:%S")
        if first.created_at
        else None,
        "rows": [
            {
                "id": row.id,
                "sinhalaName": row.sinhala_name,
                "commonName": row.common_name,
                "year": row.year,
                "month": row.month,
                "week": row.week,
                "weekLabel": row.week_label,
                "predictedPrice": float(row.predicted_price) if row.predicted_price is not None else None,
                "sourceLongFile": row.source_long_file,
                "sourcePredictionFile": row.source_prediction_file,
            }
            for row in rows
        ],
    }


@router.post("/{market}/{batch_id}/publish")
def publish_prediction_batch(
    market: str,
    batch_id: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    exists = (
        db.query(model)
        .filter(model.batch_id == batch_id)
        .first()
    )

    if not exists:
        raise HTTPException(status_code=404, detail="Prediction batch not found")

    db.query(model).update(
        {model.is_published: False},
        synchronize_session=False,
    )

    db.query(model).filter(
        model.batch_id == batch_id
    ).update(
        {model.is_published: True},
        synchronize_session=False,
    )

    db.commit()

    return {
        "message": f"{market.capitalize()} prediction batch published successfully",
        "market": market,
        "batchId": batch_id,
    }


@router.delete("/{market}/{batch_id}")
def delete_prediction_batch(
    market: str,
    batch_id: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    q = db.query(model).filter(model.batch_id == batch_id)
    count = q.count()

    if count == 0:
        raise HTTPException(status_code=404, detail="Prediction batch not found")

    q.delete(synchronize_session=False)
    db.commit()

    return {
        "message": f"{market.capitalize()} prediction batch deleted successfully",
        "market": market,
        "batchId": batch_id,
        "deletedRows": count,
    }