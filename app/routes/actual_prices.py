from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.wholesale_actual_price import WholesaleActualPrice

router = APIRouter(tags=["actual-prices"])

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


def build_week_label(year: Optional[int], month: Optional[str], week: Optional[int]) -> Optional[str]:
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


def get_actual_price_filter_options_from_model(model_class, db: Session):
    rows = db.query(model_class.year, model_class.month, model_class.week).all()

    years = sorted({r.year for r in rows if r.year is not None})
    months = sorted(
        {r.month for r in rows if r.month},
        key=lambda m: month_name_to_number(m) or 99,
    )
    weeks = sorted({r.week for r in rows if r.week is not None})

    return {
        "years": years,
        "months": months,
        "weeks": weeks,
    }


def get_actual_price_rows_from_model(
    model_class,
    db: Session,
    search: Optional[str] = None,
    year: Optional[str] = None,
    month: Optional[str] = None,
    week: Optional[str] = None,
    limit: Optional[int] = None,
):
    q = db.query(model_class)

    if search:
        s = f"%{search.strip()}%"
        q = q.filter(
            or_(
                model_class.sinhala_name.ilike(s),
                model_class.common_name.ilike(s),
            )
        )

    if year:
        q = q.filter(model_class.year == int(year))

    if month:
        q = q.filter(model_class.month == month)

    if week:
        q = q.filter(model_class.week == int(week))

    rows = q.all()

    rows = sorted(
        rows,
        key=lambda r: (
            r.year if r.year is not None else -1,
            month_name_to_number(r.month) or -1,
            r.week if r.week is not None else -1,
            r.id if getattr(r, "id", None) is not None else -1,
        ),
        reverse=True,
    )

    if limit:
        rows = rows[:limit]

    return {
        "rows": [
            {
                "id": row.id,
                "Sinhala Name": row.sinhala_name,
                "Common Name": row.common_name,
                "Year": row.year,
                "Month": row.month,
                "Week": row.week,
                "Week_Label": build_week_label(row.year, row.month, row.week),
                "Actual_Price": float(row.price) if row.price is not None else None,
                "Recorded_At": row.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if getattr(row, "created_at", None)
                else None,
            }
            for row in rows
        ]
    }


# Retail actual prices

@router.get("/actual-prices/retail/filter-options")
def get_retail_actual_price_filter_options(db: Session = Depends(get_db)):
    return get_actual_price_filter_options_from_model(FishWeeklyPrice, db)


@router.get("/actual-prices/retail/latest")
def get_retail_latest_actual_prices(
    search: Optional[str] = Query(None),
    year: Optional[str] = Query(None),
    month: Optional[str] = Query(None),
    week: Optional[str] = Query(None),
    limit: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    return get_actual_price_rows_from_model(
        model_class=FishWeeklyPrice,
        db=db,
        search=search,
        year=year,
        month=month,
        week=week,
        limit=limit,
    )



# Wholesale actual prices

@router.get("/actual-prices/wholesale/filter-options")
def get_wholesale_actual_price_filter_options(db: Session = Depends(get_db)):
    return get_actual_price_filter_options_from_model(WholesaleActualPrice, db)


@router.get("/actual-prices/wholesale/latest")
def get_wholesale_latest_actual_prices(
    search: Optional[str] = Query(None),
    year: Optional[str] = Query(None),
    month: Optional[str] = Query(None),
    week: Optional[str] = Query(None),
    limit: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    return get_actual_price_rows_from_model(
        model_class=WholesaleActualPrice,
        db=db,
        search=search,
        year=year,
        month=month,
        week=week,
        limit=limit,
    )