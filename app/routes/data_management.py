from io import BytesIO
from datetime import datetime
import os

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Body, Query
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import get_db
from app.models.fish_weekly_price import FishWeeklyPrice
from app.models.wholesale_actual_price import WholesaleActualPrice
from app.services.system_status import read_status

router = APIRouter(prefix="/admin/actual-data", tags=["admin-actual-data"])

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

MONTH_NUM_TO_NAME = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def get_model_by_market(market: str):
    market = market.strip().lower()
    if market == "retail":
        return FishWeeklyPrice
    if market == "wholesale":
        return WholesaleActualPrice
    raise HTTPException(status_code=400, detail="Invalid market. Use 'retail' or 'wholesale'.")


def normalize_name(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def month_number_to_name(month_value):
    if month_value is None:
        return None
    try:
        if isinstance(month_value, str):
            cleaned = month_value.strip()
            if cleaned.isdigit():
                return MONTH_NUM_TO_NAME.get(int(cleaned), cleaned)
            lower = cleaned.lower()
            if lower in MONTH_NAME_TO_NUM:
                return MONTH_NUM_TO_NAME[MONTH_NAME_TO_NUM[lower]]
            return cleaned
        return MONTH_NUM_TO_NAME.get(int(month_value), str(month_value))
    except Exception:
        return str(month_value)


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


def normalize_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in {"sinhala name", "sinhala_name"}:
            rename_map[col] = "Sinhala Name"
        elif c in {"common name", "common_name", "fish", "fish name"}:
            rename_map[col] = "Common Name"
        elif c == "year":
            rename_map[col] = "Year"
        elif c == "month":
            rename_map[col] = "Month"
        elif c == "week":
            rename_map[col] = "Week"
        elif c in {"price", "actual price", "actual_price"}:
            rename_map[col] = "Price"

    df = df.rename(columns=rename_map)

    required_cols = ["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )

    df = df[required_cols].copy()
    df["Sinhala Name"] = df["Sinhala Name"].apply(normalize_name)
    df["Common Name"] = df["Common Name"].apply(normalize_name)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Month"] = df["Month"].apply(month_number_to_name)

    df = df.dropna(subset=["Common Name", "Year", "Month", "Week"]).reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)

    return df


def parse_upload_file(file: UploadFile, content: bytes) -> pd.DataFrame:
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        try:
            df = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), encoding="latin1")
        return normalize_uploaded_df(df)

    if filename.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(content), engine="openpyxl")
        return normalize_uploaded_df(df)

    if filename.endswith(".xls"):
        df = pd.read_excel(BytesIO(content))
        return normalize_uploaded_df(df)

    raise HTTPException(
        status_code=400,
        detail="Only .csv, .xlsx, and .xls files are supported"
    )


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            r.year if r.year is not None else -1,
            month_name_to_number(r.month) or -1,
            r.week if r.week is not None else -1,
            r.common_name.lower() if r.common_name else "",
        ),
        reverse=True,
    )


@router.get("/{market}/db-stats")
def get_db_stats(
    market: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    row_count = db.query(func.count(model.id)).scalar() or 0
    fish_count = db.query(
        func.count(
            func.distinct(
                func.concat(
                    model.sinhala_name,
                    " | ",
                    model.common_name,
                )
            )
        )
    ).scalar() or 0

    latest = None
    if hasattr(model, "updated_at"):
        latest = db.query(func.max(model.updated_at)).scalar()
    if latest is None and hasattr(model, "created_at"):
        latest = db.query(func.max(model.created_at)).scalar()

    status = read_status()
    if market == "retail":
        source_file = status.get("lastLongFilename") or "—"
    else:
        source_file = status.get("wholesale_lastLongFilename") or "—"

    return {
        "fishCount": int(fish_count),
        "rowCount": int(row_count),
        "lastUpdated": latest.strftime("%Y-%m-%d %H:%M:%S") if latest else "—",
        "sourceFile": source_file,
    }


@router.get("/{market}/list")
def list_db_rows(
    market: str,
    limit: int | None = Query(None),
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    rows = db.query(model).all()
    rows = sort_rows(rows)

    if limit is not None and limit > 0:
        rows = rows[:limit]

    return {
        "rows": [
            {
                "id": r.id,
                "sinhala_name": r.sinhala_name,
                "common_name": r.common_name,
                "year": r.year,
                "month": r.month,
                "week": r.week,
                "price": float(r.price) if r.price is not None else None,
            }
            for r in rows
        ]
    }


@router.get("/{market}/export")
def export_dataset(
    market: str,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    rows = db.query(model).all()
    rows = sorted(
        rows,
        key=lambda r: (
            r.year if r.year is not None else -1,
            month_name_to_number(r.month) or -1,
            r.week if r.week is not None else -1,
            r.common_name.lower() if r.common_name else "",
        )
    )

    data = [
        {
            "Sinhala Name": r.sinhala_name,
            "Common Name": r.common_name,
            "Year": r.year,
            "Month": r.month,
            "Week": r.week,
            "Price": float(r.price) if r.price is not None else None,
        }
        for r in rows
    ]

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    file_path = os.path.join(export_dir, f"{market}_actual_fish_prices_export.csv")
    pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8-sig")

    return FileResponse(file_path, filename=f"{market}_actual_fish_prices_export.csv")


@router.post("/{market}/upload")
async def upload_dataset(
    market: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    df = parse_upload_file(file, content)

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        sinhala_name = normalize_name(row.get("Sinhala Name"))
        common_name = normalize_name(row.get("Common Name"))
        year = int(row["Year"])
        month = str(row["Month"]).strip()
        week = int(row["Week"])
        price = row.get("Price")
        price = float(price) if pd.notna(price) else None

        existing = (
            db.query(model)
            .filter(
                model.common_name == common_name,
                model.year == year,
                model.month == month,
                model.week == week,
            )
            .first()
        )

        if existing:
            existing.sinhala_name = sinhala_name
            existing.common_name = common_name
            existing.price = price
            if hasattr(existing, "updated_at"):
                existing.updated_at = datetime.now()
            updated += 1
        else:
            db.add(
                model(
                    sinhala_name=sinhala_name,
                    common_name=common_name,
                    year=year,
                    month=month,
                    week=week,
                    price=price,
                )
            )
            inserted += 1

    db.commit()

    return {
        "message": f"{market.capitalize()} actual data uploaded successfully",
        "rowsReceived": int(len(df)),
        "inserted": inserted,
        "updated": updated,
    }


@router.delete("/{market}/delete")
def delete_selected_rows(
    market: str,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    ids = payload.get("ids", [])
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="No row ids provided")

    rows = db.query(model).filter(model.id.in_(ids)).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No matching rows found")

    deleted_count = len(rows)
    for row in rows:
        db.delete(row)

    db.commit()

    return {
        "message": f"Deleted {deleted_count} {market} row(s) successfully",
        "deleted": deleted_count,
    }


@router.put("/{market}/update/{row_id}")
def update_row_price(
    market: str,
    row_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    model = get_model_by_market(market)

    if "price" not in payload:
        raise HTTPException(status_code=400, detail="Price is required")

    try:
        price = float(payload["price"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid price value")

    row = db.query(model).filter(model.id == row_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Row not found")

    row.price = price
    if hasattr(row, "updated_at"):
        row.updated_at = datetime.now()

    db.commit()

    return {
        "message": f"{market.capitalize()} row updated successfully",
        "id": row_id,
        "price": price,
    }