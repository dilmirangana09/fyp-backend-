from datetime import datetime
from io import BytesIO
from typing import Any, List
import math
import os
import re
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import get_db

from app.models.wholesale_actual_price import WholesaleActualPrice
from app.models.wholesale_training_price import WholesaleTrainingPrice
from app.models.wholesale_prediction_result import WholesalePredictionResult
from app.models.upload_log import UploadLog
from app.models.pipeline_activity_log import PipelineActivityLog

from app.services.system_status import (
    read_status,
    update_fish_count,
    update_last_upload,
    write_status,
)
from app.services.training_service import get_deployed_model_info
from app.services.prediction_service import generate_next_week_predictions_with_saved_hybrid_df

router = APIRouter(prefix="/admin/wholesale-pipeline", tags=["admin-wholesale-pipeline"])

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


class ValidateRequest(BaseModel):
    filename: str
    rows: List[List[Any]]


class FinalizeUploadIn(BaseModel):
    filename: str
    rows: List[List[Any]]


class MergeRequest(BaseModel):
    filename: str
    rows: List[List[Any]]


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
        if isinstance(month_value, (int, float)) and not pd.isna(month_value):
            return int(month_value)
        cleaned = str(month_value).strip()
        if cleaned.isdigit():
            return int(cleaned)
        return MONTH_NAME_TO_NUM.get(cleaned.lower())
    except Exception:
        return None


def derive_week_label(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace("_", " ").strip()
    stem = re.sub(r"\(\d+\)$", "", stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    stem = re.sub(r"(?i)^wholesale\s+", "", stem).strip()
    return stem


def split_week_label(label: str):
    s = str(label).strip()
    s = re.sub(r"\s+", " ", s)

    m = re.match(r"(?i)^(\d+)(st|nd|rd|th)\s+week\s+of\s+([A-Za-z]+)\s+(\d{4})$", s)
    if not m:
        return None, None, None

    week = int(m.group(1))
    month_name = month_number_to_name(m.group(3))
    year = int(m.group(4))
    return year, month_name, week


def make_week_label(year: int, month: str, week: int) -> str:
    suffix = "th"
    if week == 1:
        suffix = "st"
    elif week == 2:
        suffix = "nd"
    elif week == 3:
        suffix = "rd"
    return f"{week}{suffix} week of {month} {year}"


def week_label_sort_key(label: str):
    year, month, week = split_week_label(label)
    month_num = month_name_to_number(month)
    if year is None or month_num is None or week is None:
        return (9999, 99, 99, str(label))
    return (year, month_num, week, str(label))


def normalize_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Sinhala Name" in df.columns:
        df["Sinhala Name"] = df["Sinhala Name"].astype(str).str.strip()
        df["Sinhala Name"] = df["Sinhala Name"].replace(["", "nan", "None", "<NA>"], pd.NA)

    if "Common Name" in df.columns:
        df["Common Name"] = df["Common Name"].astype(str).str.strip()
        df["Common Name"] = df["Common Name"].replace(["", "nan", "None", "<NA>"], pd.NA)

    return df


def normalize_weekly_upload_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = ["Sinhala Name", "Common Name", "Price"]
    if all(c in df.columns for c in expected_cols):
        df = df[expected_cols].copy()
    else:
        if df.shape[1] < 3:
            raise HTTPException(status_code=400, detail="File must contain at least 3 columns")
        df = df.iloc[:, :3].copy()
        df.columns = expected_cols

    df = normalize_name_columns(df)
    df = df[df["Sinhala Name"].notna() | df["Common Name"].notna()].reset_index(drop=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df


def read_uploaded_week_file_from_bytes(contents: bytes, filename: str) -> pd.DataFrame:
    lower_name = (filename or "").lower()

    if lower_name.endswith(".xlsx"):
        df = pd.read_excel(
            BytesIO(contents),
            sheet_name="Wholesale",
            skiprows=2,
            engine="openpyxl",
        )
        if df.shape[1] < 6:
            raise HTTPException(status_code=400, detail="XLSX file does not contain expected columns")

        df = df.iloc[:, [1, 2, 5]].copy()
        df.columns = ["Sinhala Name", "Common Name", "Price"]
        return normalize_weekly_upload_df(df)

    if lower_name.endswith(".csv"):
        try:
            df = pd.read_csv(BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(contents), encoding="latin1")
        return normalize_weekly_upload_df(df)

    raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are supported")


def dataframe_from_payload_rows(rows: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["Sinhala Name", "Common Name", "Price"])
    return normalize_weekly_upload_df(df)


def log_upload_action(
    db: Session,
    filename: str | None,
    week_label: str | None,
    fish_count: int | None,
    action: str,
    status: str,
):
    log = UploadLog(
        filename=filename,
        stored_filename=None,
        week_label=week_label,
        fish_count=fish_count,
        action=f"Wholesale - {action}",
        status=status,
    )
    db.add(log)
    db.commit()


def add_pipeline_log(db: Session, action: str, status: str, notes: str | None = None):
    log = PipelineActivityLog(
        action=f"Wholesale - {action}",
        status=status,
        notes=notes,
    )
    db.add(log)
    db.commit()


def upsert_weekly_prices_to_db(
    weekly_df: pd.DataFrame,
    year: int,
    month: str,
    week: int,
    db: Session,
):
    for _, row in weekly_df.iterrows():
        sinhala_name = str(row.get("Sinhala Name", "")).strip()
        common_name = str(row.get("Common Name", "")).strip()
        raw_price = row.get("Price")

        if not sinhala_name and not common_name:
            continue

        price = None
        if pd.notna(raw_price):
            try:
                price = float(raw_price)
            except Exception:
                price = None

        existing = (
            db.query(WholesaleActualPrice)
            .filter(
                WholesaleActualPrice.sinhala_name == sinhala_name,
                WholesaleActualPrice.common_name == common_name,
                WholesaleActualPrice.year == year,
                WholesaleActualPrice.month == month,
                WholesaleActualPrice.week == week,
            )
            .first()
        )

        if existing:
            existing.price = price
            existing.updated_at = datetime.now()
        else:
            db.add(
                WholesaleActualPrice(
                    sinhala_name=sinhala_name,
                    common_name=common_name,
                    year=year,
                    month=month,
                    week=week,
                    price=price,
                )
            )

    db.commit()


def fetch_actual_data_df(db: Session) -> pd.DataFrame:
    rows = db.query(WholesaleActualPrice).all()

    data = [
        {
            "Sinhala Name": r.sinhala_name,
            "Common Name": r.common_name,
            "Year": int(r.year) if r.year is not None else None,
            "Month": month_number_to_name(r.month),
            "Week": int(r.week) if r.week is not None else None,
            "Price": float(r.price) if r.price is not None else None,
        }
        for r in rows
    ]

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"])

    df = normalize_name_columns(df)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "Week"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)

    return df


def build_wide_from_actual_db(db: Session) -> pd.DataFrame:
    long_df = fetch_actual_data_df(db)

    if long_df.empty:
        return pd.DataFrame(columns=["Sinhala Name", "Common Name"])

    long_df["Week_Label"] = long_df.apply(
        lambda r: make_week_label(int(r["Year"]), str(r["Month"]), int(r["Week"])),
        axis=1
    )

    wide_df = long_df.pivot_table(
        index=["Sinhala Name", "Common Name"],
        columns="Week_Label",
        values="Price",
        aggfunc="first"
    ).reset_index()

    fixed = ["Sinhala Name", "Common Name"]
    week_cols = [c for c in wide_df.columns if c not in fixed]
    week_cols = sorted(week_cols, key=week_label_sort_key)

    return wide_df[fixed + week_cols].copy()


def filter_fish_with_50pct_rule(wide_df: pd.DataFrame):
    if wide_df.empty:
        return wide_df.copy(), pd.DataFrame()

    week_cols = [c for c in wide_df.columns if c not in ["Sinhala Name", "Common Name"]]
    if not week_cols:
        return wide_df.copy(), pd.DataFrame()

    work_df = wide_df.copy()
    work_df[week_cols] = work_df[week_cols].apply(pd.to_numeric, errors="coerce")

    available_counts = work_df[week_cols].notna().sum(axis=1)
    total_weeks = len(week_cols)
    threshold = math.ceil(total_weeks * 0.5)

    summary_df = work_df[["Sinhala Name", "Common Name"]].copy()
    summary_df["available_weeks"] = available_counts
    summary_df["total_weeks"] = total_weeks
    summary_df["availability_pct"] = ((available_counts / total_weeks) * 100).round(2)
    summary_df["kept"] = available_counts >= threshold

    filtered_df = work_df[available_counts >= threshold].reset_index(drop=True)
    return filtered_df, summary_df


def interpolate_wide_df(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if filtered_df.empty:
        return filtered_df.copy()

    week_cols = [c for c in filtered_df.columns if c not in ["Sinhala Name", "Common Name"]]
    if not week_cols:
        return filtered_df.copy()

    df = filtered_df.copy()
    df[week_cols] = df[week_cols].apply(pd.to_numeric, errors="coerce")
    df[week_cols] = df[week_cols].T.interpolate(method="linear", limit_direction="both").T
    df[week_cols] = df[week_cols].bfill(axis=1).ffill(axis=1)

    return df


def wide_to_long_df(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"])

    week_cols = [c for c in wide_df.columns if c not in ["Sinhala Name", "Common Name"]]
    long_df = wide_df.melt(
        id_vars=["Sinhala Name", "Common Name"],
        value_vars=week_cols,
        var_name="Week_Label",
        value_name="Price",
    )

    long_df[["Year", "Month", "Week"]] = long_df["Week_Label"].apply(lambda x: pd.Series(split_week_label(x)))
    long_df = long_df[["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]].copy()
    long_df["Price"] = pd.to_numeric(long_df["Price"], errors="coerce")
    long_df = long_df.dropna(subset=["Year", "Month", "Week", "Price"]).reset_index(drop=True)

    long_df["Year"] = long_df["Year"].astype(int)
    long_df["Week"] = long_df["Week"].astype(int)

    long_df["month_sort"] = long_df["Month"].map(month_name_to_number)
    long_df = long_df.sort_values(
        by=["Sinhala Name", "Common Name", "Year", "month_sort", "Week"]
    ).drop(columns=["month_sort"]).reset_index(drop=True)

    return long_df


def replace_training_data_from_long_df(long_df: pd.DataFrame, db: Session):
    db.query(WholesaleTrainingPrice).delete(synchronize_session=False)
    db.commit()

    if long_df.empty:
        return

    for _, row in long_df.iterrows():
        sinhala_name = str(row.get("Sinhala Name", "")).strip()
        common_name = str(row.get("Common Name", "")).strip()
        year = int(row.get("Year"))
        month = str(row.get("Month", "")).strip()
        week = int(row.get("Week"))
        price = row.get("Price")

        db.add(
            WholesaleTrainingPrice(
                sinhala_name=sinhala_name,
                common_name=common_name,
                year=year,
                month=month,
                week=week,
                price=float(price) if pd.notna(price) else None,
            )
        )

    db.commit()


def get_latest_actual_week_from_db(db: Session):
    rows = db.query(WholesaleActualPrice).all()
    if not rows:
        return "—"

    def row_key(r):
        return (
            int(r.year) if r.year is not None else -1,
            month_name_to_number(r.month) or -1,
            int(r.week) if r.week is not None else -1,
        )

    latest = max(rows, key=row_key)
    return f"{latest.month} {latest.year} - Week {latest.week}"


@router.post("/preview-file")
async def preview_file(
    file: UploadFile = File(...),
    admin=Depends(get_current_admin),
):
    name = (file.filename or "").lower()

    if not (name.endswith(".csv") or name.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only .csv or .xlsx files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        data = read_uploaded_week_file_from_bytes(contents, file.filename or "")
        rows = data.fillna("").values.tolist()

        return {
            "columns": ["Sinhala Name", "Common Name", "Price"],
            "rows": rows,
            "rowCount": int(len(rows)),
            "filename": file.filename,
            "pipeline": "wholesale",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preview failed: {str(e)}")


@router.post("/validate-csv")
def validate_weekly_csv(
    payload: ValidateRequest,
    admin=Depends(get_current_admin),
):
    df = dataframe_from_payload_rows(payload.rows)

    rows, cols = df.shape
    if rows == 0 or cols == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    fish_count = int(df["Common Name"].dropna().nunique()) if "Common Name" in df.columns else 0
    update_fish_count(fish_count)

    missing_total = int(df.isna().sum().sum())
    missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns}
    duplicate_rows = int(df.duplicated(subset=["Sinhala Name", "Common Name"]).sum())
    duplicate_columns = [c for c in df.columns[df.columns.duplicated()].tolist()]

    errors = []
    warnings = []

    expected = {"Sinhala Name", "Common Name", "Price"}
    if set(df.columns) != expected:
        warnings.append(f"Unexpected columns after extraction: {list(df.columns)}")

    if duplicate_columns:
        errors.append(f"Duplicate column names found: {duplicate_columns[:10]}")

    if fish_count == 0:
        warnings.append("No valid fish names were detected in Common Name column.")

    summary = "Valid " if len(errors) == 0 else "Has issues ⚠️"

    return {
        "summary": summary,
        "rows": int(rows),
        "columns": int(cols),
        "fileType": "XLSX" if str(payload.filename).lower().endswith(".xlsx") else "CSV",
        "fishCount": fish_count,
        "missingTotal": missing_total,
        "missingByColumn": missing_by_col,
        "duplicateRows": duplicate_rows,
        "duplicateColumns": duplicate_columns,
        "errors": errors,
        "warnings": warnings,
        "validatedBy": admin.get("email"),
        "filename": payload.filename,
        "pipeline": "wholesale",
    }


@router.post("/finalize-upload")
def finalize_upload(
    payload: FinalizeUploadIn,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    df = dataframe_from_payload_rows(payload.rows)

    if df.empty:
        raise HTTPException(status_code=400, detail="No rows available to process")

    update_last_upload(payload.filename)

    status = read_status()
    status["wholesale_lastUploadFilename"] = None
    status["wholesale_lastUploadDate"] = datetime.now().strftime("%Y-%m-%d")
    write_status(status)

    log_upload_action(
        db=db,
        filename=payload.filename,
        week_label=None,
        fish_count=int(df["Common Name"].dropna().nunique()) if "Common Name" in df.columns else int(len(df)),
        action="upload",
        status="success",
    )

    return {
        "message": "Wholesale file accepted successfully",
        "uploadedDate": datetime.now().strftime("%Y-%m-%d"),
        "rowCount": int(len(df)),
    }


@router.post("/preprocess-merge")
def preprocess_merge(
    payload: MergeRequest,
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    uploaded_df = dataframe_from_payload_rows(payload.rows)

    if uploaded_df.empty:
        raise HTTPException(status_code=400, detail="No valid rows available")

    week_label = derive_week_label(payload.filename)
    year, month, week = split_week_label(week_label)
    if year is None or month is None or week is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid weekly filename format: {week_label}. Expected format like '1st week of June 2025'"
        )

    upsert_weekly_prices_to_db(uploaded_df, year, month, week, db)

    wide_df = build_wide_from_actual_db(db)
    fish_count = int(wide_df["Common Name"].dropna().nunique()) if "Common Name" in wide_df.columns else 0

    status = read_status()
    status["wholesale_lastMergedFilename"] = None
    status["wholesale_lastWeekLabel"] = week_label
    write_status(status)

    log_upload_action(
        db=db,
        filename=payload.filename,
        week_label=week_label,
        fish_count=fish_count,
        action="merge",
        status="success",
    )

    return {
        "message": "Wholesale weekly data saved to DB and merged in memory",
        "weekLabel": week_label,
        "columns": list(wide_df.columns),
        "rows": wide_df.fillna("").head(100).values.tolist(),
        "rowCount": int(len(wide_df)),
        "dbSaved": True,
    }


@router.post("/preprocess-filter")
def preprocess_filter(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    wide_df = build_wide_from_actual_db(db)
    if wide_df.empty:
        raise HTTPException(status_code=404, detail="No wholesale actual data found in database")

    filtered_df, summary_df = filter_fish_with_50pct_rule(wide_df)

    status = read_status()
    status["wholesale_lastFilteredFilename"] = None
    status["wholesale_lastFilteredSummaryFilename"] = None
    write_status(status)

    kept_fish_count = int(filtered_df["Common Name"].dropna().nunique()) if "Common Name" in filtered_df.columns else 0
    week_cols = [c for c in filtered_df.columns if c not in ["Sinhala Name", "Common Name"]]
    threshold = math.ceil(len(week_cols) * 0.5) if week_cols else 0

    log_upload_action(
        db=db,
        filename=None,
        week_label=status.get("wholesale_lastWeekLabel"),
        fish_count=kept_fish_count,
        action="filter_50pct",
        status="success",
    )

    return {
        "message": "Wholesale 50% filter applied",
        "stats": {
            "totalWeeks": len(week_cols),
            "thresholdWeeks": threshold,
            "keptFishCount": kept_fish_count,
        },
        "summaryPreview": summary_df.fillna("").head(50).to_dict(orient="records"),
    }


@router.post("/preprocess-interpolate")
def preprocess_interpolate(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    wide_df = build_wide_from_actual_db(db)
    if wide_df.empty:
        raise HTTPException(status_code=404, detail="No wholesale actual data found in database")

    filtered_df, _ = filter_fish_with_50pct_rule(wide_df)
    interpolated_df = interpolate_wide_df(filtered_df)

    fish_count = int(interpolated_df["Common Name"].dropna().nunique()) if "Common Name" in interpolated_df.columns else 0
    update_fish_count(fish_count)

    status = read_status()
    status["wholesale_lastInterpolatedFilename"] = None
    write_status(status)

    week_cols = [c for c in interpolated_df.columns if c not in ["Sinhala Name", "Common Name"]]

    return {
        "message": "Wholesale interpolation completed",
        "stats": {
            "fishCount": fish_count,
            "weekCount": len(week_cols),
            "rowCount": int(len(interpolated_df)),
        },
        "preview": interpolated_df.head(20).fillna("").to_dict(orient="records"),
    }


@router.post("/preprocess-long-format")
def preprocess_long_format(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    wide_df = build_wide_from_actual_db(db)
    if wide_df.empty:
        raise HTTPException(status_code=404, detail="No wholesale actual data found in database")

    filtered_df, _ = filter_fish_with_50pct_rule(wide_df)
    interpolated_df = interpolate_wide_df(filtered_df)
    long_df = wide_to_long_df(interpolated_df)

    if long_df.empty:
        raise HTTPException(status_code=400, detail="Failed to generate wholesale long format dataset")

    status = read_status()
    status["wholesale_lastLongFilename"] = None
    status["wholesale_lastPreprocessDate"] = datetime.now().strftime("%Y-%m-%d")
    write_status(status)

    return {
        "message": "Wholesale long format dataset generated in memory",
        "stats": {
            "rowCount": int(len(long_df)),
            "fishCount": int(long_df["Common Name"].dropna().nunique()) if "Common Name" in long_df.columns else 0,
        },
        "preview": long_df.head(50).fillna("").to_dict(orient="records"),
    }


@router.post("/sync-long-to-db")
def sync_long_to_db(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    wide_df = build_wide_from_actual_db(db)
    if wide_df.empty:
        raise HTTPException(status_code=404, detail="No wholesale actual data found in database")

    filtered_df, _ = filter_fish_with_50pct_rule(wide_df)
    interpolated_df = interpolate_wide_df(filtered_df)
    long_df = wide_to_long_df(interpolated_df)

    if long_df.empty:
        raise HTTPException(status_code=400, detail="No valid rows found in long format dataset")

    replace_training_data_from_long_df(long_df, db)

    add_pipeline_log(
        db,
        action="Sync Long Format to Training DB",
        status="Completed",
        notes=f"{len(long_df)} rows synced from in-memory wholesale dataset",
    )

    return {
        "message": "Wholesale long format training dataset stored in DB successfully",
        "fishCount": int(long_df["Common Name"].dropna().nunique()),
        "rowCount": int(len(long_df)),
    }


@router.post("/train-hybrid-model")
def train_hybrid_model_route(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    try:
        result = get_deployed_model_info(price_type="wholesale")

        status = read_status()
        status["wholesale_lastModelStatus"] = "Ready"
        status["wholesale_lastModelName"] = result["modelName"]
        status["wholesale_lastModelTrainedAt"] = result["trainedAt"]
        status["wholesale_lastModelSource"] = "Colab"
        status["wholesale_lastModelWeights"] = result["bestWeights"]
        status["wholesale_lastModelAnnAlpha"] = result["annAlpha"]
        status["wholesale_lastModelMetrics"] = result.get("metrics", {})
        write_status(status)

        add_pipeline_log(
            db,
            action="Load Deployed Model",
            status="Completed",
            notes=f"Model ready: {result['modelName']}",
        )

        return {
            "message": "Wholesale deployed hybrid ANN + XGBoost model is ready",
            "finalModel": result["modelName"],
            "trainedAt": result["trainedAt"],
            "bestWeights": result["bestWeights"],
            "annAlpha": result["annAlpha"],
            "source": "Colab",
            "metrics": result.get("metrics", {}),
            "testSamples": "—",
            "files": result["files"],
        }

    except Exception as e:
        add_pipeline_log(
            db,
            action="Load Deployed Model",
            status="Failed",
            notes=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-next-week")
def predict_next_week(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    try:
        training_rows = db.query(WholesaleTrainingPrice).all()
        if not training_rows:
            raise HTTPException(
                status_code=404,
                detail="Wholesale training data not found in DB. Generate and sync long format first."
            )

        long_data = pd.DataFrame([
            {
                "Sinhala Name": r.sinhala_name,
                "Common Name": r.common_name,
                "Year": int(r.year),
                "Month": r.month,
                "Week": int(r.week),
                "Price": float(r.price) if r.price is not None else None,
            }
            for r in training_rows
        ])

        if long_data.empty:
            raise HTTPException(
                status_code=404,
                detail="Wholesale training data is empty."
            )

        long_data["month_sort"] = long_data["Month"].map(month_name_to_number)
        long_data = long_data.sort_values(
            by=["Sinhala Name", "Common Name", "Year", "month_sort", "Week"]
        ).drop(columns=["month_sort"]).reset_index(drop=True)

        # direct prediction from dataframe
        result = generate_next_week_predictions_with_saved_hybrid_df(
            long_data,
            price_type="wholesale",
        )

        preview = result.get("preview", [])
        output_week = result.get("date")
        row_count = int(result.get("rowCount", 0))
        final_model = result.get("modelName", "Wholesale_Hybrid_ANN_XGBoost")

        if not preview:
            raise HTTPException(
                status_code=500,
                detail="No prediction rows returned from prediction generator."
            )

        batch_id = uuid4().hex

        # remove old unpublished rows
        db.query(WholesalePredictionResult).filter(
            WholesalePredictionResult.is_published == False
        ).delete(synchronize_session=False)
        db.commit()

        for row in preview:
            row_year = row.get("Year")
            row_month = row.get("Month")
            row_week = row.get("Week")

            row_year = int(row_year) if row_year not in [None, ""] else None
            row_week = int(row_week) if row_week not in [None, ""] else None

            # if prediction service already returns month name, keep it
            if row_month not in [None, ""]:
                if isinstance(row_month, str):
                    month_name = row_month.strip()
                else:
                    month_name = month_number_to_name(row_month)
            else:
                month_name = None

            week_label = None
            if row_week and month_name and row_year:
                week_label = make_week_label(row_year, month_name, row_week)

            predicted_price = row.get("Predicted_Price")
            if predicted_price is None:
                predicted_price = row.get("Predicted Price")

            db.add(
                WholesalePredictionResult(
                    batch_id=batch_id,
                    model_name=final_model,
                    sinhala_name=row.get("Sinhala Name"),
                    common_name=row.get("Common Name"),
                    year=row_year,
                    month=month_name,
                    week=row_week,
                    week_label=week_label,
                    predicted_price=float(predicted_price) if predicted_price not in [None, ""] else None,
                    source_long_file=None,
                    source_prediction_file=None,
                    is_published=False,
                )
            )

        db.commit()

        add_pipeline_log(
            db,
            action="Generate Predictions",
            status="Completed",
            notes=f"{row_count} wholesale predictions saved to DB. Batch: {batch_id}",
        )

        status = read_status()
        status["wholesale_lastPredictionFilename"] = None
        status["wholesale_lastPredictionBatchId"] = batch_id
        status["wholesale_lastPredictionDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status["wholesale_lastModelName"] = final_model
        status["wholesale_latestPredictionWeek"] = output_week
        write_status(status)

        return {
            "message": "Wholesale next week predictions generated and saved to DB successfully",
            "date": output_week,
            "rowCount": row_count,
            "batchId": batch_id,
            "savedToDb": True,
            "preview": preview[:50],
            "modelName": final_model,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        add_pipeline_log(
            db,
            action="Generate Predictions",
            status="Failed",
            notes=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Wholesale prediction failed: {str(e)}")

@router.post("/publish-predictions")
def publish_predictions(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()
    batch_id = status.get("wholesale_lastPredictionBatchId")

    if not batch_id:
        raise HTTPException(status_code=404, detail="No wholesale prediction batch found to publish.")

    rows = db.query(WholesalePredictionResult).filter(WholesalePredictionResult.batch_id == batch_id).all()

    if not rows:
        raise HTTPException(status_code=404, detail="Wholesale prediction rows not found for latest batch.")

    db.query(WholesalePredictionResult).filter(WholesalePredictionResult.is_published == True).update(
        {"is_published": False, "published_at": None},
        synchronize_session=False
    )

    published_time = datetime.utcnow()
    for row in rows:
        row.is_published = True
        row.published_at = published_time

    db.commit()

    add_pipeline_log(
        db,
        action="Publish to User Pages",
        status="Completed",
        notes=f"Published wholesale batch: {batch_id} ({len(rows)} predictions)",
    )

    status["wholesale_lastPublishedBatchId"] = batch_id
    status["wholesale_lastPublishedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_status(status)

    return {
        "message": "Wholesale predictions published successfully",
        "batchId": batch_id,
        "publishedCount": len(rows),
        "publishedAt": published_time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/summary")
def get_pipeline_summary(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    status = read_status()

    latest_pred = (
        db.query(WholesalePredictionResult)
        .order_by(desc(WholesalePredictionResult.created_at), desc(WholesalePredictionResult.id))
        .first()
    )

    latest_5 = (
        db.query(WholesalePredictionResult)
        .order_by(desc(WholesalePredictionResult.created_at), desc(WholesalePredictionResult.id))
        .limit(5)
        .all()
    )

    latest_actual_week = get_latest_actual_week_from_db(db)

    latest_week = status.get("wholesale_latestPredictionWeek") or (
        latest_pred.week_label if latest_pred else "—"
    )

    latest_published = bool(latest_pred.is_published) if latest_pred else False
    last_upload_date = status.get("wholesale_lastUploadDate", "—")

    return {
        "pipeline": "wholesale",
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


@router.get("/activity-logs")
def get_activity_logs(
    db: Session = Depends(get_db),
    admin=Depends(get_current_admin),
):
    logs = (
        db.query(PipelineActivityLog)
        .order_by(desc(PipelineActivityLog.created_at))
        .limit(5)
        .all()
    )

    rows = []
    for row in logs:
        action_text = row.action or ""
        if not action_text.lower().startswith("wholesale"):
            continue

        rows.append(
            {
                "id": row.id,
                "date": row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else "—",
                "pipeline": "Wholesale",
                "action": row.action,
                "status": row.status,
                "notes": row.notes or "—",
            }
        )

    return {"rows": rows}