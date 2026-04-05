import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from app.services.model_loader import load_ann_xgb_hybrid

PRICE_TYPE = Literal["retail", "wholesale"]

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def normalize_price_type(price_type: str | None) -> str:
    value = (price_type or "retail").strip().lower()
    if value not in {"retail", "wholesale"}:
        raise ValueError("price_type must be 'retail' or 'wholesale'")
    return value


def normalize_month_to_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() > 0.6:
        return numeric.round().astype("Int64")

    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    s2 = s.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return s2.map(month_map).astype("Int64")


def normalize_week_to_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    nums = s.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(nums, errors="coerce").astype("Int64")


def build_week_start(year, month, week):
    month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
    return month_start + pd.Timedelta(days=(int(week) - 1) * 7)


def get_next_week_label(last_year, last_month, last_week):
    if last_week < 4:
        return last_year, last_month, last_week + 1
    if last_month < 12:
        return last_year, last_month + 1, 1
    return last_year + 1, 1, 1


def week_suffix(week: int) -> str:
    if week == 1:
        return "st"
    if week == 2:
        return "nd"
    if week == 3:
        return "rd"
    return "th"


def month_int_to_name(month: int) -> str:
    month_map = {
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
    return month_map.get(int(month), str(month))


def add_tabular_lags(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.sort_values(["fish_id", "week_start"]).copy()

    for lag_no in [1, 2, 3, 4]:
        out[f"lag_{lag_no}"] = out.groupby("fish_id")["price"].shift(lag_no)

    out["roll4_mean"] = (
        out.groupby("fish_id")["price"]
        .shift(1)
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    out["diff_1"] = out["lag_1"] - out["lag_2"]
    out["diff_2"] = out["lag_2"] - out["lag_3"]
    out["pct_change_1"] = (
        (out["lag_1"] - out["lag_2"]) /
        np.clip(np.abs(out["lag_2"]), 1e-6, None)
    )

    return out


def _load_and_prepare_long_df(long_csv_path: str, fish_to_idx: dict) -> pd.DataFrame:
    if not os.path.exists(long_csv_path):
        raise FileNotFoundError(f"Long format dataset not found: {long_csv_path}")

    df = pd.read_csv(long_csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Sinhala Name", "Common Name", "Year", "Month", "Week", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in long format file: {missing}")

    df = df.rename(columns={
        "Sinhala Name": "sinhala_name",
        "Common Name": "common_name",
        "Year": "year",
        "Month": "month",
        "Week": "week_in_month",
        "Price": "price",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = normalize_month_to_int(df["month"])
    df["week_in_month"] = normalize_week_to_int(df["week_in_month"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(
        subset=["sinhala_name", "common_name", "year", "month", "week_in_month", "price"]
    ).copy()

    if df.empty:
        raise ValueError("No usable rows found in long format file after cleaning.")

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["week_in_month"] = df["week_in_month"].astype(int)
    df["price"] = df["price"].astype(np.float32)

    df = df[df["week_in_month"].between(1, 4)].copy()
    if df.empty:
        raise ValueError("No usable rows remain after keeping weeks 1 to 4 only.")

    df["fish_id"] = (
        df["sinhala_name"].astype(str).str.strip()
        + " | "
        + df["common_name"].astype(str).str.strip()
    )

    df["fish_idx"] = df["fish_id"].map(fish_to_idx).fillna(0).astype("int32")

    df["week_start"] = df.apply(
        lambda r: build_week_start(r["year"], r["month"], r["week_in_month"]),
        axis=1,
    )
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)

    df = (
        df.groupby(["fish_id", "week_start", "week_end"], as_index=False)
        .agg(
            sinhala_name=("sinhala_name", "first"),
            common_name=("common_name", "first"),
            year=("year", "first"),
            month=("month", "first"),
            week_in_month=("week_in_month", "first"),
            price=("price", "mean"),
        )
    )

    df["fish_idx"] = df["fish_id"].map(fish_to_idx).fillna(0).astype("int32")

    df = df.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows found after grouping duplicate weekly rows.")

    return df


def _build_future_feature_rows(df: pd.DataFrame, num_cols: list[str]):
    last_row = df.sort_values(["year", "month", "week_in_month"]).tail(1).iloc[0]

    pred_year, pred_month, pred_week = get_next_week_label(
        int(last_row["year"]),
        int(last_row["month"]),
        int(last_row["week_in_month"]),
    )

    pred_week_start = build_week_start(pred_year, pred_month, pred_week)
    pred_week_end = pred_week_start + pd.Timedelta(days=6)

    future = df[["fish_id", "fish_idx", "sinhala_name", "common_name"]].drop_duplicates().copy()
    future["fish_idx"] = future["fish_idx"].fillna(0).astype("int32")

    future["week_start"] = pred_week_start
    future["week_end"] = pred_week_end
    future["price"] = np.nan
    future["year"] = pred_year
    future["month"] = pred_month
    future["week_in_month"] = pred_week

    # placeholder exogenous features
    future["holiday_count"] = 0.0
    future["is_holiday_week"] = 0.0
    future["poya_count"] = 0.0
    future["temp_mean"] = 0.0
    future["precip_sum"] = 0.0
    future["wind_max"] = 0.0
    future["humidity_mean"] = 0.0
    future["month_sin"] = np.sin(2 * np.pi * future["month"] / 12.0).astype(np.float32)
    future["month_cos"] = np.cos(2 * np.pi * future["month"] / 12.0).astype(np.float32)
    future["year_trend"] = (future["year"] - df["year"].min()).astype(np.float32)

    known_df = df.copy()
    known_df["holiday_count"] = 0.0
    known_df["is_holiday_week"] = 0.0
    known_df["poya_count"] = 0.0
    known_df["temp_mean"] = 0.0
    known_df["precip_sum"] = 0.0
    known_df["wind_max"] = 0.0
    known_df["humidity_mean"] = 0.0
    known_df["month_sin"] = np.sin(2 * np.pi * known_df["month"] / 12.0).astype(np.float32)
    known_df["month_cos"] = np.cos(2 * np.pi * known_df["month"] / 12.0).astype(np.float32)
    known_df["year_trend"] = (known_df["year"] - known_df["year"].min()).astype(np.float32)

    df_all = pd.concat([known_df, future], ignore_index=True)
    df_all = df_all.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

    tab_df = add_tabular_lags(df_all)

    required_lags = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "roll4_mean", "diff_1", "diff_2", "pct_change_1",
    ]

    future_df = tab_df[tab_df["week_start"] == pred_week_start].copy()
    future_df = future_df.dropna(subset=required_lags).copy()

    if future_df.empty:
        raise ValueError("No future rows available after lag creation.")

    for col in num_cols:
        if col not in future_df.columns:
            future_df[col] = 0.0

    for col in num_cols:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")

    future_df[num_cols] = future_df[num_cols].replace([np.inf, -np.inf], np.nan)
    future_df[num_cols] = future_df[num_cols].fillna(0.0)

    for col in num_cols:
        future_df[col] = future_df[col].astype(np.float32)

    future_df["fish_id"] = future_df["fish_id"].fillna("UNKNOWN").astype(str)
    future_df["fish_idx"] = future_df["fish_idx"].fillna(0).astype("int32")

    return future_df, pred_year, pred_month, pred_week

def generate_next_week_predictions_with_saved_hybrid_df(
    df: pd.DataFrame,
    price_type: str = "wholesale",
):
    price_type = normalize_price_type(price_type)

    ann_model, xgb_model, metadata = load_ann_xgb_hybrid(price_type=price_type)

    fish_to_idx = metadata.get("fish_to_idx", {})
    num_cols = metadata.get("num_cols", [])
    feature_cols = metadata.get("feature_cols", [])
    best_w_ann = float(metadata.get("best_w_ann", 0.5))
    best_w_xgb = float(metadata.get("best_w_xgb", 0.5))
    ann_alpha = float(metadata.get("ann_alpha", 1.0))

    if not num_cols:
        raise ValueError("Metadata missing 'num_cols'.")
    if not feature_cols:
        raise ValueError("Metadata missing 'feature_cols'.")
    if not fish_to_idx:
        raise ValueError("Metadata missing 'fish_to_idx'.")

    # ✅ NO FILE → use dataframe directly
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={
        "Sinhala Name": "sinhala_name",
        "Common Name": "common_name",
        "Year": "year",
        "Month": "month",
        "Week": "week_in_month",
        "Price": "price",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = normalize_month_to_int(df["month"])
    df["week_in_month"] = normalize_week_to_int(df["week_in_month"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["sinhala_name", "common_name", "year", "month", "week_in_month", "price"]).copy()

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["week_in_month"] = df["week_in_month"].astype(int)
    df["price"] = df["price"].astype(np.float32)

    df["fish_id"] = df["sinhala_name"].str.strip() + " | " + df["common_name"].str.strip()
    df["fish_idx"] = df["fish_id"].map(fish_to_idx).fillna(0).astype("int32")

    df["week_start"] = df.apply(
        lambda r: build_week_start(r["year"], r["month"], r["week_in_month"]),
        axis=1,
    )

    df = df.sort_values(["fish_id", "week_start"]).reset_index(drop=True)

    # build future features
    future_df, pred_year, pred_month, pred_week = _build_future_feature_rows(df, num_cols)

    # ✅ ANN FIX (VERY IMPORTANT)
    x_ann_future = {
        "fish_idx": tf.constant(
            future_df["fish_idx"].values.reshape(-1, 1),
            dtype=tf.int32,
        ),
        "num_features": tf.constant(
            future_df[num_cols].values,
            dtype=tf.float32,
        ),
    }

    ann_future_delta = ann_model.predict(x_ann_future, verbose=0).reshape(-1)
    ann_pred = future_df["lag_1"].values + (ann_alpha * ann_future_delta)

    # XGBoost
    xgb_feature_future = pd.get_dummies(
        future_df[["fish_id"] + num_cols],
        columns=["fish_id"],
        drop_first=False,
    )

    xgb_feature_future = xgb_feature_future.reindex(columns=feature_cols, fill_value=0)
    xgb_feature_future = xgb_feature_future.astype(np.float32)

    xgb_pred = xgb_model.predict(xgb_feature_future)

    final_pred = (best_w_ann * ann_pred) + (best_w_xgb * xgb_pred)

    out = future_df[["sinhala_name", "common_name", "year", "month", "week_in_month"]].copy()
    out["Predicted_Price"] = np.round(final_pred, 2)

    out = out.rename(columns={
        "sinhala_name": "Sinhala Name",
        "common_name": "Common Name",
        "year": "Year",
        "month": "Month",
        "week_in_month": "Week",
    })

    out["Month"] = out["Month"].apply(month_int_to_name)

    display_week_label = (
        f"{pred_week}{week_suffix(pred_week)} week of {month_int_to_name(pred_month)} {pred_year}"
    )

    return {
        "date": display_week_label,
        "rowCount": int(len(out)),
        "modelName": "Hybrid_ANN_XGBoost",
        "preview": out.to_dict(orient="records"),
    }