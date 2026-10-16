
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT / "data" /
    "finalized_tft_weekly_tea_demand_model_data.csv"
)

OUTPUT_DIR = ROOT / "data" / "validated"

OUTPUT_PATH = (
    OUTPUT_DIR / "tea_demand_corrected.csv"
)

AUDIT_PATH = (
    OUTPUT_DIR / "dataset_correction_audit.json"
)

GRADES = {"BOP", "BOPF", "Dust", "OP", "Pekoe"}


def prepare_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_PATH)

    required = {
        "week_start_date",
        "time_idx",
        "series_id",
        "grade",
        "demand_kg",
        "colombo_auction_price_lkr",
    }

    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        format="mixed",
        errors="raise",
    )

    if df["week_start_date"].isna().any():
        raise ValueError("Invalid week_start_date found.")

    df["grade"] = df["grade"].astype(str)
    df["series_id"] = df["series_id"].astype(str)

    if set(df["grade"]) != GRADES:
        raise ValueError(
            "Dataset does not contain the expected five grades."
        )

    if df.duplicated(
        ["series_id", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate series and week combinations found."
        )

    df = df.sort_values(
        ["series_id", "week_start_date"]
    ).reset_index(drop=True)

    for column in [
        "demand_kg",
        "production_kg",
        "inventory_kg",
        "colombo_auction_price_lkr",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if df[column].isna().any():
            raise ValueError(
                f"Missing numeric values in {column}"
            )

    if (df["demand_kg"] < 0).any():
        raise ValueError("Negative demand values found.")

    if (df["colombo_auction_price_lkr"] <= 0).any():
        raise ValueError("Invalid auction prices found.")

    grouped = df.groupby("series_id", sort=False)

    # Historical auction prices only.
    df["auction_price_lag1w"] = (
        grouped["colombo_auction_price_lkr"]
        .shift(1)
    )

    df["auction_price_lag4w"] = (
        grouped["colombo_auction_price_lkr"]
        .shift(4)
    )

    # Historical demand only. The current week's
    # demand is deliberately excluded.
    df["demand_rolling4w_avg"] = (
        grouped["demand_kg"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=4,
                min_periods=1,
            ).mean()
        )
    )

    df["demand_rolling12w_avg"] = (
        grouped["demand_kg"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=12,
                min_periods=1,
            ).mean()
        )
    )

    # Explicitly flag unavailable historical values.
    history_columns = [
        "auction_price_lag1w",
        "auction_price_lag4w",
        "demand_rolling4w_avg",
        "demand_rolling12w_avg",
    ]

    for column in history_columns:
        df[f"{column}_missing"] = (
            df[column].isna().astype(int)
        )

    # Preserve missing historical information as NaN.
    # Do not fill it with future or current demand.
    df["week_start_date"] = (
        df["week_start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    df.to_csv(OUTPUT_PATH, index=False)

    audit = {
        "source_file": str(SOURCE_PATH.name),
        "output_file": str(OUTPUT_PATH.name),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "grades": sorted(df["grade"].unique().tolist()),
        "first_week": df["week_start_date"].min(),
        "last_week": df["week_start_date"].max(),
        "missing_history_by_column": {
            column: int(df[column].isna().sum())
            for column in history_columns
        },
        "price_source_status": "unverified",
        "target_source_status": "not_independently_verified",
        "note": (
            "Historical features recalculated without "
            "using current-week demand. Auction prices "
            "still require independent source verification."
        ),
    }

    with open(
        AUDIT_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(audit, file, indent=2)

    print("Dataset correction completed.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Grades:", sorted(df["grade"].unique()))
    print("Output:", OUTPUT_PATH)
    print("Audit:", AUDIT_PATH)


if __name__ == "__main__":
    prepare_dataset()
