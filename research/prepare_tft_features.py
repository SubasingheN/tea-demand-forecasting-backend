
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "external"
    / "validated_audit"
    / "tea_demand_factory_auction_merged.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

FEATURE_FILE = OUTPUT_DIR / "tft_weekly_features.csv"
REPORT_FILE = OUTPUT_DIR / "tft_feature_report.json"

GRADES = ["BOP", "BOPF", "Dust", "OP", "Pekoe"]

PRICE_COLUMNS = [
    "auction_price_high_lkr_per_kg",
    "auction_price_medium_lkr_per_kg",
]

HISTORY_LENGTH = 52
FORECAST_HORIZON = 4


def main():
    print("PHASE 2 - STEP 7: TFT FEATURE PREPARATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "week_start_date",
        "grade",
        "demand_kg",
        "production_kg",
        "inventory_kg",
    ] + PRICE_COLUMNS

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"], errors="raise"
    )

    df["grade"] = df["grade"].astype(str).str.strip()

    if len(df) != 5680:
        raise ValueError(
            f"Expected 5680 rows; found {len(df)}."
        )

    if set(df["grade"]) != set(GRADES):
        raise ValueError("Unexpected tea grades.")

    keys = ["grade", "week_start_date"]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate week/grade observations detected."
        )

    if df["week_start_date"].dt.weekday.ne(0).any():
        raise ValueError(
            "Weekly dates must start on Monday."
        )

    df = df.sort_values(keys).reset_index(drop=True)

    numeric_columns = [
        "demand_kg",
        "production_kg",
        "inventory_kg",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

        if (
            df[column].isna().any()
            or df[column].lt(0).any()
        ):
            raise ValueError(
                f"Invalid factory values: {column}"
            )

    # Consistent weekly index shared by all grades.
    weeks = sorted(df["week_start_date"].unique())

    if len(weeks) != 1136:
        raise ValueError(
            f"Expected 1136 weeks; found {len(weeks)}."
        )

    expected_weeks = pd.date_range(
        min(weeks), max(weeks), freq="W-MON"
    )

    if not pd.DatetimeIndex(weeks).equals(
        expected_weeks
    ):
        raise ValueError(
            "Missing or discontinuous weekly dates."
        )

    week_index = {
        week: index
        for index, week in enumerate(weeks)
    }

    df["time_idx"] = df["week_start_date"].map(
        week_index
    ).astype(int)

    # Known future calendar features.
    iso_calendar = df["week_start_date"].dt.isocalendar()

    df["year"] = df["week_start_date"].dt.year
    df["month"] = df["week_start_date"].dt.month
    df["week_of_year"] = (
        iso_calendar.week.astype(int)
    )

    df["week_sin"] = np.sin(
        2 * np.pi * df["week_of_year"] / 52.1775
    )

    df["week_cos"] = np.cos(
        2 * np.pi * df["week_of_year"] / 52.1775
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Historical observations.
    # Lagged demand features are valid at their own
    # dates but must not be used as known-future
    # decoder features.
    for lag in [1, 4, 12, 52]:
        df[f"demand_lag_{lag}w"] = (
            df.groupby("grade")["demand_kg"]
            .shift(lag)
        )

    # Only past demand is used in rolling features.
    grouped = df.groupby("grade")["demand_kg"]

    for window in [4, 12]:
        df[f"demand_rolling_{window}w"] = (
            grouped.transform(
                lambda series: (
                    series.shift(1).rolling(
                        window=window,
                        min_periods=window
                    ).mean()
                )
            )
        )

    # Preserve missing auction prices.
    # Do not forward-fill them across missing years.
    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

        if df[column].dropna().le(0).any():
            raise ValueError(
                f"Nonpositive price: {column}"
            )

        df[f"{column}_available"] = (
            df[column].notna().astype(int)
        )

        # Historical observed inputs only.
        # Zero is merely a numerical placeholder,
        # and the corresponding mask preserves
        # missingness information.
        df[f"{column}_input"] = (
            df[column].fillna(0.0)
        )

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    df["week_start_date"] = (
        df["week_start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    df.to_csv(
        FEATURE_FILE,
        index=False
    )

    report = {
        "rows": int(len(df)),
        "weeks": int(len(weeks)),
        "grades": GRADES,
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders including "
            "unfulfilled orders"
        ),
        "high_price_observations": int(
            df[PRICE_COLUMNS[0]].notna().sum()
        ),
        "medium_price_observations": int(
            df[PRICE_COLUMNS[1]].notna().sum()
        ),
        "historical_only_features": [
            "demand_kg",
            "production_kg",
            "inventory_kg",
            "auction_price_high_lkr_per_kg_input",
            "auction_price_medium_lkr_per_kg_input",
        ],
        "known_future_features": [
            "time_idx",
            "week_of_year",
            "month",
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
        ],
        "decoder_exclusions": [
            "future demand_kg",
            "future production_kg",
            "future inventory_kg",
            "future auction prices",
            "demand lag/rolling features",
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Weeks:", len(weeks))
    print("Grades:", len(GRADES))

    print("\nAUCTION PRICE COVERAGE")
    for column in PRICE_COLUMNS:
        print(
            column,
            ":", int(df[column].notna().sum())
        )

    print("\nOUTPUT FILES")
    print(FEATURE_FILE)
    print(REPORT_FILE)

    print("\nFEATURE PREPARATION: PASS")
    print(
        "Note: Leakage safety must also be "
        "enforced when constructing TFT windows."
    )


if __name__ == "__main__":
    main()
