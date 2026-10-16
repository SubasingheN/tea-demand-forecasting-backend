
from pathlib import Path
from collections import Counter
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "tft_weekly_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

HISTORY_LENGTH = 52
FORECAST_HORIZON = 4

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

GRADES = ["BOP", "BOPF", "Dust", "OP", "Pekoe"]

KEYS = ["week_start_date", "grade"]

HISTORICAL_ONLY_COLUMNS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "demand_lag_1w",
    "demand_lag_4w",
    "demand_lag_12w",
    "demand_lag_52w",
    "demand_rolling_4w",
    "demand_rolling_12w",
    "auction_price_high_lkr_per_kg_input",
    "auction_price_medium_lkr_per_kg_input",
    "auction_price_high_lkr_per_kg_available",
    "auction_price_medium_lkr_per_kg_available",
]

FUTURE_KNOWN_COLUMNS = [
    "time_idx",
    "year",
    "month",
    "week_of_year",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
]


def get_partition(target_start, target_end):
    if target_end < VALIDATION_START:
        return "train"

    if (
        target_start >= VALIDATION_START
        and target_end < TEST_START
    ):
        return "validation"

    if target_start >= TEST_START:
        return "test"

    # The four target weeks cross a partition boundary.
    return None


def validate_source(df):
    required = (
        KEYS
        + HISTORICAL_ONLY_COLUMNS
        + FUTURE_KNOWN_COLUMNS
    )

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            "Missing preprocessing columns: "
            + ", ".join(missing)
        )

    if len(df) != 5680:
        raise ValueError(
            f"Expected 5680 rows; found {len(df)}"
        )

    if set(df["grade"]) != set(GRADES):
        raise ValueError("Unexpected tea grades")

    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate grade/week rows")

    if df["week_start_date"].isna().any():
        raise ValueError("Invalid date values")

    if df["week_start_date"].dt.weekday.ne(0).any():
        raise ValueError("Dates must be Mondays")

    expected_weeks = pd.date_range(
        df["week_start_date"].min(),
        df["week_start_date"].max(),
        freq="W-MON",
    )

    if len(expected_weeks) != 1136:
        raise ValueError("Unexpected timeline length")

    for grade in GRADES:
        grade_df = df[df["grade"] == grade]
        actual_weeks = pd.DatetimeIndex(
            grade_df["week_start_date"].sort_values()
        )

        if not actual_weeks.equals(expected_weeks):
            raise ValueError(
                f"Incomplete weekly history for {grade}"
            )

    if (
        set(HISTORICAL_ONLY_COLUMNS)
        & set(FUTURE_KNOWN_COLUMNS)
    ):
        raise ValueError(
            "Observed and future-known features overlap"
        )

    return expected_weeks


def create_windows(df, weeks):
    records = []
    excluded = []

    for grade in GRADES:
        grade_df = (
            df[df["grade"] == grade]
            .sort_values("week_start_date")
            .reset_index(drop=True)
        )

        for decoder_start_idx in range(
            HISTORY_LENGTH,
            len(weeks) - FORECAST_HORIZON + 1,
        ):
            encoder_start_idx = (
                decoder_start_idx - HISTORY_LENGTH
            )
            encoder_end_idx = decoder_start_idx - 1

            decoder_end_idx = (
                decoder_start_idx + FORECAST_HORIZON - 1
            )

            origin_date = weeks[encoder_end_idx]
            target_start = weeks[decoder_start_idx]
            target_end = weeks[decoder_end_idx]

            partition = get_partition(
                target_start, target_end
            )

            if partition is None:
                excluded.append({
                    "grade": grade,
                    "forecast_origin": str(origin_date.date()),
                    "target_start": str(target_start.date()),
                    "target_end": str(target_end.date()),
                    "reason": "crosses_partition_boundary",
                })
                continue

            encoder = grade_df.iloc[
                encoder_start_idx:decoder_start_idx
            ]

            decoder = grade_df.iloc[
                decoder_start_idx:decoder_end_idx + 1
            ]

            if len(encoder) != HISTORY_LENGTH:
                raise ValueError("Invalid encoder length")

            if len(decoder) != FORECAST_HORIZON:
                raise ValueError("Invalid decoder length")

            if not (
                encoder["week_start_date"].max()
                < decoder["week_start_date"].min()
            ):
                raise ValueError("Encoder/decoder leakage")

            if not (
                encoder["demand_kg"].notna().all()
                and decoder["demand_kg"].notna().all()
            ):
                raise ValueError(
                    "Missing demand in forecasting window"
                )

            # Audit observed-price availability in the
            # encoder, not in the future decoder.
            high_available = int(
                encoder[
                    "auction_price_high_lkr_per_kg_available"
                ].sum()
            )

            medium_available = int(
                encoder[
                    "auction_price_medium_lkr_per_kg_available"
                ].sum()
            )

            records.append({
                "grade": grade,
                "partition": partition,
                "encoder_start": str(
                    weeks[encoder_start_idx].date()
                ),
                "encoder_end": str(
                    weeks[encoder_end_idx].date()
                ),
                "forecast_origin": str(origin_date.date()),
                "target_start": str(target_start.date()),
                "target_end": str(target_end.date()),
                "encoder_length": HISTORY_LENGTH,
                "forecast_horizon": FORECAST_HORIZON,
                "high_price_encoder_weeks": high_available,
                "medium_price_encoder_weeks": medium_available,
            })

    return (
        pd.DataFrame(records),
        pd.DataFrame(excluded),
    )


def validate_windows(windows):
    if windows.empty:
        raise ValueError("No usable windows created")

    if windows.duplicated(
        ["grade", "forecast_origin"]
    ).any():
        raise ValueError("Duplicate forecasting origins")

    starts = pd.to_datetime(windows["target_start"])
    ends = pd.to_datetime(windows["target_end"])
    origins = pd.to_datetime(windows["forecast_origin"])

    if (origins >= starts).any():
        raise ValueError("Future target leakage detected")

    if (
        (ends - starts).dt.days != 21
    ).any():
        raise ValueError(
            "Forecast window must span four Mondays"
        )

    train = windows["partition"] == "train"
    validation = windows["partition"] == "validation"
    test = windows["partition"] == "test"

    if (ends[train] >= VALIDATION_START).any():
        raise ValueError("Training/validation leakage")

    if (
        (starts[validation] < VALIDATION_START).any()
        or (ends[validation] >= TEST_START).any()
    ):
        raise ValueError("Invalid validation targets")

    if (starts[test] < TEST_START).any():
        raise ValueError("Invalid test targets")

    if windows["encoder_length"].ne(
        HISTORY_LENGTH
    ).any():
        raise ValueError("Incorrect encoder length")

    if windows["forecast_horizon"].ne(
        FORECAST_HORIZON
    ).any():
        raise ValueError("Incorrect prediction horizon")

    return True


def main():
    print("PHASE 2 - STEP 8: TFT CHRONOLOGICAL SPLITS")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        errors="raise",
    )

    df["grade"] = df["grade"].astype(str).str.strip()

    df = df.sort_values(KEYS).reset_index(drop=True)

    weeks = validate_source(df)

    windows, excluded = create_windows(df, weeks)

    validate_windows(windows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {}

    for partition in ("train", "validation", "test"):
        subset = windows[
            windows["partition"] == partition
        ].copy()

        path = (
            OUTPUT_DIR
            / f"tft_{partition}_window_manifest.csv"
        )

        subset.to_csv(path, index=False)

        output_paths[partition] = str(path)

    excluded_path = (
        OUTPUT_DIR / "tft_excluded_boundary_windows.csv"
    )

    excluded.to_csv(excluded_path, index=False)

    counts = Counter(windows["partition"])

    per_grade = {}

    for grade in GRADES:
        per_grade[grade] = {
            partition: int(
                len(
                    windows[
                        (windows["grade"] == grade)
                        & (windows["partition"] == partition)
                    ]
                )
            )
            for partition in ("train", "validation", "test")
        }

    report = {
        "status": "PASS",
        "source_file": str(INPUT_FILE),
        "total_source_rows": int(len(df)),
        "total_weeks": int(len(weeks)),
        "grades": GRADES,
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders including "
            "unfulfilled orders"
        ),
        "encoder_length": HISTORY_LENGTH,
        "prediction_length": FORECAST_HORIZON,
        "validation_start": str(
            VALIDATION_START.date()
        ),
        "test_start": str(TEST_START.date()),
        "window_counts": {
            key: int(counts[key])
            for key in ("train", "validation", "test")
        },
        "per_grade": per_grade,
        "excluded_boundary_windows": int(
            len(excluded)
        ),
        "historical_only_columns": (
            HISTORICAL_ONLY_COLUMNS
        ),
        "known_future_columns": FUTURE_KNOWN_COLUMNS,
        "modeling_rule": (
            "Decoder inputs must contain only "
            "future-known calendar features. "
            "Demand is the decoder target, not "
            "a future-known predictor. Observed "
            "production, inventory, lagged demand "
            "and auction prices must never be "
            "passed as future-known values."
        ),
        "files": output_paths,
    }

    report_path = (
        OUTPUT_DIR / "tft_split_readiness.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Source rows:", len(df))
    print("Weekly periods:", len(weeks))
    print("Grades:", len(GRADES))

    print("\nFORECAST WINDOW COUNTS")
    for partition in ("train", "validation", "test"):
        print(
            f"{partition}: {counts[partition]}"
        )

    print(
        "Excluded boundary-crossing windows:",
        len(excluded)
    )

    print("\nCOUNTS PER GRADE")
    for grade, partitions in per_grade.items():
        print(grade, partitions)

    print("\nOUTPUT FILES")
    for path in output_paths.values():
        print(path)

    print(excluded_path)
    print(report_path)

    print("\nCHRONOLOGICAL SPLIT VALIDATION: PASS")
    print(
        "Note: Actual TFT dataset construction must "
        "still enforce encoder/decoder feature roles."
    )


if __name__ == "__main__":
    main()
