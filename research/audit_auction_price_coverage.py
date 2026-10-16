
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# ============================================================
# PHASE 3 - STEP 15
# AUCTION-PRICE COVERAGE AND ABLATION AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "data" / "processed"
    / "tft_weekly_features.csv"
)

MANIFEST_DIR = ROOT / "data" / "processed"

ABLATION_FILE = (
    ROOT / "research" / "outputs"
    / "tft_ablation"
    / "matched_validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "research" / "outputs"
    / "tft_price_coverage_audit"
)

ENCODER_LENGTH = 52
FORECAST_HORIZON = 4

GRADES = ["BOP", "BOPF", "Dust", "OP", "Pekoe"]

PRICE_MASKS = {
    "high": "auction_price_high_lkr_per_kg_available",
    "medium": "auction_price_medium_lkr_per_kg_available",
}

PARTITIONS = {
    "train": "tft_train_window_manifest.csv",
    "validation": "tft_validation_window_manifest.csv",
}

WINDOW_OUTPUT = (
    OUTPUT_DIR / "price_coverage_by_window.csv"
)

GRADE_OUTPUT = (
    OUTPUT_DIR / "price_coverage_by_grade.csv"
)

PERFORMANCE_OUTPUT = (
    OUTPUT_DIR / "price_coverage_vs_validation_error.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "price_coverage_audit.json"
)


def load_features():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    df = pd.read_csv(FEATURE_FILE)

    required = [
        "week_start_date",
        "grade",
        "time_idx",
        *PRICE_MASKS.values(),
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        errors="raise",
    ).dt.normalize()

    df["grade"] = (
        df["grade"].astype(str).str.strip()
    )

    if len(df) != 5680:
        raise ValueError(
            f"Expected 5680 rows, got {len(df)}"
        )

    if set(df["grade"]) != set(GRADES):
        raise ValueError("Unexpected tea grades")

    if df.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate weekly grade records"
        )

    for column in PRICE_MASKS.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"Invalid price availability mask: {column}"
            )

    df = df.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)

    return df


def load_manifest(partition):
    path = MANIFEST_DIR / PARTITIONS[partition]

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    for column in [
        "encoder_start",
        "encoder_end",
        "forecast_origin",
        "target_start",
        "target_end",
    ]:
        if column not in df.columns:
            raise ValueError(
                f"Missing manifest column: {column}"
            )

        df[column] = pd.to_datetime(
            df[column],
            errors="raise",
        ).dt.normalize()

    if df.duplicated(
        ["grade", "forecast_origin"]
    ).any():
        raise ValueError(
            f"Duplicate {partition} forecast origins"
        )

    return df


def audit_windows(features, manifests):
    all_results = []

    grade_lookup = {
        grade: (
            features[features["grade"] == grade]
            .set_index("week_start_date")
            .sort_index()
        )
        for grade in GRADES
    }

    for partition, manifest in manifests.items():
        for row in manifest.itertuples(index=False):
            grade = str(row.grade)

            if grade not in grade_lookup:
                raise ValueError(
                    f"Unknown grade: {grade}"
                )

            history = grade_lookup[grade].loc[
                row.encoder_start:row.encoder_end
            ]

            if len(history) != ENCODER_LENGTH:
                raise ValueError(
                    f"{grade}: incorrect encoder length "
                    f"at {row.forecast_origin}"
                )

            if history.index.max() != row.forecast_origin:
                raise ValueError(
                    "Forecast origin does not match "
                    "the last encoder week"
                )

            if history.index.min() != row.encoder_start:
                raise ValueError(
                    "Encoder start date mismatch"
                )

            expected_dates = pd.date_range(
                row.encoder_start,
                periods=ENCODER_LENGTH,
                freq="W-MON",
            )

            if not history.index.equals(expected_dates):
                raise ValueError(
                    "Discontinuous encoder dates"
                )

            if row.target_start <= row.forecast_origin:
                raise ValueError(
                    "Future target leakage detected"
                )

            if (
                row.target_end - row.target_start
            ).days != 7 * (FORECAST_HORIZON - 1):
                raise ValueError(
                    "Incorrect four-week target horizon"
                )

            high = history[
                PRICE_MASKS["high"]
            ].to_numpy(dtype=int)

            medium = history[
                PRICE_MASKS["medium"]
            ].to_numpy(dtype=int)

            any_price = (
                (high == 1) | (medium == 1)
            ).astype(int)

            high_count = int(high.sum())
            medium_count = int(medium.sum())
            any_count = int(any_price.sum())

            latest_price_age = None

            observed_positions = np.flatnonzero(
                any_price
            )

            if len(observed_positions):
                latest_price_age = int(
                    ENCODER_LENGTH
                    - 1
                    - observed_positions[-1]
                )

            if any_count == 0:
                coverage_group = "none"
            elif any_count < 13:
                coverage_group = "low_1_to_12"
            elif any_count < 39:
                coverage_group = "moderate_13_to_38"
            else:
                coverage_group = "high_39_to_52"

            all_results.append({
                "partition": partition,
                "grade": grade,
                "forecast_origin": (
                    row.forecast_origin.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "target_start": (
                    row.target_start.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "encoder_start": (
                    row.encoder_start.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "encoder_end": (
                    row.encoder_end.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "encoder_weeks": ENCODER_LENGTH,
                "high_price_weeks": high_count,
                "medium_price_weeks": medium_count,
                "any_price_weeks": any_count,
                "high_coverage_percent": round(
                    100 * high_count / ENCODER_LENGTH,
                    4,
                ),
                "medium_coverage_percent": round(
                    100 * medium_count / ENCODER_LENGTH,
                    4,
                ),
                "any_coverage_percent": round(
                    100 * any_count / ENCODER_LENGTH,
                    4,
                ),
                "price_observed_at_origin": int(
                    any_price[-1]
                ),
                "latest_price_age_weeks": (
                    latest_price_age
                ),
                "coverage_group": coverage_group,
            })

    result = pd.DataFrame(all_results)

    for partition, manifest in manifests.items():
        actual_count = int(
            (result["partition"] == partition).sum()
        )

        if actual_count != len(manifest):
            raise ValueError(
                f"{partition}: coverage-window count mismatch"
            )

    return result


def build_grade_summary(windows):
    rows = []

    for partition in PARTITIONS:
        for grade in GRADES:
            subset = windows[
                (windows["partition"] == partition)
                & (windows["grade"] == grade)
            ]

            rows.append({
                "partition": partition,
                "grade": grade,
                "forecast_windows": int(len(subset)),
                "average_high_price_weeks": float(
                    subset["high_price_weeks"].mean()
                ),
                "average_medium_price_weeks": float(
                    subset["medium_price_weeks"].mean()
                ),
                "average_any_price_weeks": float(
                    subset["any_price_weeks"].mean()
                ),
                "windows_without_prices": int(
                    subset["any_price_weeks"].eq(0).sum()
                ),
                "windows_with_price_at_origin": int(
                    subset[
                        "price_observed_at_origin"
                    ].sum()
                ),
            })

    return pd.DataFrame(rows)


def analyze_validation_performance(windows):
    if not ABLATION_FILE.exists():
        raise FileNotFoundError(
            f"Step 14 prediction file missing: {ABLATION_FILE}"
        )

    predictions = pd.read_csv(ABLATION_FILE)

    required = [
        "grade",
        "forecast_origin",
        "target_start",
        "target_week",
        "horizon",
        "actual_demand_kg",
        "with_price_q50_kg",
        "no_price_q50_kg",
    ]

    missing = [
        column for column in required
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Step 14 columns: {missing}"
        )

    validation = windows[
        windows["partition"] == "validation"
    ].copy()

    keys = [
        "grade",
        "forecast_origin",
        "target_start",
    ]

    merged = predictions.merge(
        validation[
            keys + [
                "any_price_weeks",
                "price_observed_at_origin",
                "coverage_group",
                "latest_price_age_weeks",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Unmatched validation forecasts"
        )

    merged = merged.drop(columns=["_merge"])

    if len(merged) != 1000:
        raise ValueError(
            f"Expected 1000 predictions, found {len(merged)}"
        )

    merged["with_price_absolute_error"] = (
        merged["actual_demand_kg"]
        - merged["with_price_q50_kg"]
    ).abs()

    merged["no_price_absolute_error"] = (
        merged["actual_demand_kg"]
        - merged["no_price_q50_kg"]
    ).abs()

    # Positive values mean the price-aware model
    # has lower MAE than the no-price model.
    merged["price_absolute_error_benefit"] = (
        merged["no_price_absolute_error"]
        - merged["with_price_absolute_error"]
    )

    report_rows = []

    for group_name, subset in (
        merged.groupby(
            "coverage_group",
            dropna=False,
        )
    ):
        with_mae = float(
            subset[
                "with_price_absolute_error"
            ].mean()
        )

        no_mae = float(
            subset[
                "no_price_absolute_error"
            ].mean()
        )

        report_rows.append({
            "coverage_group": group_name,
            "forecast_windows": int(
                len(subset) // FORECAST_HORIZON
            ),
            "prediction_rows": int(len(subset)),
            "average_encoder_price_weeks": float(
                subset["any_price_weeks"].mean()
            ),
            "with_price_mae_kg": with_mae,
            "no_price_mae_kg": no_mae,
            "price_mae_benefit_kg": (
                no_mae - with_mae
            ),
            "price_mae_benefit_percent": (
                100 * (no_mae - with_mae) / no_mae
                if no_mae > 0 else None
            ),
        })

    report_df = pd.DataFrame(report_rows)

    return report_df


def main():
    print(
        "STEP 15 - HISTORICAL AUCTION PRICE COVERAGE AUDIT"
    )
    print("=" * 70)

    features = load_features()

    manifests = {
        name: load_manifest(name)
        for name in PARTITIONS
    }

    windows = audit_windows(
        features,
        manifests,
    )

    grade_summary = build_grade_summary(
        windows
    )

    performance = analyze_validation_performance(
        windows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows.to_csv(
        WINDOW_OUTPUT,
        index=False,
    )

    grade_summary.to_csv(
        GRADE_OUTPUT,
        index=False,
    )

    performance.to_csv(
        PERFORMANCE_OUTPUT,
        index=False,
    )

    summary = {
        "status": "PASS",
        "encoder_length": ENCODER_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "target": "demand_kg",
        "coverage_definition": (
            "Auction-price availability across "
            "the 52 weeks preceding each forecast."
        ),
        "window_counts": {
            name: int(
                (windows["partition"] == name).sum()
            )
            for name in PARTITIONS
        },
        "validation_windows_without_any_price": int(
            windows.loc[
                windows["partition"] == "validation",
                "any_price_weeks",
            ].eq(0).sum()
        ),
        "validation_windows_with_price_at_origin": int(
            windows.loc[
                windows["partition"] == "validation",
                "price_observed_at_origin",
            ].sum()
        ),
        "comparison_groups": performance.to_dict(
            orient="records"
        ),
        "outputs": {
            "windows": str(WINDOW_OUTPUT),
            "grade_summary": str(GRADE_OUTPUT),
            "performance": str(PERFORMANCE_OUTPUT),
        },
        "test_partition_evaluated": False,
        "limitations": [
            (
                "Price coverage does not independently "
                "verify quotation accuracy."
            ),
            (
                "Auction-price availability differs "
                "by tea grade and calendar period."
            ),
            (
                "Rolling forecast windows overlap."
            ),
            (
                "Observed subgroup differences "
                "do not establish causality."
            ),
            (
                "No test forecasts were generated."
            ),
        ],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nWINDOW COUNTS")

    for name, count in summary[
        "window_counts"
    ].items():
        print(f"{name}: {count}")

    print("\nVALIDATION PRICE COVERAGE BY GRADE")

    print(
        grade_summary[
            grade_summary["partition"] == "validation"
        ][
            [
                "grade",
                "forecast_windows",
                "average_high_price_weeks",
                "average_medium_price_weeks",
                "windows_without_prices",
                "windows_with_price_at_origin",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nPRICE COVERAGE VS VALIDATION ACCURACY")

    print(
        performance.round(2).to_string(
            index=False
        )
    )

    print("\nOUTPUT FILES")
    print(WINDOW_OUTPUT)
    print(GRADE_OUTPUT)
    print(PERFORMANCE_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nTEST SET: NOT EVALUATED")
    print("AUCTION PRICE COVERAGE AUDIT: PASS")


if __name__ == "__main__":
    main()
