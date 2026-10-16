
"""
Build and validate chronological datasets for the
six-input tea demand forecasting TFT.

This module does not train a model or modify
existing datasets and checkpoints.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from tft_six_input_config import (
    DATA_FILE,
    OUTPUT_DIR,
    TARGET,
    GROUP_IDS,
    TEA_GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    VALIDATION_START,
    TEST_START,
    KNOWN_FUTURE_REALS,
    HISTORICAL_ONLY_REALS,
    REQUIRED_INPUT_COLUMNS,
    validate_feature_configuration,
)


def load_six_input_data():
    """Read and validate the cleaned weekly dataset."""

    validate_feature_configuration()

    df = pd.read_csv(DATA_FILE)

    missing_columns = sorted(
        set(REQUIRED_INPUT_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        format="%Y-%m-%d",
        errors="raise",
    )

    if not (
        df["week_start_date"].dt.dayofweek == 0
    ).all():
        raise ValueError(
            "All week_start_date values must be Mondays."
        )

    if df.duplicated(
        subset=["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate grade/week combinations found."
        )

    actual_grades = set(df["grade"].unique())

    if actual_grades != set(TEA_GRADES):
        raise ValueError(
            "Unexpected tea grades in the dataset."
        )

    for column in HISTORICAL_ONLY_REALS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if (
            df[column].isna().any()
            or not np.isfinite(
                df[column].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(
                f"Invalid values in {column}."
            )

    positive_columns = [
        "demand_kg",
        "usd_lkr_rate",
        "colombo_auction_price_lkr",
    ]

    for column in positive_columns:
        if (df[column] <= 0).any():
            raise ValueError(
                f"{column} must be positive."
            )

    for column in [
        "production_kg",
        "inventory_kg",
    ]:
        if (df[column] < 0).any():
            raise ValueError(
                f"{column} cannot be negative."
            )

    if not df["festival_flag"].isin([0, 1]).all():
        raise ValueError(
            "festival_flag must contain only 0 or 1."
        )

    for grade, group in df.groupby("grade"):
        dates = group[
            "week_start_date"
        ].sort_values().reset_index(drop=True)

        expected = pd.date_range(
            start=dates.iloc[0],
            end=dates.iloc[-1],
            freq="W-MON",
        )

        if not dates.equals(
            pd.Series(expected)
        ):
            raise ValueError(
                f"Missing or irregular weeks for {grade}."
            )

    # Calendar index shared across every tea grade.
    first_date = df["week_start_date"].min()

    day_offsets = (
        df["week_start_date"] - first_date
    ).dt.days

    if (day_offsets % 7 != 0).any():
        raise ValueError(
            "Weekly dates are not aligned."
        )

    df["time_idx"] = (
        day_offsets // 7
    ).astype("int64")

    iso_weeks = (
        df["week_start_date"]
        .dt.isocalendar()
        .week.astype("int64")
    )

    months = (
        df["week_start_date"].dt.month
    )

    df["week_sin"] = np.sin(
        2.0 * np.pi * iso_weeks / 52.1775
    )

    df["week_cos"] = np.cos(
        2.0 * np.pi * iso_weeks / 52.1775
    )

    df["month_sin"] = np.sin(
        2.0 * np.pi * months / 12.0
    )

    df["month_cos"] = np.cos(
        2.0 * np.pi * months / 12.0
    )

    for column in (
        HISTORICAL_ONLY_REALS
        + [
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
        ]
    ):
        df[column] = df[column].astype("float32")

    df = df.sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    return df


def count_complete_windows(df):
    """
    Count windows whose full four-week target
    horizon remains within one partition.
    """

    counts = {
        "training": 0,
        "validation": 0,
        "testing": 0,
    }

    by_grade = {}

    for grade, group in df.groupby("grade"):
        group = group.sort_values(
            "time_idx"
        ).reset_index(drop=True)

        dates = group["week_start_date"]

        grade_counts = {
            "training": 0,
            "validation": 0,
            "testing": 0,
        }

        for origin_index in range(
            ENCODER_LENGTH - 1,
            len(group) - PREDICTION_LENGTH,
        ):
            target_start = dates.iloc[
                origin_index + 1
            ]

            target_end = dates.iloc[
                origin_index + PREDICTION_LENGTH
            ]

            if target_end < VALIDATION_START:
                partition = "training"

            elif (
                target_start >= VALIDATION_START
                and target_end < TEST_START
            ):
                partition = "validation"

            elif target_start >= TEST_START:
                partition = "testing"

            else:
                continue

            counts[partition] += 1
            grade_counts[partition] += 1

        by_grade[grade] = grade_counts

    return counts, by_grade


def get_cutoff_index(df, cutoff):
    matching = df.loc[
        df["week_start_date"] == cutoff,
        "time_idx",
    ].unique()

    if len(matching) != 1:
        raise ValueError(
            f"Cutoff not found in dataset: {cutoff}"
        )

    return int(matching[0])


def create_six_input_datasets(df):
    """Create chronologically isolated TFT datasets."""

    validation_index = get_cutoff_index(
        df,
        VALIDATION_START,
    )

    test_index = get_cutoff_index(
        df,
        TEST_START,
    )

    # The training dataset contains no
    # validation or test target observations.
    training_frame = df.loc[
        df["week_start_date"] < VALIDATION_START
    ].copy()

    # Validation includes earlier training
    # history as encoder context, but no test rows.
    validation_frame = df.loc[
        df["week_start_date"] < TEST_START
    ].copy()

    # Test includes earlier history so that
    # a 52-week encoder can be constructed.
    test_frame = df.copy()

    training = TimeSeriesDataSet(
        training_frame,
        time_idx="time_idx",
        target=TARGET,
        group_ids=GROUP_IDS,
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["grade"],
        time_varying_known_reals=KNOWN_FUTURE_REALS,
        time_varying_unknown_reals=HISTORICAL_ONLY_REALS,
        target_normalizer=GroupNormalizer(
            groups=GROUP_IDS,
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_frame,
        min_prediction_idx=validation_index,
        stop_randomization=True,
        predict=False,
    )

    testing = TimeSeriesDataSet.from_dataset(
        training,
        test_frame,
        min_prediction_idx=test_index,
        stop_randomization=True,
        predict=False,
    )

    return {
        "training": training,
        "validation": validation,
        "testing": testing,
    }


def check_dataset_integrity(df, datasets):
    """Validate feature roles and partition sizes."""

    training = datasets["training"]

    known = set(
        training.time_varying_known_reals
    )

    historical = set(
        training.time_varying_unknown_reals
    )

    if known.intersection(historical):
        raise ValueError(
            "Known and historical features overlap."
        )

    if not set(
        HISTORICAL_ONLY_REALS
    ).issubset(historical):
        raise ValueError(
            "Some historical inputs were omitted."
        )

    for feature in HISTORICAL_ONLY_REALS:
        if feature in known:
            raise ValueError(
                f"Future leakage risk: {feature}"
            )

    expected, by_grade = count_complete_windows(
        df
    )

    actual = {
        name: len(dataset)
        for name, dataset in datasets.items()
    }

    print("\n=== WINDOW COUNTS ===")

    for name in [
        "training",
        "validation",
        "testing",
    ]:
        print(
            f"{name}: "
            f"expected={expected[name]}, "
            f"actual={actual[name]}"
        )

        if actual[name] != expected[name]:
            raise ValueError(
                f"{name} dataset window count mismatch."
            )

    print("\n=== WINDOWS BY TEA GRADE ===")

    for grade, counts in by_grade.items():
        print(
            f"{grade}: "
            f"train={counts['training']}, "
            f"validation={counts['validation']}, "
            f"test={counts['testing']}"
        )

    print("\n=== FEATURE ROLE VALIDATION ===")
    print(
        "Future-known:",
        sorted(known)
    )
    print(
        "Historical-only:",
        sorted(historical)
    )

    print("\nFEATURE ROLE VALIDATION: PASS")

    return {
        "window_counts": actual,
        "windows_by_grade": by_grade,
        "known_future_features": sorted(known),
        "historical_only_features": sorted(historical),
    }


def main():
    print("\n=== SIX-INPUT TFT DATASET BUILDER ===")

    df = load_six_input_data()

    print("Dataset:", DATA_FILE)
    print("Total rows:", len(df))
    print("Tea grades:", df["grade"].nunique())
    print(
        "First week:",
        df["week_start_date"].min().date()
    )
    print(
        "Last week:",
        df["week_start_date"].max().date()
    )

    datasets = create_six_input_datasets(
        df
    )

    check_dataset_integrity(
        df,
        datasets,
    )

    print("\nSIX-INPUT TFT DATASETS: PASS")
    print("Existing checkpoints modified: NO")
    print("Model training performed: NO")


if __name__ == "__main__":
    main()
