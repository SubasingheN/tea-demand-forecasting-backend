
"""
Build and validate datasets for the revised
tea demand forecasting TFT.

The revised dataset includes nine historical
numeric inputs:

1. demand_kg
2. production_kg
3. inventory_kg
4. usd_lkr_monday_rate
5. usd_lkr_sunday_rate
6. festival_flag
7. auction_price_high_lkr
8. auction_price_medium_lkr
9. auction_price_low_lkr

The original USD/LKR exchange-rate range is
preserved in the source CSV but is not supplied
directly to the forecasting model.

Forecasting configuration:
- 52 consecutive historical weeks
- Four consecutive future weeks
- Five tea grades
- Chronological train/validation/test splits

This script does not train a model or overwrite
the existing training dataset or checkpoints.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from tft_eight_input_config import (
    TARGET,
    GROUP_IDS,
    TEA_GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    VALIDATION_START,
    TEST_START,
    KNOWN_FUTURE_REALS,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BACKEND_ROOT
    / "data"
    / "processed"
    / "tft_eight_input_model_ready_ranges_2021_2026.csv"
)

EXCHANGE_RANGE_COLUMN = (
    "usd_lkr_exchange_rate_range"
)

EXCHANGE_FEATURES = [
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
]

AUCTION_PRICE_FEATURES = [
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]

HISTORICAL_ONLY_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    *EXCHANGE_FEATURES,
    "festival_flag",
    *AUCTION_PRICE_FEATURES,
]

REQUIRED_INPUT_COLUMNS = [
    "week_start_date",
    "grade",
    *HISTORICAL_ONLY_REALS,
    EXCHANGE_RANGE_COLUMN,
]

SHARED_WEEKLY_FEATURES = [
    *EXCHANGE_FEATURES,
    "festival_flag",
    *AUCTION_PRICE_FEATURES,
]

POSITIVE_FEATURES = [
    "demand_kg",
    *EXCHANGE_FEATURES,
    *AUCTION_PRICE_FEATURES,
]

NONNEGATIVE_FEATURES = [
    "production_kg",
    "inventory_kg",
]

ACCEPTED_AUDIT_STATUSES = {
    "valid",
    "verified",
    "approved",
    "corrected_verified",
}


def validate_revised_configuration():
    """
    Validate the model configuration required
    for the revised dataset.

    The previous eight-input feature validator
    is intentionally not called because its
    feature list still uses usd_lkr_rate.
    """

    if ENCODER_LENGTH != 52:
        raise ValueError(
            "The revised model currently requires "
            "a 52-week encoder."
        )

    if PREDICTION_LENGTH != 4:
        raise ValueError(
            "The revised model must forecast "
            "four consecutive weeks."
        )

    if TARGET != "demand_kg":
        raise ValueError(
            "The model target must be demand_kg."
        )

    if list(GROUP_IDS) != ["grade"]:
        raise ValueError(
            "The model must group time series "
            "by tea grade."
        )

    if len(TEA_GRADES) != 5:
        raise ValueError(
            "Five tea grades are required."
        )

    if len(set(TEA_GRADES)) != 5:
        raise ValueError(
            "Duplicate tea grades are not allowed."
        )

    if len(HISTORICAL_ONLY_REALS) != 9:
        raise ValueError(
            "Exactly nine historical numeric "
            "inputs are required."
        )

    if set(KNOWN_FUTURE_REALS).intersection(
        HISTORICAL_ONLY_REALS
    ):
        raise ValueError(
            "Historical-only features cannot "
            "also be known-future features."
        )

    if (
        "usd_lkr_rate" in KNOWN_FUTURE_REALS
        or "usd_lkr_rate" in HISTORICAL_ONLY_REALS
    ):
        raise ValueError(
            "The original numeric USD/LKR "
            "feature is no longer supported."
        )

    validation = pd.Timestamp(
        VALIDATION_START
    )

    testing = pd.Timestamp(
        TEST_START
    )

    if validation >= testing:
        raise ValueError(
            "Validation must start before testing."
        )

    if (
        validation.dayofweek != 0
        or testing.dayofweek != 0
    ):
        raise ValueError(
            "Dataset split dates must be Mondays."
        )


def validate_exchange_ranges(df):
    """
    Verify that the original textual ranges
    agree with the numeric Monday and Sunday
    rate columns.

    Exchange-rate ranges are not assumed to
    have an ascending order: Monday's value
    can be higher or lower than Sunday's.
    """

    ranges = (
        df[EXCHANGE_RANGE_COLUMN]
        .astype("string")
        .str.strip()
    )

    parts = ranges.str.extract(
        r"^([0-9]+(?:\.[0-9]+)?)"
        r"\s*-\s*"
        r"([0-9]+(?:\.[0-9]+)?)$"
    )

    if parts.isna().any().any():
        raise ValueError(
            "One or more USD/LKR ranges "
            "have an invalid format."
        )

    monday = pd.to_numeric(
        parts[0],
        errors="raise",
    ).to_numpy(dtype=float)

    sunday = pd.to_numeric(
        parts[1],
        errors="raise",
    ).to_numpy(dtype=float)

    stored_monday = df[
        "usd_lkr_monday_rate"
    ].to_numpy(dtype=float)

    stored_sunday = df[
        "usd_lkr_sunday_rate"
    ].to_numpy(dtype=float)

    if not np.allclose(
        monday,
        stored_monday,
        rtol=0,
        atol=0.00005,
    ):
        raise ValueError(
            "Monday exchange-rate values "
            "do not match the original ranges."
        )

    if not np.allclose(
        sunday,
        stored_sunday,
        rtol=0,
        atol=0.00005,
    ):
        raise ValueError(
            "Sunday exchange-rate values "
            "do not match the original ranges."
        )

    if (
        (monday <= 0).any()
        or (sunday <= 0).any()
    ):
        raise ValueError(
            "USD/LKR rates must be positive."
        )


def validate_audit_statuses(df):
    """
    Reject unresolved audit statuses when
    such columns are supplied.
    """

    for status_column in [
        "validation_status",
        "auction_validation_status",
        "auction_audit_status",
    ]:
        if status_column not in df.columns:
            continue

        status = (
            df[status_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        if not status.isin(
            ACCEPTED_AUDIT_STATUSES
        ).all():
            invalid = sorted(
                status[
                    ~status.isin(
                        ACCEPTED_AUDIT_STATUSES
                    )
                ]
                .fillna("<missing>")
                .unique()
                .tolist()
            )

            raise ValueError(
                "Unresolved audit statuses in "
                f"{status_column}: {invalid}"
            )


def validate_weekly_continuity(df):
    """
    Verify all five grades have identical,
    consecutive Monday reporting weeks.
    """

    reference_dates = None

    for grade in TEA_GRADES:
        group = df.loc[
            df["grade"] == grade
        ]

        dates = (
            group["week_start_date"]
            .sort_values()
            .reset_index(drop=True)
        )

        if dates.empty:
            raise ValueError(
                f"No observations for {grade}."
            )

        expected_dates = pd.Series(
            pd.date_range(
                start=dates.iloc[0],
                end=dates.iloc[-1],
                freq="W-MON",
            )
        )

        if not dates.equals(
            expected_dates
        ):
            raise ValueError(
                "Missing or irregular reporting "
                f"weeks for {grade}."
            )

        if reference_dates is None:
            reference_dates = dates
        elif not dates.equals(
            reference_dates
        ):
            raise ValueError(
                "Reporting weeks are not "
                "identical for every tea grade."
            )

    return len(reference_dates)


def validate_shared_weekly_values(df):
    """
    Check values shared by every grade
    within a reporting week.
    """

    for feature in SHARED_WEEKLY_FEATURES:
        counts = (
            df.groupby(
                "week_start_date"
            )[feature]
            .nunique(dropna=False)
        )

        if (counts != 1).any():
            raise ValueError(
                f"{feature} differs between "
                "tea grades within the "
                "same reporting week."
            )

    range_counts = (
        df.groupby(
            "week_start_date"
        )[EXCHANGE_RANGE_COLUMN]
        .nunique(dropna=False)
    )

    if (range_counts != 1).any():
        raise ValueError(
            "USD/LKR exchange-rate ranges "
            "differ between tea grades in "
            "the same reporting week."
        )


def load_eight_input_data():
    """
    Load and validate the revised historical
    dataset for TFT training.
    """

    validate_revised_configuration()

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            "Revised training dataset not found: "
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "grade": "string",
            EXCHANGE_RANGE_COLUMN: "string",
        },
    )

    missing_columns = sorted(
        set(REQUIRED_INPUT_COLUMNS)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required dataset columns: "
            f"{missing_columns}"
        )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        format="%Y-%m-%d",
        errors="raise",
    )

    if df["week_start_date"].isna().any():
        raise ValueError(
            "Invalid reporting-week dates."
        )

    if not (
        df["week_start_date"].dt.dayofweek == 0
    ).all():
        raise ValueError(
            "Every reporting week must "
            "start on Monday."
        )

    if df.duplicated(
        subset=[
            "grade",
            "week_start_date",
        ]
    ).any():
        raise ValueError(
            "Duplicate tea-grade and "
            "reporting-week combinations."
        )

    actual_grades = set(
        df["grade"].dropna().tolist()
    )

    if actual_grades != set(TEA_GRADES):
        raise ValueError(
            "Unexpected or missing tea grades: "
            f"{actual_grades}"
        )

    for column in HISTORICAL_ONLY_REALS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        values = df[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Missing or invalid values "
                f"in {column}."
            )

    for column in POSITIVE_FEATURES:
        if (df[column] <= 0).any():
            raise ValueError(
                f"{column} must contain "
                "positive values."
            )

    for column in NONNEGATIVE_FEATURES:
        if (df[column] < 0).any():
            raise ValueError(
                f"{column} cannot contain "
                "negative values."
            )

    if not df[
        "festival_flag"
    ].isin([0, 1]).all():
        raise ValueError(
            "festival_flag must be 0 or 1."
        )

    validate_exchange_ranges(df)
    validate_audit_statuses(df)

    week_count = validate_weekly_continuity(
        df
    )

    validate_shared_weekly_values(
        df
    )

    # Regenerate the integer time index
    # from the actual reporting dates.
    first_date = df[
        "week_start_date"
    ].min()

    offsets = (
        df["week_start_date"]
        - first_date
    ).dt.days

    if (offsets % 7 != 0).any():
        raise ValueError(
            "Reporting weeks are not "
            "seven-day aligned."
        )

    df["time_idx"] = (
        offsets // 7
    ).astype("int64")

    # Calendar features are available
    # in advance for future weeks.
    iso_week = (
        df["week_start_date"]
        .dt.isocalendar()
        .week.astype("int64")
    )

    month = (
        df["week_start_date"]
        .dt.month
    )

    df["week_sin"] = np.sin(
        2.0
        * np.pi
        * iso_week
        / 52.1775
    )

    df["week_cos"] = np.cos(
        2.0
        * np.pi
        * iso_week
        / 52.1775
    )

    df["month_sin"] = np.sin(
        2.0
        * np.pi
        * month
        / 12.0
    )

    df["month_cos"] = np.cos(
        2.0
        * np.pi
        * month
        / 12.0
    )

    for column in [
        *HISTORICAL_ONLY_REALS,
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
    ]:
        df[column] = df[column].astype(
            "float32"
        )

    df["grade"] = df[
        "grade"
    ].astype(str)

    df = (
        df.sort_values(
            [
                "grade",
                "time_idx",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\n=== REVISED TFT DATA VALIDATION ==="
    )

    print(
        "Dataset:",
        DATA_FILE.name,
    )

    print(
        "Records:",
        len(df),
    )

    print(
        "Tea grades:",
        df["grade"].nunique(),
    )

    print(
        "Reporting weeks:",
        week_count,
    )

    print(
        "First week:",
        df[
            "week_start_date"
        ].min().date(),
    )

    print(
        "Last week:",
        df[
            "week_start_date"
        ].max().date(),
    )

    print(
        "\nHistorical input features:",
        len(HISTORICAL_ONLY_REALS),
    )

    for feature in HISTORICAL_ONLY_REALS:
        print(
            feature,
            "min:",
            round(
                float(
                    df[feature].min()
                ),
                4,
            ),
            "max:",
            round(
                float(
                    df[feature].max()
                ),
                4,
            ),
        )

    print(
        "\nDATA VALIDATION: PASS"
    )

    return df


def count_complete_windows(df):
    """
    Count valid forecasting windows
    by all four prediction-target dates.
    """

    counts = {
        "training": 0,
        "validation": 0,
        "testing": 0,
    }

    by_grade = {}

    for grade, group in df.groupby(
        "grade"
    ):
        group = (
            group.sort_values(
                "time_idx"
            )
            .reset_index(drop=True)
        )

        dates = group[
            "week_start_date"
        ]

        grade_counts = {
            "training": 0,
            "validation": 0,
            "testing": 0,
        }

        for origin in range(
            ENCODER_LENGTH - 1,
            len(group)
            - PREDICTION_LENGTH,
        ):
            target_start = dates.iloc[
                origin + 1
            ]

            target_end = dates.iloc[
                origin
                + PREDICTION_LENGTH
            ]

            if target_end < VALIDATION_START:
                partition = "training"

            elif (
                target_start
                >= VALIDATION_START
                and target_end
                < TEST_START
            ):
                partition = "validation"

            elif (
                target_start >= TEST_START
            ):
                partition = "testing"

            else:
                # Exclude horizons crossing
                # chronological split boundaries.
                continue

            counts[partition] += 1

            grade_counts[
                partition
            ] += 1

        by_grade[grade] = (
            grade_counts
        )

    return counts, by_grade


def get_cutoff_index(df, cutoff):
    """
    Resolve a configured cutoff Monday
    into its integer time index.
    """

    matching = df.loc[
        df["week_start_date"] == cutoff,
        "time_idx",
    ].unique()

    if len(matching) != 1:
        raise ValueError(
            "Cutoff date is missing "
            f"from the dataset: {cutoff}"
        )

    return int(
        matching[0]
    )


def create_eight_input_datasets(df):
    """
    Construct chronological training,
    validation, and testing datasets
    for the revised nine-input TFT.
    """

    validation_index = get_cutoff_index(
        df,
        VALIDATION_START,
    )

    test_index = get_cutoff_index(
        df,
        TEST_START,
    )

    training_frame = df.loc[
        df["week_start_date"]
        < VALIDATION_START
    ].copy()

    validation_frame = df.loc[
        df["week_start_date"]
        < TEST_START
    ].copy()

    testing_frame = df.copy()

    training = TimeSeriesDataSet(
        training_frame,
        time_idx="time_idx",
        target=TARGET,
        group_ids=GROUP_IDS,
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=[
            "grade"
        ],
        time_varying_known_reals=KNOWN_FUTURE_REALS,
        time_varying_unknown_reals=(
            HISTORICAL_ONLY_REALS
        ),
        target_normalizer=GroupNormalizer(
            groups=GROUP_IDS,
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = (
        TimeSeriesDataSet.from_dataset(
            training,
            validation_frame,
            min_prediction_idx=(
                validation_index
            ),
            stop_randomization=True,
            predict=False,
        )
    )

    testing = (
        TimeSeriesDataSet.from_dataset(
            training,
            testing_frame,
            min_prediction_idx=(
                test_index
            ),
            stop_randomization=True,
            predict=False,
        )
    )

    return {
        "training": training,
        "validation": validation,
        "testing": testing,
    }


def check_dataset_integrity(
    df,
    datasets,
):
    """
    Verify feature roles and
    chronological prediction windows.
    """

    training = datasets[
        "training"
    ]

    known = set(
        training.time_varying_known_reals
    )

    historical = set(
        training.time_varying_unknown_reals
    )

    if known.intersection(
        historical
    ):
        raise ValueError(
            "Known-future and historical "
            "features overlap."
        )

    missing_historical = (
        set(
            HISTORICAL_ONLY_REALS
        )
        - historical
    )

    if missing_historical:
        raise ValueError(
            "Missing historical features: "
            f"{sorted(missing_historical)}"
        )

    if not set(
        AUCTION_PRICE_FEATURES
    ).issubset(historical):
        raise ValueError(
            "All three auction prices must "
            "remain historical-only inputs."
        )

    if not set(
        EXCHANGE_FEATURES
    ).issubset(historical):
        raise ValueError(
            "Monday and Sunday exchange rates "
            "must be historical-only inputs."
        )

    if (
        "usd_lkr_rate" in known
        or "usd_lkr_rate"
        in historical
    ):
        raise ValueError(
            "Obsolete numeric USD/LKR feature "
            "must not enter the revised model."
        )

    expected, by_grade = (
        count_complete_windows(df)
    )

    actual = {
        name: len(dataset)
        for name, dataset
        in datasets.items()
    }

    print(
        "\n=== CHRONOLOGICAL WINDOW COUNTS ==="
    )

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

        if (
            actual[name]
            != expected[name]
        ):
            raise ValueError(
                f"{name} prediction-window "
                "count mismatch."
            )

        if actual[name] == 0:
            raise ValueError(
                "No complete prediction "
                f"windows for {name}."
            )

    print(
        "\n=== WINDOWS BY TEA GRADE ==="
    )

    for grade, counts in by_grade.items():
        print(
            f"{grade}: "
            f"train={counts['training']}, "
            f"validation={counts['validation']}, "
            f"test={counts['testing']}"
        )

    print(
        "\n=== FEATURE ROLE CHECK ==="
    )

    print(
        "Known-future features:",
        sorted(known),
    )

    print(
        "Historical-only features:",
        sorted(historical),
    )

    print(
        "\nDATASET INTEGRITY: PASS"
    )

    return {
        "window_counts": actual,
        "windows_by_grade": by_grade,
        "known_future_features": (
            sorted(known)
        ),
        "historical_only_features": (
            sorted(historical)
        ),
    }


def main():
    """
    Validate the revised dataset
    without training or modifying
    any existing checkpoints.
    """

    print(
        "\n=== REVISED TFT DATASET BUILDER ==="
    )

    df = load_eight_input_data()

    datasets = (
        create_eight_input_datasets(df)
    )

    check_dataset_integrity(
        df,
        datasets,
    )

    print(
        "\nREVISED TFT DATASETS: PASS"
    )

    print(
        "Existing checkpoints modified: NO"
    )

    print(
        "Model training performed: NO"
    )


if __name__ == "__main__":
    main()
