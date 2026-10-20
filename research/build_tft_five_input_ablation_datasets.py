
"""
Build and validate datasets for the five-input TFT
controlled feature-ablation experiment.

Research question:
Do High, Medium and Low Grown auction-price features
improve four-week tea demand forecasting?

The experiment uses the same:
    - Version 2 weekly dataset
    - Five tea grades
    - 52-week encoder
    - Four-week prediction horizon
    - Chronological train/validation/test boundaries
    - TimeSeriesDataSet construction methodology

Five historical model inputs:
    1. demand_kg
    2. production_kg
    3. inventory_kg
    4. usd_lkr_rate
    5. festival_flag

All auction-price features are excluded from the
model inputs, but existing source-data validation
and audit limitations remain applicable.

No model training is performed by this script.
Existing checkpoints are not modified.
"""

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from build_tft_eight_input_datasets import (
    load_eight_input_data,
    count_complete_windows,
    get_cutoff_index,
)

from tft_five_input_ablation_config import (
    DATA_FILE,
    TARGET,
    GROUP_IDS,
    TEA_GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    VALIDATION_START,
    TEST_START,
    KNOWN_FUTURE_REALS,
    HISTORICAL_ONLY_REALS,
    EXCLUDED_AUCTION_FEATURES,
    validate_feature_configuration,
)


def load_five_input_ablation_data():
    """
    Load exactly the same validated source dataframe
    as the eight-input TFT.

    Reusing the source validation ensures that the
    two experiments share identical observations,
    reporting weeks, time indices and calendar data.

    Auction prices remain present in the dataframe
    for source validation, but they are not supplied
    to the ablation model.
    """

    validate_feature_configuration()

    print("\n=== FIVE-INPUT ABLATION DATA LOADING ===")

    df = load_eight_input_data()

    expected_columns = set(
        [
            "week_start_date",
            "time_idx",
            "grade",
            *KNOWN_FUTURE_REALS,
            *HISTORICAL_ONLY_REALS,
        ]
    )

    missing = sorted(
        expected_columns - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Required ablation columns missing: {missing}"
        )

    print("\n=== FIVE-INPUT FEATURE CONFIGURATION ===")
    print("Historical features:", HISTORICAL_ONLY_REALS)
    print("Excluded auction features:", EXCLUDED_AUCTION_FEATURES)

    print("\nABLATION SOURCE DATA: PASS")

    return df


def create_five_input_ablation_datasets(df):
    """
    Build TFT datasets using the same chronological
    partitioning strategy as the eight-input model.
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
        df["week_start_date"] < VALIDATION_START
    ].copy()

    validation_frame = df.loc[
        df["week_start_date"] < TEST_START
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
        testing_frame,
        min_prediction_idx=test_index,
        stop_randomization=True,
        predict=False,
    )

    return {
        "training": training,
        "validation": validation,
        "testing": testing,
    }


def check_five_input_ablation_integrity(df, datasets):
    """
    Verify feature isolation and chronological
    forecast-window equivalence.
    """

    training = datasets["training"]

    known = set(
        training.time_varying_known_reals
    )

    historical = set(
        training.time_varying_unknown_reals
    )

    model_reals = set(training.reals)

    excluded = set(EXCLUDED_AUCTION_FEATURES)

    if known & historical:
        raise ValueError(
            "Known-future and historical-only "
            "features overlap."
        )

    if historical != set(HISTORICAL_ONLY_REALS):
        raise ValueError(
            "Ablation historical features do not "
            "match the configuration."
        )

    if known != set(KNOWN_FUTURE_REALS):
        raise ValueError(
            "Ablation known-future features do not "
            "match the configuration."
        )

    # Critical ablation check: no auction-price
    # variable may enter the TFT as a model feature.
    leaked = excluded & (
        known | historical | model_reals
    )

    if leaked:
        raise ValueError(
            "Auction-price features entered the "
            f"ablation model: {sorted(leaked)}"
        )

    if TARGET not in historical:
        raise ValueError(
            "Demand target must be historical-only."
        )

    expected, by_grade = count_complete_windows(df)

    actual = {
        name: len(dataset)
        for name, dataset in datasets.items()
    }

    print("\n=== CHRONOLOGICAL WINDOW COUNTS ===")

    for name in [
        "training",
        "validation",
        "testing",
    ]:
        print(
            f"{name}: expected={expected[name]}, "
            f"actual={actual[name]}"
        )

        if actual[name] != expected[name]:
            raise ValueError(
                f"{name} window count mismatch."
            )

        if actual[name] == 0:
            raise ValueError(
                f"No complete {name} windows."
            )

    # Known expected counts for this frozen
    # Version 2 experiment.
    expected_reference_counts = {
        "training": 240,
        "validation": 245,
        "testing": 710,
    }

    if actual != expected_reference_counts:
        raise ValueError(
            "Ablation window counts differ from "
            "the eight-input experiment: "
            f"{actual}"
        )

    print("\n=== WINDOWS BY TEA GRADE ===")

    for grade in TEA_GRADES:
        counts = by_grade[grade]

        print(
            f"{grade}: "
            f"train={counts['training']}, "
            f"validation={counts['validation']}, "
            f"test={counts['testing']}"
        )

    print("\n=== FEATURE ROLE CHECK ===")

    print("Known-future features:", sorted(known))

    print(
        "Historical-only features:",
        sorted(historical),
    )

    print(
        "Excluded auction-price features:",
        sorted(excluded),
    )

    print(
        "Auction features entering model:",
        sorted(leaked),
    )

    print("\n=== COMPARABILITY CHECK ===")
    print("Version 2 source dataset:", DATA_FILE)
    print("Encoder length:", ENCODER_LENGTH)
    print("Prediction length:", PREDICTION_LENGTH)
    print("Validation cutoff:", VALIDATION_START.date())
    print("Test cutoff:", TEST_START.date())
    print("Random auction-feature inclusion: NO")

    print("\nABLATION DATASET INTEGRITY: PASS")

    return {
        "window_counts": actual,
        "windows_by_grade": by_grade,
        "known_future_features": sorted(known),
        "historical_only_features": sorted(historical),
        "excluded_auction_features": sorted(excluded),
        "auction_features_in_model": sorted(leaked),
        "source_dataset": str(DATA_FILE),
        "source_approval_status": "provisional",
        "auction_publication_timestamps_verified": False,
    }


def main():
    """Validate ablation datasets without training."""

    print(
        "\n=== FIVE-INPUT TFT ABLATION DATASET BUILDER ==="
    )

    df = load_five_input_ablation_data()

    datasets = create_five_input_ablation_datasets(df)

    check_five_input_ablation_integrity(
        df,
        datasets,
    )

    print("\nFIVE-INPUT TFT ABLATION DATASETS: PASS")
    print("Auction-price features in model: NO")
    print("Existing model checkpoints modified: NO")
    print("Model training performed: NO")


if __name__ == "__main__":
    main()
