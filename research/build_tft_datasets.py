
from pathlib import Path
import json
import warnings

import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


# ============================================================
# PHASE 2 - STEP 9
# BUILD TFT TRAINING, VALIDATION, AND TEST DATASETS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "tft_weekly_features.csv"
)

MANIFEST_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "tft_datasets"
)

ENCODER_LENGTH = 52
PREDICTION_LENGTH = 4

GRADES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

# These features can be calculated in advance
# for the upcoming four weeks.
KNOWN_FUTURE_REALS = [
    "time_idx",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
]

# These are historical observations.
# Actual future values must never be treated as
# known-future features when forecasting.
HISTORICAL_ONLY_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "auction_price_high_lkr_per_kg_input",
    "auction_price_medium_lkr_per_kg_input",
    "auction_price_high_lkr_per_kg_available",
    "auction_price_medium_lkr_per_kg_available",
]

PRICE_INPUTS = [
    "auction_price_high_lkr_per_kg_input",
    "auction_price_medium_lkr_per_kg_input",
]

PRICE_MASKS = [
    "auction_price_high_lkr_per_kg_available",
    "auction_price_medium_lkr_per_kg_available",
]

REQUIRED_COLUMNS = list(
    dict.fromkeys(
        [
            "week_start_date",
            "grade",
        ]
        + KNOWN_FUTURE_REALS
        + HISTORICAL_ONLY_REALS
    )
)

MANIFEST_FILES = {
    "train": "tft_train_window_manifest.csv",
    "validation": "tft_validation_window_manifest.csv",
    "test": "tft_test_window_manifest.csv",
}


# ============================================================
# LOAD AND VALIDATE FEATURE DATA
# ============================================================

def load_features():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required TFT columns: "
            + ", ".join(missing)
        )

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        errors="raise",
    ).dt.normalize()

    df["grade"] = (
        df["grade"]
        .astype(str)
        .str.strip()
    )

    if len(df) != 5680:
        raise ValueError(
            f"Expected 5680 rows, found {len(df)}"
        )

    if set(df["grade"]) != set(GRADES):
        raise ValueError(
            "Unexpected tea grades: "
            + str(sorted(df["grade"].unique()))
        )

    keys = [
        "grade",
        "week_start_date",
    ]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate grade/week combinations detected."
        )

    if df["week_start_date"].dt.weekday.ne(0).any():
        raise ValueError(
            "All week_start_date values must be Mondays."
        )

    df = df.sort_values(keys).reset_index(drop=True)

    # Convert numerical model inputs to float32.
    # This also fixes the GroupNormalizer softplus
    # error caused by integer demand values.
    for column in (
        KNOWN_FUTURE_REALS
        + HISTORICAL_ONLY_REALS
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if df[column].isna().any():
            raise ValueError(
                f"Missing numerical values in {column}"
            )

        df[column] = df[column].astype("float32")

    df["demand_kg"] = df["demand_kg"].astype(
        "float32"
    )

    if df["demand_kg"].le(0).any():
        raise ValueError(
            "Demand values must be positive for "
            "the configured softplus normalizer."
        )

    for column in [
        "production_kg",
        "inventory_kg",
    ]:
        if df[column].lt(0).any():
            raise ValueError(
                f"Negative values found in {column}"
            )

    # Verify price availability masks.
    for column in PRICE_MASKS:
        if not df[column].isin([0.0, 1.0]).all():
            raise ValueError(
                f"Invalid binary availability mask: {column}"
            )

    for price_column, mask_column in zip(
        PRICE_INPUTS,
        PRICE_MASKS,
    ):
        unavailable = df[mask_column].eq(0.0)
        available = df[mask_column].eq(1.0)

        # A zero is permitted only as a masked
        # numerical placeholder, not as a real price.
        if df.loc[available, price_column].le(0).any():
            raise ValueError(
                f"Nonpositive available price: {price_column}"
            )

        if df.loc[unavailable, price_column].ne(0).any():
            raise ValueError(
                f"Missing-price placeholders are "
                f"inconsistent: {price_column}"
            )

    # The time_idx column is numerical but represents
    # exact integer weekly positions.
    time_values = df["time_idx"]

    if not time_values.eq(time_values.round()).all():
        raise ValueError(
            "time_idx contains non-integer values."
        )

    df["time_idx"] = df["time_idx"].astype(
        "int64"
    )

    expected_dates = pd.date_range(
        "2005-01-03",
        "2026-10-05",
        freq="W-MON",
    )

    if len(expected_dates) != 1136:
        raise ValueError(
            "Unexpected expected timeline length."
        )

    for grade in GRADES:
        subset = (
            df[df["grade"] == grade]
            .sort_values("week_start_date")
        )

        if len(subset) != 1136:
            raise ValueError(
                f"Incorrect number of weeks for {grade}"
            )

        actual_dates = pd.DatetimeIndex(
            subset["week_start_date"]
        )

        if not actual_dates.equals(expected_dates):
            raise ValueError(
                f"Incomplete weekly timeline for {grade}"
            )

        expected_indices = list(range(1136))

        if (
            subset["time_idx"].tolist()
            != expected_indices
        ):
            raise ValueError(
                f"Incorrect time_idx sequence for {grade}"
            )

    if (
        set(KNOWN_FUTURE_REALS)
        & set(HISTORICAL_ONLY_REALS)
    ):
        raise ValueError(
            "Known and historical-only feature lists overlap."
        )

    # Exclude lagged/rolling demand columns created
    # in Step 7. They are not needed for the initial
    # TFT implementation and must never leak into
    # the forecast decoder.
    model_df = df[
        REQUIRED_COLUMNS
    ].copy()

    print("Feature rows:", len(model_df))
    print("Model columns:", len(model_df.columns))
    print("Tea grades:", len(GRADES))

    return model_df


# ============================================================
# LOAD STEP 8 WINDOW MANIFESTS
# ============================================================

def load_manifests():
    manifests = {}

    for partition, filename in MANIFEST_FILES.items():
        path = MANIFEST_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Step 8 manifest: {path}"
            )

        manifest = pd.read_csv(
            path,
            dtype={"grade": str},
        )

        required = [
            "grade",
            "forecast_origin",
            "target_start",
            "target_end",
        ]

        missing = [
            column
            for column in required
            if column not in manifest.columns
        ]

        if missing:
            raise ValueError(
                f"{partition} manifest missing: {missing}"
            )

        if manifest.empty:
            raise ValueError(
                f"{partition} manifest is empty."
            )

        for column in [
            "forecast_origin",
            "target_start",
            "target_end",
        ]:
            manifest[column] = pd.to_datetime(
                manifest[column],
                errors="raise",
            ).dt.normalize()

        if manifest.duplicated(
            ["grade", "target_start"]
        ).any():
            raise ValueError(
                f"Duplicate {partition} forecast windows."
            )

        manifests[partition] = manifest

    return manifests


# ============================================================
# FIND CHRONOLOGICAL SPLIT INDICES
# ============================================================

def get_start_index(df, cutoff):
    subset = (
        df[
            df["week_start_date"] >= cutoff
        ]
        .sort_values("week_start_date")
    )

    if subset.empty:
        raise ValueError(
            f"No observations after cutoff: {cutoff}"
        )

    return int(subset.iloc[0]["time_idx"])


# ============================================================
# CONSTRUCT TFT DATASETS
# ============================================================

def create_datasets(model_df):
    validation_start_idx = get_start_index(
        model_df,
        VALIDATION_START,
    )

    test_start_idx = get_start_index(
        model_df,
        TEST_START,
    )

    print("\nSPLIT INDICES")
    print(
        "Validation start index:",
        validation_start_idx,
    )
    print(
        "Test start index:",
        test_start_idx,
    )

    # Train targets must end before 2024.
    train_df = model_df[
        model_df["week_start_date"]
        < VALIDATION_START
    ].copy()

    # Validation targets lie in 2024.
    # Historical encoder context can come from
    # the preceding training period.
    validation_df = model_df[
        model_df["week_start_date"]
        < TEST_START
    ].copy()

    # Test targets lie in 2025 and 2026.
    # Historical encoder context may come from
    # the preceding training/validation periods.
    test_df = model_df.copy()

    print("\nSOURCE ROWS BY DATASET")
    print("Training source rows:", len(train_df))
    print(
        "Validation context rows:",
        len(validation_df),
    )
    print("Test context rows:", len(test_df))

    # Fit normalization and categorical encoders
    # using training observations only.
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="demand_kg",
        group_ids=["grade"],

        max_encoder_length=ENCODER_LENGTH,
        min_encoder_length=ENCODER_LENGTH,

        max_prediction_length=PREDICTION_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,

        static_categoricals=[
            "grade",
        ],

        time_varying_known_reals=(
            KNOWN_FUTURE_REALS
        ),

        time_varying_unknown_reals=(
            HISTORICAL_ONLY_REALS
        ),

        target_normalizer=GroupNormalizer(
            groups=["grade"],
            transformation="softplus",
        ),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,

        allow_missing_timesteps=False,
        randomize_length=False,
    )

    # Reuse training encoders/scalers.
    # Limit available validation data to 2024,
    # preventing any forecast horizon from
    # crossing into the test period.
    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_df,
        min_prediction_idx=validation_start_idx,
        stop_randomization=True,
        predict=False,
    )

    # Test samples begin in 2025 and retain
    # earlier observations only as encoder context.
    testing = TimeSeriesDataSet.from_dataset(
        training,
        test_df,
        min_prediction_idx=test_start_idx,
        stop_randomization=True,
        predict=False,
    )

    return training, validation, testing


# ============================================================
# VALIDATE FORECAST WINDOW ALIGNMENT
# ============================================================

def validate_datasets(datasets, manifests, model_df):
    results = {}

    # Map each integer time index back to its date.
    index_to_date = {
        int(time_idx): pd.Timestamp(date)
        for time_idx, date in (
            model_df[
                ["time_idx", "week_start_date"]
            ]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    }

    for name, dataset in datasets.items():
        expected_manifest = manifests[name]
        expected_count = len(expected_manifest)
        actual_count = len(dataset)

        if actual_count != expected_count:
            raise ValueError(
                f"{name}: Step 8 expected "
                f"{expected_count} windows, "
                f"but TFT produced {actual_count}."
            )

        decoded = dataset.decoded_index.copy()

        required_decoded = [
            "grade",
            "time_idx_first_prediction",
        ]

        missing_decoded = [
            col
            for col in required_decoded
            if col not in decoded.columns
        ]

        if missing_decoded:
            raise ValueError(
                f"{name}: unexpected decoded_index "
                f"columns: {list(decoded.columns)}"
            )

        if len(decoded) != expected_count:
            raise ValueError(
                f"{name}: decoded window count mismatch."
            )

        actual_pairs = set()

        for row in decoded.itertuples(index=False):
            grade = str(
                getattr(row, "grade")
            )

            start_idx = int(
                getattr(
                    row,
                    "time_idx_first_prediction",
                )
            )

            if start_idx not in index_to_date:
                raise ValueError(
                    f"{name}: invalid prediction time index."
                )

            start_date = index_to_date[start_idx]

            actual_pairs.add(
                (grade, start_date.strftime("%Y-%m-%d"))
            )

        expected_pairs = {
            (
                str(row.grade),
                row.target_start.strftime("%Y-%m-%d"),
            )
            for row in expected_manifest.itertuples(
                index=False
            )
        }

        if actual_pairs != expected_pairs:
            missing_pairs = sorted(
                expected_pairs - actual_pairs
            )

            extra_pairs = sorted(
                actual_pairs - expected_pairs
            )

            raise ValueError(
                f"{name}: TFT windows do not match "
                f"Step 8 manifest.\n"
                f"Missing examples: {missing_pairs[:5]}\n"
                f"Extra examples: {extra_pairs[:5]}"
            )

        if len(actual_pairs) != actual_count:
            raise ValueError(
                f"{name}: duplicate TFT forecast origins."
            )

        results[name] = {
            "expected_windows": expected_count,
            "actual_windows": actual_count,
            "manifest_alignment": "PASS",
        }

        print(
            f"{name}: {actual_count} "
            "windows - PASS"
        )

    return results


# ============================================================
# CHECK ENCODER AND DECODER CONFIGURATION
# ============================================================

def validate_feature_roles(training):
    known_set = set(
        training.time_varying_known_reals
    )

    unknown_set = set(
        training.time_varying_unknown_reals
    )

    if known_set.intersection(unknown_set):
        raise ValueError(
            "Known and unknown TFT features overlap."
        )

    forbidden_future = set(
        HISTORICAL_ONLY_REALS
    )

    leaked = known_set.intersection(
        forbidden_future
    )

    if leaked:
        raise ValueError(
            "Future feature leakage detected: "
            + ", ".join(sorted(leaked))
        )

    if "demand_kg" in known_set:
        raise ValueError(
            "Target demand cannot be future-known."
        )

    print("\nFEATURE ROLE VALIDATION")
    print("Known future features:", sorted(known_set))
    print("Historical-only features:", sorted(unknown_set))
    print("Feature role validation: PASS")


def inspect_sample(training):
    loader = training.to_dataloader(
        train=False,
        batch_size=2,
        num_workers=0,
    )

    x, y = next(iter(loader))

    print("\nSAMPLE TFT BATCH")
    print(
        "Encoder target shape:",
        tuple(x["encoder_target"].shape),
    )
    print(
        "Decoder target shape:",
        tuple(y[0].shape),
    )
    print(
        "Encoder lengths:",
        x["encoder_lengths"].tolist(),
    )
    print(
        "Decoder lengths:",
        x["decoder_lengths"].tolist(),
    )

    if x["encoder_target"].shape[1] != ENCODER_LENGTH:
        raise ValueError(
            "Incorrect encoder length."
        )

    if y[0].shape[1] != PREDICTION_LENGTH:
        raise ValueError(
            "Incorrect decoder target length."
        )

    if not torch.all(
        x["encoder_lengths"] == ENCODER_LENGTH
    ):
        raise ValueError(
            "Variable encoder lengths detected."
        )

    if not torch.all(
        x["decoder_lengths"] == PREDICTION_LENGTH
    ):
        raise ValueError(
            "Variable decoder lengths detected."
        )

    print("Sample shape validation: PASS")


# ============================================================
# SAVE REPRODUCIBLE DATASET CONFIGURATION
# ============================================================

def save_outputs(training, results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters_path = (
        OUTPUT_DIR
        / "tft_dataset_parameters.pt"
    )

    report_path = (
        OUTPUT_DIR
        / "tft_dataset_readiness.json"
    )

    # Contains fitted training encoders,
    # scalers, and dataset construction settings.
    # Load this file only from trusted sources.
    torch.save(
        training.get_parameters(),
        parameters_path,
    )

    report = {
        "status": "PASS",
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders, "
            "including unfulfilled orders"
        ),
        "grades": GRADES,
        "encoder_length": ENCODER_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "validation_start": "2024-01-01",
        "test_start": "2025-01-01",
        "normalizer": "GroupNormalizer",
        "normalizer_transformation": "softplus",
        "normalizer_fitted_on": "training",
        "target_dtype": "float32",
        "known_future_reals": KNOWN_FUTURE_REALS,
        "historical_only_reals": HISTORICAL_ONLY_REALS,
        "datasets": results,
        "dataset_parameters_file": str(
            parameters_path
        ),
        "research_notes": [
            (
                "Demand is total customer orders, "
                "including unfulfilled demand."
            ),
            (
                "Auction price inputs are "
                "quotation-derived historical proxies."
            ),
            (
                "Missing auction prices use zero "
                "placeholders with availability masks."
            ),
            (
                "Future production, inventory, "
                "auction prices, and demand are "
                "not declared as future-known features."
            ),
            (
                "The held-out test period must not "
                "be used to tune model hyperparameters."
            ),
            (
                "PyTorch Forecasting dataset batches "
                "may contain future unknown-variable "
                "tensors for training targets; TFT "
                "feature roles determine which are "
                "used as decoder predictors."
            ),
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return parameters_path, report_path


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print(
        "PHASE 2 - STEP 9: BUILD TFT DATASETS"
    )
    print("=" * 65)

    print("PyTorch:", torch.__version__)

    try:
        import pytorch_forecasting

        print(
            "PyTorch Forecasting:",
            pytorch_forecasting.__version__,
        )
    except ImportError:
        pass

    print("\nLOADING TFT FEATURES")

    model_df = load_features()

    print("\nLOADING CHRONOLOGICAL MANIFESTS")

    manifests = load_manifests()

    print("\nCONSTRUCTING TFT DATASETS")

    training, validation, testing = (
        create_datasets(model_df)
    )

    datasets = {
        "train": training,
        "validation": validation,
        "test": testing,
    }

    print("\nVALIDATING TFT WINDOW ALIGNMENT")

    results = validate_datasets(
        datasets,
        manifests,
        model_df,
    )

    validate_feature_roles(training)

    inspect_sample(training)

    parameters_path, report_path = save_outputs(
        training,
        results,
    )

    print("\nFINAL TFT DATASET COUNTS")

    for partition, result in results.items():
        print(
            f"{partition}: "
            f"{result['actual_windows']} "
            "windows"
        )

    print("\nOUTPUT FILES")
    print(parameters_path)
    print(report_path)

    print("\nTFT DATASET CONSTRUCTION: PASS")

    print(
        "\nNEXT STEP: "
        "Configure the Temporal Fusion "
        "Transformer training model."
    )


if __name__ == "__main__":
    main()
