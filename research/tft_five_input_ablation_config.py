
"""
Configuration for controlled TFT feature ablation.

Research question:
Do three elevation-specific auction-price features
improve four-week tea demand forecasts?

This model uses the same Version 2 factory dataset
and chronological cutoffs as the eight-input TFT,
but deliberately excludes all auction prices.

No existing TFT checkpoints are modified.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Same dataset used by the eight-input TFT.
DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "tft_eight_input_cleaned.csv"
)

# Independent ablation output directory.
OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_five_input_ablation"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MODEL_NAME = "five_input_ablation_tft"

TEA_GRADES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]

TARGET = "demand_kg"

GROUP_IDS = ["grade"]

ENCODER_LENGTH = 52

PREDICTION_LENGTH = 4

QUANTILES = [0.1, 0.5, 0.9]

VALIDATION_START = pd.Timestamp("2023-01-02")

TEST_START = pd.Timestamp("2024-01-01")

KNOWN_FUTURE_REALS = [
    "time_idx",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
]

# No auction prices in this experiment.
HISTORICAL_ONLY_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_rate",
    "festival_flag",
]

REQUIRED_INPUT_COLUMNS = [
    "week_start_date",
    "grade",
    *HISTORICAL_ONLY_REALS,
]

EXCLUDED_AUCTION_FEATURES = [
    "colombo_auction_price_lkr",
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]

BATCH_SIZE = 32

MAX_EPOCHS = 30

LEARNING_RATE = 0.001

RANDOM_SEED = 42


def validate_feature_configuration():
    """Validate controlled ablation settings."""

    known = set(KNOWN_FUTURE_REALS)
    historical = set(HISTORICAL_ONLY_REALS)

    if known & historical:
        raise ValueError(
            "Known-future and historical-only "
            "features must not overlap."
        )

    if TARGET not in historical:
        raise ValueError(
            "Demand target must be historical-only."
        )

    if len(HISTORICAL_ONLY_REALS) != 5:
        raise ValueError(
            "Ablation model requires exactly "
            "five historical features."
        )

    if set(EXCLUDED_AUCTION_FEATURES) & (
        known | historical
    ):
        raise ValueError(
            "Auction-price features must be "
            "excluded from the ablation model."
        )

    if VALIDATION_START >= TEST_START:
        raise ValueError(
            "Invalid chronological split dates."
        )

    if ENCODER_LENGTH != 52:
        raise ValueError(
            "Encoder must match the eight-input TFT."
        )

    if PREDICTION_LENGTH != 4:
        raise ValueError(
            "Forecast horizon must match the eight-input TFT."
        )

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Training dataset missing: {DATA_FILE}"
        )

    eight_input_checkpoint_dir = (
        ROOT
        / "research"
        / "outputs"
        / "tft_eight_input"
        / "checkpoints"
    )

    six_input_checkpoint_dir = (
        ROOT
        / "research"
        / "outputs"
        / "tft_six_input"
        / "checkpoints"
    )

    if CHECKPOINT_DIR.resolve() in {
        eight_input_checkpoint_dir.resolve(),
        six_input_checkpoint_dir.resolve(),
    }:
        raise ValueError(
            "Ablation checkpoint directory overlaps "
            "with an existing model."
        )

    return True


if __name__ == "__main__":
    validate_feature_configuration()

    print("\n=== FIVE-INPUT TFT ABLATION ===")
    print("Model:", MODEL_NAME)
    print("Dataset:", DATA_FILE)
    print("Tea grades:", TEA_GRADES)
    print("Encoder weeks:", ENCODER_LENGTH)
    print("Forecast weeks:", PREDICTION_LENGTH)
    print("Quantiles:", QUANTILES)
    print("Validation begins:", VALIDATION_START.date())
    print("Test begins:", TEST_START.date())
    print("Historical features:", HISTORICAL_ONLY_REALS)
    print("Excluded prices:", EXCLUDED_AUCTION_FEATURES)
    print("Checkpoint directory:", CHECKPOINT_DIR)

    print("\nABLATION CONFIGURATION: PASS")
