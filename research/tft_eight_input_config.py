
"""
Configuration for the revised nine-input
Tea Demand Forecasting TFT model.

Nine historical inputs:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_monday_rate
    usd_lkr_sunday_rate
    festival_flag
    auction_price_high_lkr
    auction_price_medium_lkr
    auction_price_low_lkr

The original USD/LKR range is retained
in the dataset for reference, but the model
uses its two numeric endpoint values.

Previous six-input and eight-input TFT
checkpoints remain unchanged.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "tft_eight_input_model_ready_ranges_2021_2026.csv"
)

# Separate output directory protects
# the previously trained model files.
OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_nine_input"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

TEA_GRADES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]

TARGET = "demand_kg"

GROUP_IDS = ["grade"]

# Historical observations used by the TFT.
ENCODER_LENGTH = 52

# Number of future weeks to predict.
PREDICTION_LENGTH = 4

# Lower, expected and upper demand quantiles.
QUANTILES = [0.1, 0.5, 0.9]

# Preserve the existing chronological
# validation and testing boundaries.
VALIDATION_START = pd.Timestamp("2023-01-02")

TEST_START = pd.Timestamp("2024-01-01")

# Future calendar values are known in advance.
KNOWN_FUTURE_REALS = [
    "time_idx",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
]

EXCHANGE_RATE_FEATURES = [
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
]

AUCTION_PRICE_FEATURES = [
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]

# These features must only use observed
# historical values, never invented
# future values.
HISTORICAL_ONLY_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    *EXCHANGE_RATE_FEATURES,
    "festival_flag",
    *AUCTION_PRICE_FEATURES,
]

REQUIRED_INPUT_COLUMNS = [
    "week_start_date",
    "grade",
    "usd_lkr_exchange_rate_range",
    *HISTORICAL_ONLY_REALS,
]

# Distinct model identity avoids mixing
# new nine-input forecasts with existing
# eight-input forecasts.
MODEL_NAME = "nine_input_tft"

BATCH_SIZE = 32

MAX_EPOCHS = 30

LEARNING_RATE = 0.001

RANDOM_SEED = 42


def validate_feature_configuration():
    """
    Validate the revised feature definitions,
    dataset path, and forecasting settings.
    """

    known = set(KNOWN_FUTURE_REALS)
    historical = set(HISTORICAL_ONLY_REALS)

    if len(known) != len(KNOWN_FUTURE_REALS):
        raise ValueError(
            "Duplicate known-future features."
        )

    if len(historical) != len(HISTORICAL_ONLY_REALS):
        raise ValueError(
            "Duplicate historical-only features."
        )

    if known & historical:
        raise ValueError(
            "Known-future and historical-only "
            "features must not overlap."
        )

    if TARGET not in historical:
        raise ValueError(
            "The forecasting target must be "
            "a historical-only feature."
        )

    if len(HISTORICAL_ONLY_REALS) != 9:
        raise ValueError(
            "Exactly nine historical features "
            "are required."
        )

    if len(EXCHANGE_RATE_FEATURES) != 2:
        raise ValueError(
            "Both Monday and Sunday USD/LKR "
            "features are required."
        )

    if not set(EXCHANGE_RATE_FEATURES).issubset(
        historical
    ):
        raise ValueError(
            "Exchange-rate features must be "
            "historical-only."
        )

    if len(AUCTION_PRICE_FEATURES) != 3:
        raise ValueError(
            "Exactly three elevation-specific "
            "auction prices are required."
        )

    if not set(AUCTION_PRICE_FEATURES).issubset(
        historical
    ):
        raise ValueError(
            "Auction prices must be "
            "historical-only features."
        )

    if "usd_lkr_rate" in historical:
        raise ValueError(
            "The obsolete numeric USD/LKR "
            "feature must not be used."
        )

    if ENCODER_LENGTH != 52:
        raise ValueError(
            "The encoder must contain "
            "52 historical weeks."
        )

    if PREDICTION_LENGTH != 4:
        raise ValueError(
            "The forecasting horizon must "
            "contain four future weeks."
        )

    if TARGET != "demand_kg":
        raise ValueError(
            "The target must be demand_kg."
        )

    if GROUP_IDS != ["grade"]:
        raise ValueError(
            "The series must be grouped "
            "by tea grade."
        )

    if set(TEA_GRADES) != {
        "BOP",
        "BOPF",
        "Dust",
        "OP",
        "Pekoe",
    }:
        raise ValueError(
            "The required tea grades are "
            "BOP, BOPF, Dust, OP and Pekoe."
        )

    if (
        VALIDATION_START >= TEST_START
    ):
        raise ValueError(
            "Validation must begin "
            "before testing."
        )

    if (
        VALIDATION_START.dayofweek != 0
        or TEST_START.dayofweek != 0
    ):
        raise ValueError(
            "Split dates must be Mondays."
        )

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            "Revised training dataset "
            f"not found: {DATA_FILE}"
        )

    return True


if __name__ == "__main__":
    validate_feature_configuration()

    print(
        "\n=== NINE-INPUT TFT CONFIGURATION ==="
    )

    print("Dataset:", DATA_FILE)
    print("Model:", MODEL_NAME)
    print("Tea grades:", TEA_GRADES)
    print("Encoder length:", ENCODER_LENGTH)
    print("Forecast horizon:", PREDICTION_LENGTH)
    print("Quantiles:", QUANTILES)

    print(
        "Validation begins:",
        VALIDATION_START.date(),
    )

    print(
        "Testing begins:",
        TEST_START.date(),
    )

    print(
        "Known-future features:",
        KNOWN_FUTURE_REALS,
    )

    print(
        "Historical-only features:",
        HISTORICAL_ONLY_REALS,
    )

    print(
        "\nFEATURE CONFIGURATION: PASS"
    )
