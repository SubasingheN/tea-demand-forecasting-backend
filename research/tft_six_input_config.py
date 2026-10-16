
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "tft_six_input_cleaned.csv"
)

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_six_input"
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

HISTORICAL_ONLY_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_rate",
    "festival_flag",
    "colombo_auction_price_lkr",
]

REQUIRED_INPUT_COLUMNS = [
    "week_start_date",
    "grade",
    *HISTORICAL_ONLY_REALS,
]

MODEL_NAME = "six_input_tft"

BATCH_SIZE = 32

MAX_EPOCHS = 30

LEARNING_RATE = 0.001

RANDOM_SEED = 42


def validate_feature_configuration():
    known = set(KNOWN_FUTURE_REALS)
    historical = set(HISTORICAL_ONLY_REALS)

    if known & historical:
        raise ValueError(
            "Known-future and historical-only "
            "features must not overlap."
        )

    if TARGET not in historical:
        raise ValueError(
            "Forecast target must be historical-only."
        )

    if len(HISTORICAL_ONLY_REALS) != 6:
        raise ValueError(
            "Exactly six historical input features "
            "are required."
        )

    if VALIDATION_START >= TEST_START:
        raise ValueError(
            "Validation must begin before testing."
        )

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}"
        )

    return True


if __name__ == "__main__":
    validate_feature_configuration()

    print("\n=== SIX-INPUT TFT CONFIGURATION ===")
    print("Dataset:", DATA_FILE)
    print("Model:", MODEL_NAME)
    print("Tea grades:", TEA_GRADES)
    print("Encoder length:", ENCODER_LENGTH)
    print("Forecast horizon:", PREDICTION_LENGTH)
    print("Quantiles:", QUANTILES)
    print("Validation begins:", VALIDATION_START.date())
    print("Testing begins:", TEST_START.date())
    print("Known future:", KNOWN_FUTURE_REALS)
    print("Historical only:", HISTORICAL_ONLY_REALS)

    print("\nFEATURE CONFIGURATION: PASS")
