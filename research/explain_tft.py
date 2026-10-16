
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from build_tft_datasets import (
    load_manifests,
    validate_datasets,
)

from train_tft_no_price import (
    prepare_no_price_data,
    create_no_price_datasets,
)


# ============================================================
# STEP 17 - EXPLAINABLE AI ANALYSIS
#
# Selected frozen no-price TFT model.
# Analysis dataset: 2024 validation.
#
# Outputs:
# - Static feature importance
# - Encoder feature importance
# - Decoder feature importance
# - Historical temporal attention
# - JSON explanation summary
#
# No retraining. No test-set model selection.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "research"
    / "outputs"
    / "tft_no_price"
    / "checkpoints"
    / "tft-no-price-epoch09-val470.2676.ckpt"
)

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_explainability"
)

STATIC_FILE = (
    OUTPUT_DIR / "static_feature_importance.csv"
)

ENCODER_FILE = (
    OUTPUT_DIR / "encoder_feature_importance.csv"
)

DECODER_FILE = (
    OUTPUT_DIR / "decoder_feature_importance.csv"
)

ATTENTION_FILE = (
    OUTPUT_DIR / "historical_attention.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "explanation_summary.json"
)

EXPECTED_WINDOWS = 250
ENCODER_LENGTH = 52
BATCH_SIZE = 32


def prepare_data():
    print("\nPREPARING EXPLANATION DATA")
    print("-" * 65)

    df = prepare_no_price_data()
    manifests = load_manifests()

    training, validation, testing = (
        create_no_price_datasets(df)
    )

    validate_datasets(
        {
            "train": training,
            "validation": validation,
            "test": testing,
        },
        manifests,
        df,
    )

    if len(validation) != EXPECTED_WINDOWS:
        raise ValueError(
            "Unexpected number of validation windows: "
            f"{len(validation)}"
        )

    print("Validation windows:", len(validation))
    print("Interpretation period: 2024")
    print("Test predictions evaluated: NO")

    return validation


def load_model():
    print("\nLOADING SELECTED TFT CHECKPOINT")
    print("-" * 65)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    # Load only your own trusted local checkpoint.
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
        weights_only=False,
    )

    model.cpu()
    model.eval()

    print("Model:", type(model).__name__)
    print("Checkpoint:", CHECKPOINT)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    return model


def tensor_to_vector(value, name):
    if isinstance(value, torch.Tensor):
        array = (
            value.detach()
            .cpu()
            .numpy()
        )
    else:
        array = np.asarray(value)

    array = np.asarray(
        array,
        dtype=np.float64,
    ).reshape(-1)

    if not np.isfinite(array).all():
        raise ValueError(
            f"Non-finite interpretation values: {name}"
        )

    if (array < -1e-7).any():
        raise ValueError(
            f"Unexpected negative values: {name}"
        )

    return np.maximum(array, 0.0)


def prepare_importance_table(
    raw_values,
    variable_names,
    category,
):
    values = tensor_to_vector(
        raw_values,
        category,
    )

    names = list(variable_names)

    if len(values) != len(names):
        raise ValueError(
            f"{category} importance length mismatch: "
            f"{len(values)} values vs "
            f"{len(names)} names"
        )

    if len(set(names)) != len(names):
        raise ValueError(
            f"Duplicate {category} variable names"
        )

    total = float(values.sum())

    if total <= 0:
        raise ValueError(
            f"Zero total {category} importance"
        )

    table = pd.DataFrame({
        "feature": names,
        "raw_importance": values,
        "relative_importance_percent": (
            100.0 * values / total
        ),
        "feature_group": category,
    })

    table = table.sort_values(
        "relative_importance_percent",
        ascending=False,
    ).reset_index(drop=True)

    table.insert(
        0,
        "rank",
        np.arange(1, len(table) + 1),
    )

    return table


def prepare_attention_table(raw_attention):
    values = tensor_to_vector(
        raw_attention,
        "historical_attention",
    )

    if len(values) != ENCODER_LENGTH:
        raise ValueError(
            "Expected one attention weight for each "
            f"of the {ENCODER_LENGTH} encoder weeks; "
            f"received {len(values)}. "
            "The installed library may use a "
            "different attention representation."
        )

    total = float(values.sum())

    if total <= 0:
        raise ValueError(
            "Attention weights sum to zero"
        )

    # Relative historical position:
    # -51 is 51 weeks before the latest input.
    # 0 is the latest observed week.
    offsets = np.arange(
        -(ENCODER_LENGTH - 1),
        1,
        dtype=int,
    )

    table = pd.DataFrame({
        "encoder_position": np.arange(
            1,
            ENCODER_LENGTH + 1,
        ),
        "weeks_before_latest_observation": (
            -offsets
        ),
        "raw_attention": values,
        "relative_attention_percent": (
            100.0 * values / total
        ),
    })

    return table


def generate_interpretation(model, validation):
    print("\nGENERATING TFT INTERPRETABILITY OUTPUT")
    print("-" * 65)

    loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # mode="raw" retains the internal model
    # outputs needed by TFT interpret_output().
    predictions = model.predict(
        loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": True,
        },
    )

    print("Raw TFT predictions: GENERATED")

    with torch.no_grad():
        interpretation = model.interpret_output(
            predictions.output,
            reduction="sum",
        )

    print(
        "Interpretation keys:",
        list(interpretation.keys()),
    )

    required_keys = [
        "static_variables",
        "encoder_variables",
        "decoder_variables",
        "attention",
    ]

    missing = [
        key
        for key in required_keys
        if key not in interpretation
    ]

    if missing:
        raise ValueError(
            "TFT interpretation is missing keys: "
            f"{missing}"
        )

    return interpretation


def save_results(
    static_table,
    encoder_table,
    decoder_table,
    attention_table,
    interpretation,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    static_table.to_csv(
        STATIC_FILE,
        index=False,
    )

    encoder_table.to_csv(
        ENCODER_FILE,
        index=False,
    )

    decoder_table.to_csv(
        DECODER_FILE,
        index=False,
    )

    attention_table.to_csv(
        ATTENTION_FILE,
        index=False,
    )

    highest_attention = (
        attention_table
        .sort_values(
            "relative_attention_percent",
            ascending=False,
        )
        .head(10)
    )

    report = {
        "status": "PASS",
        "step": 17,
        "model": "no_price_tft",
        "checkpoint": str(CHECKPOINT),
        "interpretation_partition": (
            "2024_validation"
        ),
        "forecast_horizon_weeks": 4,
        "encoder_history_weeks": ENCODER_LENGTH,
        "validation_forecast_windows": (
            EXPECTED_WINDOWS
        ),
        "auction_price_inputs_in_model": False,
        "static_feature_importance": (
            static_table[
                [
                    "feature",
                    "relative_importance_percent",
                ]
            ].to_dict(orient="records")
        ),
        "encoder_feature_importance": (
            encoder_table[
                [
                    "feature",
                    "relative_importance_percent",
                ]
            ].to_dict(orient="records")
        ),
        "decoder_feature_importance": (
            decoder_table[
                [
                    "feature",
                    "relative_importance_percent",
                ]
            ].to_dict(orient="records")
        ),
        "top_historical_attention_positions": (
            highest_attention[
                [
                    "weeks_before_latest_observation",
                    "relative_attention_percent",
                ]
            ].to_dict(orient="records")
        ),
        "output_files": {
            "static": str(STATIC_FILE),
            "encoder": str(ENCODER_FILE),
            "decoder": str(DECODER_FILE),
            "attention": str(ATTENTION_FILE),
        },
        "limitations": [
            (
                "Importance values describe the "
                "TFT variable-selection network, "
                "not causal feature effects."
            ),
            (
                "Static, encoder and decoder "
                "importance percentages are "
                "normalized within their own groups "
                "and should not be compared directly "
                "between groups."
            ),
            (
                "TFT attention is an internal "
                "temporal weighting mechanism, "
                "not a causal explanation."
            ),
            (
                "Attention is aggregated over the "
                "2024 validation windows."
            ),
            (
                "No-price TFT does not use "
                "auction-price inputs."
            ),
            (
                "This analysis does not change "
                "the frozen test evaluation."
            ),
        ],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def print_importance(title, table):
    print(f"\n{title}")
    print("-" * 65)

    print(
        table[
            [
                "rank",
                "feature",
                "relative_importance_percent",
            ]
        ].round(3).to_string(index=False)
    )


def main():
    print(
        "PHASE 3 - STEP 17: "
        "EXPLAINABLE TFT ANALYSIS"
    )
    print("=" * 65)

    print("PyTorch:", torch.__version__)

    validation = prepare_data()
    model = load_model()

    interpretation = generate_interpretation(
        model,
        validation,
    )

    static_names = list(
        model.static_variables
    )

    encoder_names = list(
        model.encoder_variables
    )

    decoder_names = list(
        model.decoder_variables
    )

    static_table = prepare_importance_table(
        interpretation["static_variables"],
        static_names,
        "static",
    )

    encoder_table = prepare_importance_table(
        interpretation["encoder_variables"],
        encoder_names,
        "encoder",
    )

    decoder_table = prepare_importance_table(
        interpretation["decoder_variables"],
        decoder_names,
        "decoder",
    )

    attention_table = prepare_attention_table(
        interpretation["attention"]
    )

    save_results(
        static_table,
        encoder_table,
        decoder_table,
        attention_table,
        interpretation,
    )

    print_importance(
        "STATIC FEATURE IMPORTANCE",
        static_table,
    )

    print_importance(
        "ENCODER (HISTORICAL) FEATURE IMPORTANCE",
        encoder_table,
    )

    print_importance(
        "DECODER (KNOWN FUTURE) FEATURE IMPORTANCE",
        decoder_table,
    )

    print("\nTOP 10 HISTORICAL ATTENTION POSITIONS")
    print("-" * 65)

    print(
        attention_table.sort_values(
            "relative_attention_percent",
            ascending=False,
        )[
            [
                "weeks_before_latest_observation",
                "relative_attention_percent",
            ]
        ].head(10).round(3).to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print(STATIC_FILE)
    print(ENCODER_FILE)
    print(DECODER_FILE)
    print(ATTENTION_FILE)
    print(SUMMARY_FILE)

    print("\nMODEL RETRAINING: NONE")
    print("TEST METRICS MODIFIED: NO")
    print("TFT EXPLAINABILITY ANALYSIS: PASS")


if __name__ == "__main__":
    main()
