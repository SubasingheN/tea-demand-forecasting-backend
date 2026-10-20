
"""
Final production training for the selected nine-input TFT.

Selected configuration: Candidate B

Historical inputs:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_monday_rate
    usd_lkr_sunday_rate
    festival_flag
    auction_price_high_lkr
    auction_price_medium_lkr
    auction_price_low_lkr

Training:
    All approved historical observations
    52 historical weeks
    4 forecast weeks
    5 tea grades
    14 fixed epochs

Safeguards:
    No checkpoint selection using the test set
    No modification of existing model checkpoints
    No changes to existing factory forecasts
    No automatic model deployment

The final model is saved separately under:
    research/outputs/tft_nine_input/production
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from build_eight_input_dataset import (
    load_eight_input_data,
)

from tft_eight_input_config import (
    DATA_FILE,
    OUTPUT_DIR,
    TARGET,
    GROUP_IDS,
    TEA_GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    KNOWN_FUTURE_REALS,
    HISTORICAL_ONLY_REALS,
    BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
    validate_feature_configuration,
)


PRODUCTION_DIR = OUTPUT_DIR / "production"

CHECKPOINT_DIR = PRODUCTION_DIR / "checkpoints"

CHECKPOINT_FILE = (
    CHECKPOINT_DIR / "nine_input_tft_production.ckpt"
)

DATASET_PARAMETERS_FILE = (
    PRODUCTION_DIR / "dataset_parameters.pt"
)

METADATA_FILE = (
    PRODUCTION_DIR / "production_metadata.json"
)

TRAINING_EPOCHS = 14

HIDDEN_SIZE = 32
HIDDEN_CONTINUOUS_SIZE = 16
ATTENTION_HEADS = 4
DROPOUT = 0.20

GRADIENT_CLIP_VAL = 0.1


def calculate_file_sha256(path):
    """Return a SHA256 checksum for a source file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_environment():
    """Validate configuration and protect outputs."""

    validate_feature_configuration()

    if ENCODER_LENGTH != 52:
        raise ValueError(
            "Production encoder must use 52 weeks."
        )

    if PREDICTION_LENGTH != 4:
        raise ValueError(
            "Production forecast horizon must be four weeks."
        )

    if len(HISTORICAL_ONLY_REALS) != 9:
        raise ValueError(
            "Production model requires nine historical inputs."
        )

    if len(TEA_GRADES) != 5:
        raise ValueError(
            "Production model requires five tea grades."
        )

    if list(QUANTILES) != [0.1, 0.5, 0.9]:
        raise ValueError(
            "Expected P10, P50 and P90 quantiles."
        )

    protected_files = [
        CHECKPOINT_FILE,
        DATASET_PARAMETERS_FILE,
        METADATA_FILE,
    ]

    for path in protected_files:
        if path.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing "
                f"production artifact: {path}"
            )

    if PRODUCTION_DIR.exists() and any(
        PRODUCTION_DIR.iterdir()
    ):
        raise FileExistsError(
            "Production output directory already "
            "contains files. Existing artifacts "
            "must not be overwritten automatically."
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.seed_everything(
        RANDOM_SEED,
        workers=True,
    )

    print("\n=== FINAL NINE-INPUT TFT TRAINING ===")
    print("Dataset:", DATA_FILE)
    print("Training epochs:", TRAINING_EPOCHS)
    print("Historical weeks:", ENCODER_LENGTH)
    print("Forecast weeks:", PREDICTION_LENGTH)
    print("Device: CPU")


def prepare_full_dataset():
    """
    Build the TFT training dataset using
    all historical observations.

    The fitted normalization and encoders
    will be saved for production inference.
    """

    dataframe = load_eight_input_data()

    if dataframe.empty:
        raise ValueError(
            "The production dataset is empty."
        )

    if len(dataframe) != 1500:
        raise ValueError(
            "Expected 1,500 approved records. "
            f"Found {len(dataframe)}."
        )

    if dataframe["week_start_date"].nunique() != 300:
        raise ValueError(
            "Expected 300 consecutive reporting weeks."
        )

    # The full dataset is used for fitting
    # preprocessing and training.
    #
    # A four-week decoder is required during
    # supervised training, so the final four
    # observations are eligible as targets.
    training_dataset = TimeSeriesDataSet(
        dataframe,
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

    expected_windows_per_grade = (
        300 - ENCODER_LENGTH - PREDICTION_LENGTH + 1
    )

    expected_windows = (
        expected_windows_per_grade * len(TEA_GRADES)
    )

    if len(training_dataset) != expected_windows:
        raise ValueError(
            "Unexpected number of production training "
            f"windows: {len(training_dataset)} "
            f"(expected {expected_windows})."
        )

    print("\n=== FULL DATASET ===")
    print("Records:", len(dataframe))
    print(
        "Reporting weeks:",
        dataframe["week_start_date"].nunique(),
    )
    print(
        "First week:",
        dataframe["week_start_date"].min().date(),
    )
    print(
        "Last week:",
        dataframe["week_start_date"].max().date(),
    )
    print("Training windows:", len(training_dataset))
    print(
        "Windows per grade:",
        expected_windows_per_grade,
    )

    return dataframe, training_dataset


def create_model(training_dataset):
    """Construct the validation-selected TFT architecture."""

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        output_size=len(QUANTILES),
        loss=QuantileLoss(
            quantiles=QUANTILES,
        ),
        log_interval=20,
        reduce_on_plateau_patience=3,
        optimizer="adam",
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\n=== PRODUCTION ARCHITECTURE ===")
    print("Hidden size:", HIDDEN_SIZE)
    print(
        "Hidden continuous size:",
        HIDDEN_CONTINUOUS_SIZE,
    )
    print("Attention heads:", ATTENTION_HEADS)
    print("Dropout:", DROPOUT)
    print("Parameter count:", parameter_count)

    return model, parameter_count


def create_trainer():
    """
    Train for a fixed number of epochs.

    No validation/test scores are used
    to choose a production checkpoint.
    """

    logger = CSVLogger(
        save_dir=str(PRODUCTION_DIR),
        name="training_logs",
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=TRAINING_EPOCHS,
        min_epochs=TRAINING_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        log_every_n_steps=1,
    )

    return trainer, logger


def save_production_artifacts(
    trainer,
    training_dataset,
    dataframe,
    parameter_count,
    logger,
):
    """
    Save the trained checkpoint, fitted
    preprocessing parameters and metadata.
    """

    trainer.save_checkpoint(
        str(CHECKPOINT_FILE)
    )

    # These parameters contain the fitted
    # encoders, scalers and dataset settings
    # used during final model training.
    #
    # Load this file only from trusted
    # project artifacts.
    torch.save(
        training_dataset.get_parameters(),
        str(DATASET_PARAMETERS_FILE),
    )

    if not CHECKPOINT_FILE.is_file():
        raise RuntimeError(
            "Production checkpoint was not saved."
        )

    if not DATASET_PARAMETERS_FILE.is_file():
        raise RuntimeError(
            "Production preprocessing parameters "
            "were not saved."
        )

    metadata = {
        "model": "nine_input_tft",
        "model_version": "production_v1",
        "status": "trained_not_deployed",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection": {
            "configuration": "candidate_b",
            "selection_metric": "validation_mae_kg",
            "selected_validation_mae_kg": 384.77,
            "validation_interval_coverage_percent": 73.9796,
        },
        "dataset": {
            "path": str(DATA_FILE),
            "sha256": calculate_file_sha256(DATA_FILE),
            "records": int(len(dataframe)),
            "reporting_weeks": int(
                dataframe["week_start_date"].nunique()
            ),
            "first_week": str(
                dataframe["week_start_date"].min().date()
            ),
            "last_week": str(
                dataframe["week_start_date"].max().date()
            ),
            "tea_grades": list(TEA_GRADES),
            "all_available_data_used_for_training": True,
        },
        "architecture": {
            "type": "TemporalFusionTransformer",
            "hidden_size": HIDDEN_SIZE,
            "hidden_continuous_size": HIDDEN_CONTINUOUS_SIZE,
            "attention_heads": ATTENTION_HEADS,
            "dropout": DROPOUT,
            "parameter_count": parameter_count,
        },
        "training": {
            "epochs": TRAINING_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "random_seed": RANDOM_SEED,
            "training_windows": int(
                len(training_dataset)
            ),
            "selection_policy": "fixed_epochs",
            "held_out_test_used": False,
        },
        "forecasting": {
            "encoder_weeks": ENCODER_LENGTH,
            "forecast_weeks": PREDICTION_LENGTH,
            "quantiles": list(QUANTILES),
            "target": TARGET,
            "target_unit": "kg",
            "group_ids": list(GROUP_IDS),
            "historical_only_features": list(
                HISTORICAL_ONLY_REALS
            ),
            "known_future_features": list(
                KNOWN_FUTURE_REALS
            ),
        },
        "artifacts": {
            "checkpoint": str(CHECKPOINT_FILE),
            "dataset_parameters": str(
                DATASET_PARAMETERS_FILE
            ),
            "training_logs": str(logger.log_dir),
        },
        "deployment": {
            "backend_integrated": False,
            "frontend_integrated": False,
            "historical_forecasts_initialized": False,
        },
        "notes": [
            "Production preprocessing is fitted on the complete approved dataset.",
            "This checkpoint is trained on all available historical observations.",
            "The final checkpoint is not an independent test evaluation.",
            "Historical backfilled forecasts using this model must be labelled as backfilled.",
            "Weekly Data Entry remains restricted to the latest completed reporting week.",
            "The existing factory checkpoint remains unchanged until deployment.",
        ],
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return metadata


def main():
    prepare_environment()

    dataframe, training_dataset = (
        prepare_full_dataset()
    )

    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Production training dataloader is empty."
        )

    model, parameter_count = create_model(
        training_dataset
    )

    trainer, logger = create_trainer()

    print("\n=== STARTING PRODUCTION TRAINING ===")
    print("Training batches:", len(train_loader))
    print("Training epochs:", TRAINING_EPOCHS)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
    )

    metadata = save_production_artifacts(
        trainer,
        training_dataset,
        dataframe,
        parameter_count,
        logger,
    )

    print("\n=== PRODUCTION TRAINING RESULTS ===")
    print(
        "Checkpoint:",
        metadata["artifacts"]["checkpoint"],
    )
    print(
        "Preprocessing:",
        metadata["artifacts"]["dataset_parameters"],
    )
    print("Metadata:", METADATA_FILE)
    print(
        "Training windows:",
        metadata["training"]["training_windows"],
    )
    print(
        "Epochs:",
        metadata["training"]["epochs"],
    )

    print("\nPRODUCTION TFT TRAINING: PASS")
    print("Backend model replaced: NO")
    print("Frontend modified: NO")
    print("Historical forecasts created: NO")


if __name__ == "__main__":
    main()
