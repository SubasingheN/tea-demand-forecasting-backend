
"""
Train the revised nine-input Temporal Fusion Transformer.

Historical inputs:
    1. demand_kg
    2. production_kg
    3. inventory_kg
    4. usd_lkr_monday_rate
    5. usd_lkr_sunday_rate
    6. festival_flag
    7. auction_price_high_lkr
    8. auction_price_medium_lkr
    9. auction_price_low_lkr

Forecast:
    Four future weeks of demand, in kilograms.

Forecast quantiles:
    P10 - lower estimate
    P50 - median estimate
    P90 - upper estimate

The model uses 52 completed historical weeks.

Training and validation are chronological.
The held-out test set is not evaluated during training.

The previous model checkpoints are not modified.
"""

import json
import math

from datetime import datetime, timezone

import lightning.pytorch as pl

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import (
    TemporalFusionTransformer,
)

from pytorch_forecasting.metrics import QuantileLoss

from build_eight_input_dataset import (
    load_eight_input_data,
    create_eight_input_datasets,
    check_dataset_integrity,
)

from tft_eight_input_config import (
    DATA_FILE,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    QUANTILES,
    RANDOM_SEED,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    VALIDATION_START,
    TEST_START,
    HISTORICAL_ONLY_REALS,
    KNOWN_FUTURE_REALS,
    EXCHANGE_RATE_FEATURES,
    AUCTION_PRICE_FEATURES,
    validate_feature_configuration,
)


# Preserve the existing TFT architecture.
HIDDEN_SIZE = 16
HIDDEN_CONTINUOUS_SIZE = 8
ATTENTION_HEADS = 4
DROPOUT = 0.1

EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_VAL = 0.1

REPORT_PATH = (
    OUTPUT_DIR / "nine_input_training_metadata.json"
)


def prepare_environment():
    """
    Validate configuration and prepare
    separate nine-input output directories.
    """

    validate_feature_configuration()

    if MODEL_NAME != "nine_input_tft":
        raise ValueError(
            "Expected the nine-input TFT configuration."
        )

    if len(HISTORICAL_ONLY_REALS) != 9:
        raise ValueError(
            "Exactly nine historical features "
            "are required."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.seed_everything(
        RANDOM_SEED,
        workers=True,
    )

    print("\n=== NINE-INPUT TFT TRAINING ===")
    print("Dataset:", DATA_FILE)
    print("Model:", MODEL_NAME)
    print("Device: CPU")
    print("Random seed:", RANDOM_SEED)
    print("Historical weeks:", ENCODER_LENGTH)
    print("Forecast weeks:", PREDICTION_LENGTH)
    print("Historical features:", len(HISTORICAL_ONLY_REALS))


def create_dataloaders(training, validation):
    """
    Create training and validation loaders
    with consistent batch settings.
    """

    train_loader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Training dataloader is empty."
        )

    if len(validation_loader) == 0:
        raise ValueError(
            "Validation dataloader is empty."
        )

    print("\n=== DATALOADERS ===")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    return train_loader, validation_loader


def create_model(training):
    """
    Initialize the revised nine-input
    quantile Temporal Fusion Transformer.
    """

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
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

    print("\n=== MODEL ARCHITECTURE ===")
    print("Architecture: TemporalFusionTransformer")
    print("Parameters:", parameter_count)
    print("Hidden size:", HIDDEN_SIZE)
    print("Attention heads:", ATTENTION_HEADS)
    print("Dropout:", DROPOUT)
    print("Quantiles:", QUANTILES)
    print("Historical features:", HISTORICAL_ONLY_REALS)
    print("Known-future features:", KNOWN_FUTURE_REALS)

    return model, parameter_count


def create_trainer():
    """
    Configure CPU training, early stopping,
    training logs, and checkpoint selection.
    """

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename=(
            "nine-input-epoch{epoch:02d}"
            "-val{val_loss:.4f}"
        ),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=EARLY_STOPPING_PATIENCE,
        mode="min",
        verbose=True,
    )

    logger = CSVLogger(
        save_dir=str(OUTPUT_DIR),
        name="training_logs",
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        min_epochs=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        deterministic=True,
    )

    return trainer, checkpoint_callback, logger


def save_training_report(
    trainer,
    checkpoint_callback,
    logger,
    parameter_count,
    dataset_report,
    dataframe,
):
    """
    Save training metadata without
    evaluating the held-out test set.
    """

    best_checkpoint = (
        checkpoint_callback.best_model_path
    )

    if not best_checkpoint:
        raise RuntimeError(
            "No trained model checkpoint was saved."
        )

    best_score = (
        checkpoint_callback.best_model_score
    )

    if best_score is None:
        raise RuntimeError(
            "Best validation score is unavailable."
        )

    best_loss = float(best_score)

    if not math.isfinite(best_loss):
        raise ValueError(
            "Validation loss is not finite."
        )

    completed_epochs = (
        int(trainer.current_epoch) + 1
    )

    completed_epochs = min(
        completed_epochs,
        int(trainer.max_epochs),
    )

    report = {
        "model": MODEL_NAME,
        "status": "trained",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(DATA_FILE),
        "dataset_records": len(dataframe),
        "dataset_first_week": str(
            dataframe["week_start_date"].min().date()
        ),
        "dataset_last_week": str(
            dataframe["week_start_date"].max().date()
        ),
        "checkpoint": best_checkpoint,
        "best_validation_quantile_loss": best_loss,
        "training_epochs_reported": completed_epochs,
        "model_parameters": parameter_count,
        "architecture": {
            "model": "TemporalFusionTransformer",
            "hidden_size": HIDDEN_SIZE,
            "attention_heads": ATTENTION_HEADS,
            "hidden_continuous_size": (
                HIDDEN_CONTINUOUS_SIZE
            ),
            "dropout": DROPOUT,
            "optimizer": "adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "encoder_weeks": ENCODER_LENGTH,
            "forecast_weeks": PREDICTION_LENGTH,
            "quantiles": QUANTILES,
            "seed": RANDOM_SEED,
        },
        "data_split": {
            "validation_start": str(
                VALIDATION_START.date()
            ),
            "test_start": str(
                TEST_START.date()
            ),
            **dataset_report,
        },
        "known_future_features": (
            KNOWN_FUTURE_REALS
        ),
        "historical_only_features": (
            HISTORICAL_ONLY_REALS
        ),
        "exchange_rate_features": (
            EXCHANGE_RATE_FEATURES
        ),
        "auction_price_features": (
            AUCTION_PRICE_FEATURES
        ),
        "exchange_rate_range_format": (
            "monday_rate-sunday_rate"
        ),
        "exchange_rate_range_used_directly": False,
        "exchange_rate_numeric_endpoints_used": True,
        "test_evaluated": False,
        "test_metrics": None,
        "training_log_directory": str(
            logger.log_dir
        ),
        "deployment_status": (
            "not_deployed"
        ),
        "notes": [
            "Nine historical numeric inputs are used.",
            "Monday and Sunday USD/LKR rates replace the original numeric exchange rate.",
            "Three elevation-specific auction prices remain historical-only.",
            "Training and validation follow chronological splits.",
            "Normalization is fitted on training data.",
            "The best checkpoint is selected using validation loss.",
            "The held-out test set is not evaluated in this training step.",
            "Previous model checkpoints remain separate.",
            "Backend inference must use the same nine-input preprocessing.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return report


def main():
    """
    Train the nine-input TFT using the
    revised historical dataset.
    """

    prepare_environment()

    dataframe = load_eight_input_data()

    datasets = create_eight_input_datasets(
        dataframe
    )

    dataset_report = check_dataset_integrity(
        dataframe,
        datasets,
    )

    train_loader, validation_loader = (
        create_dataloaders(
            datasets["training"],
            datasets["validation"],
        )
    )

    model, parameter_count = create_model(
        datasets["training"]
    )

    trainer, checkpoint_callback, logger = (
        create_trainer()
    )

    print("\n=== STARTING MODEL TRAINING ===")
    print("Maximum epochs:", MAX_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print(
        "Training windows:",
        len(datasets["training"]),
    )
    print(
        "Validation windows:",
        len(datasets["validation"]),
    )
    print(
        "Held-out test windows:",
        len(datasets["testing"]),
    )
    print("Test evaluation: DISABLED")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )

    report = save_training_report(
        trainer,
        checkpoint_callback,
        logger,
        parameter_count,
        dataset_report,
        dataframe,
    )

    print("\n=== TRAINING RESULTS ===")
    print(
        "Best validation quantile loss:",
        report["best_validation_quantile_loss"],
    )
    print(
        "Best checkpoint:",
        report["checkpoint"],
    )
    print("Report:", REPORT_PATH)
    print("Test set evaluated: NO")
    print("Backend model replaced: NO")

    print("\nNINE-INPUT TFT TRAINING: PASS")


if __name__ == "__main__":
    main()
