
"""
Train the six-input Temporal Fusion Transformer.

Inputs:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_rate
    festival_flag
    colombo_auction_price_lkr

Forecast:
    Four future weeks of demand
    Quantiles P10, P50 and P90

Validation selects the checkpoint.
The held-out test dataset is not evaluated here.

Existing research checkpoints are not modified.
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

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from build_tft_six_input_datasets import (
    load_six_input_data,
    create_six_input_datasets,
    check_dataset_integrity,
)

from tft_six_input_config import (
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
)


# Match the established TFT architecture.
HIDDEN_SIZE = 16
HIDDEN_CONTINUOUS_SIZE = 8
ATTENTION_HEADS = 4
DROPOUT = 0.1

EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_VAL = 0.1

REPORT_PATH = (
    OUTPUT_DIR / "six_input_training_metadata.json"
)


def prepare_environment():
    """Initialize deterministic training settings."""

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

    print("\n=== SIX-INPUT TFT TRAINING ===")
    print("Dataset:", DATA_FILE)
    print("Model:", MODEL_NAME)
    print("Device: CPU")
    print("Random seed:", RANDOM_SEED)


def create_dataloaders(training, validation):
    """Build complete training and validation loaders."""

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
    print(
        "Validation batches:",
        len(validation_loader),
    )

    return train_loader, validation_loader


def create_model(training):
    """Create a six-input TFT with quantile outputs."""

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
    print("Model parameters:", parameter_count)
    print("Hidden size:", HIDDEN_SIZE)
    print("Attention heads:", ATTENTION_HEADS)
    print(
        "Continuous hidden size:",
        HIDDEN_CONTINUOUS_SIZE,
    )
    print("Dropout:", DROPOUT)
    print("Quantiles:", QUANTILES)
    print(
        "Historical-only inputs:",
        HISTORICAL_ONLY_REALS,
    )
    print(
        "Future-known inputs:",
        KNOWN_FUTURE_REALS,
    )

    return model, parameter_count


def create_trainer():
    """Configure CPU training and checkpoint selection."""

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="six-input-epoch{epoch:02d}-val{val_loss:.4f}",
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
):
    """Write reproducible training metadata."""

    best_checkpoint = (
        checkpoint_callback.best_model_path
    )

    if not best_checkpoint:
        raise RuntimeError(
            "No best checkpoint was saved."
        )

    best_loss = float(
        checkpoint_callback.best_model_score
    )

    if not math.isfinite(best_loss):
        raise ValueError(
            "Validation loss is not finite."
        )

    report = {
        "model": MODEL_NAME,
        "status": "trained",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(DATA_FILE),
        "checkpoint": best_checkpoint,
        "best_validation_quantile_loss": best_loss,
        "training_epochs_completed": int(
            trainer.current_epoch + (
                0 if trainer.fit_loop.epoch_loop._batches_that_stepped == 0
                else 1
            )
        ),
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
            "test_start": str(TEST_START.date()),
            **dataset_report,
        },
        "known_future_features": KNOWN_FUTURE_REALS,
        "historical_only_features": (
            HISTORICAL_ONLY_REALS
        ),
        "test_evaluated": False,
        "test_metrics": None,
        "training_log_directory": str(
            logger.log_dir
        ),
        "notes": [
            "Training and validation use chronological splits.",
            "Normalization is fitted on training observations.",
            "The best checkpoint is selected by validation loss.",
            "The held-out test set is not evaluated during training.",
            "Future decoder features exclude future observed demand, production, inventory, prices, and exchange rates.",
            "Existing no-price and full TFT checkpoints are preserved.",
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
    prepare_environment()

    df = load_six_input_data()

    datasets = create_six_input_datasets(
        df
    )

    dataset_report = check_dataset_integrity(
        df,
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

    print("\nSIX-INPUT TFT TRAINING: PASS")


if __name__ == "__main__":
    main()
