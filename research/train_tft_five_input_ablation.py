
"""
Train the five-input Temporal Fusion Transformer
for the controlled auction-price feature-ablation study.

Uses the same Version 2 source observations,
chronological splits, TFT architecture, and
training settings as the eight-input experiment.

Five historical features:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_rate
    festival_flag

No auction-price features enter this model.

The held-out test set is NOT evaluated here.

Existing six-input and eight-input TFT models,
checkpoints, and evaluation outputs remain unchanged.

Research limitation:
The underlying Version 2 dataset is provisional.
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

from build_tft_five_input_ablation_datasets import (
    load_five_input_ablation_data,
    create_five_input_ablation_datasets,
    check_five_input_ablation_integrity,
)

from tft_five_input_ablation_config import (
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
    EXCLUDED_AUCTION_FEATURES,
    validate_feature_configuration,
)


# Identical architecture to the eight-input TFT.
HIDDEN_SIZE = 16
HIDDEN_CONTINUOUS_SIZE = 8
ATTENTION_HEADS = 4
DROPOUT = 0.1

EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_VAL = 0.1

REPORT_PATH = (
    OUTPUT_DIR / "five_input_ablation_training_metadata.json"
)


def prepare_environment():
    """Verify isolation and prepare experiment directories."""

    validate_feature_configuration()

    if REPORT_PATH.exists():
        raise FileExistsError(
            "Training metadata already exists. "
            "Refusing to overwrite an earlier experiment: "
            f"{REPORT_PATH}"
        )

    if CHECKPOINT_DIR.exists():
        previous_checkpoints = list(
            CHECKPOINT_DIR.glob("*.ckpt")
        )

        if previous_checkpoints:
            raise FileExistsError(
                "Ablation checkpoints already exist. "
                "Refusing to overwrite them: "
                f"{previous_checkpoints}"
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

    print("\n=== FIVE-INPUT TFT ABLATION TRAINING ===")
    print("Dataset:", DATA_FILE)
    print("Model:", MODEL_NAME)
    print("Device: CPU")
    print("Random seed:", RANDOM_SEED)
    print("Excluded prices:", EXCLUDED_AUCTION_FEATURES)


def create_dataloaders(training, validation):
    """Create chronological training/validation loaders."""

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
    """Initialize the five-input quantile TFT."""

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
    print("Known future features:", KNOWN_FUTURE_REALS)

    return model, parameter_count


def create_trainer():
    """Configure matching CPU training and early stopping."""

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename=(
            "five-input-ablation-epoch{epoch:02d}"
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
    """Save model metadata without evaluating test windows."""

    best_checkpoint = checkpoint_callback.best_model_path

    if not best_checkpoint:
        raise RuntimeError(
            "No model checkpoint was saved."
        )

    best_score = checkpoint_callback.best_model_score

    if best_score is None:
        raise RuntimeError(
            "Best validation score is unavailable."
        )

    best_loss = float(best_score)

    if not math.isfinite(best_loss):
        raise ValueError(
            "Validation loss must be finite."
        )

    audit_status_counts = {}

    if "auction_price_record_status" in dataframe.columns:
        audit_status_counts = {
            str(key): int(value)
            for key, value in dataframe[
                "auction_price_record_status"
            ].value_counts(dropna=False).items()
        }

    completed_epochs = int(trainer.current_epoch)

    if not trainer.should_stop:
        completed_epochs = min(
            int(trainer.max_epochs),
            completed_epochs + 1,
        )

    report = {
        "model": MODEL_NAME,
        "status": "trained",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "auction_price_feature_ablation",
        "dataset": str(DATA_FILE),
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
        "known_future_features": KNOWN_FUTURE_REALS,
        "historical_only_features": HISTORICAL_ONLY_REALS,
        "excluded_auction_features": (
            EXCLUDED_AUCTION_FEATURES
        ),
        "auction_features_used_by_model": False,
        "auction_source_audit_status": (
            audit_status_counts
        ),
        "auction_publication_timestamp_verified": False,
        "dataset_approval_status": "provisional",
        "test_evaluated": False,
        "test_metrics": None,
        "training_log_directory": str(
            logger.log_dir
        ),
        "notes": [
            "Uses the same prepared Version 2 observations.",
            "Only five historical model inputs are supplied.",
            "Auction prices are excluded from model inputs.",
            "Chronological training and validation splits.",
            "Normalization is fitted on training data.",
            "Best checkpoint is selected using validation loss.",
            "Held-out test data is not evaluated during training.",
            "The Version 2 dataset retains source audit limitations.",
            "Six-input and eight-input checkpoints are unchanged.",
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

    dataframe = load_five_input_ablation_data()

    datasets = create_five_input_ablation_datasets(
        dataframe
    )

    dataset_report = check_five_input_ablation_integrity(
        dataframe,
        datasets,
    )

    train_loader, validation_loader = create_dataloaders(
        datasets["training"],
        datasets["validation"],
    )

    model, parameter_count = create_model(
        datasets["training"]
    )

    trainer, checkpoint_callback, logger = (
        create_trainer()
    )

    print("\n=== STARTING ABLATION TRAINING ===")
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
    print("Auction-price features used: NO")
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

    print("\n=== ABLATION TRAINING RESULTS ===")
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

    print("\nFIVE-INPUT TFT ABLATION TRAINING: PASS")


if __name__ == "__main__":
    main()
