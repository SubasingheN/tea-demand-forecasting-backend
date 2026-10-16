
from pathlib import Path
from datetime import datetime, timezone
import json
import random

import numpy as np
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from build_tft_datasets import (
    GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    KNOWN_FUTURE_REALS,
    HISTORICAL_ONLY_REALS,
    load_features,
    load_manifests,
    create_datasets,
    validate_datasets,
    validate_feature_roles,
    inspect_sample,
)


# ============================================================
# PHASE 2 - STEP 11
# FULL TEMPORAL FUSION TRANSFORMER TRAINING
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_full_training"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_FILE = OUTPUT_DIR / "training_results.json"

SEED = 42

# Model architecture
HIDDEN_SIZE = 16
ATTENTION_HEADS = 2
HIDDEN_CONTINUOUS_SIZE = 8
DROPOUT = 0.1

# Training configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_VAL = 0.1

QUANTILES = [0.1, 0.5, 0.9]

# CPU-compatible settings
NUM_WORKERS = 0

# False means use every available batch.
FAST_DEV_RUN = False


# ============================================================
# REPRODUCIBILITY
# ============================================================

def configure_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    pl.seed_everything(
        SEED,
        workers=True,
    )

    print("Random seed:", SEED)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def prepare_directories():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD AND VALIDATE TFT DATASETS
# ============================================================

def prepare_datasets():
    print("\nLOADING FEATURE DATA")
    print("-" * 65)

    model_df = load_features()

    print("\nLOADING STEP 8 WINDOW MANIFESTS")

    manifests = load_manifests()

    print("\nCONSTRUCTING TFT DATASETS")

    training, validation, testing = create_datasets(
        model_df
    )

    datasets = {
        "train": training,
        "validation": validation,
        "test": testing,
    }

    print("\nVALIDATING CHRONOLOGICAL WINDOWS")

    results = validate_datasets(
        datasets,
        manifests,
        model_df,
    )

    validate_feature_roles(training)
    inspect_sample(training)

    print("\nDATASET VALIDATION: PASS")

    return (
        training,
        validation,
        testing,
        results,
    )


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def prepare_dataloaders(training, validation):
    print("\nCREATING DATA LOADERS")
    print("-" * 65)

    train_loader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("Training windows:", len(training))
    print("Validation windows:", len(validation))

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    print(
        "Batch limits: NONE - full available data"
    )

    return train_loader, val_loader


# ============================================================
# CONFIGURE TFT MODEL
# ============================================================

def build_model(training):
    print("\nCONFIGURING TFT MODEL")
    print("-" * 65)

    quantile_loss = QuantileLoss(
        quantiles=QUANTILES,
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        dropout=DROPOUT,
        output_size=len(QUANTILES),
        loss=quantile_loss,
        log_interval=20,
        reduce_on_plateau_patience=3,
        optimizer="adam",
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Model: TemporalFusionTransformer")
    print("Total parameters:", total_parameters)
    print(
        "Trainable parameters:",
        trainable_parameters,
    )
    print("Hidden size:", HIDDEN_SIZE)
    print("Attention heads:", ATTENTION_HEADS)
    print("Dropout:", DROPOUT)
    print("Learning rate:", LEARNING_RATE)
    print("Forecast horizon:", PREDICTION_LENGTH)
    print("Encoder length:", ENCODER_LENGTH)
    print("Quantiles:", QUANTILES)

    return model


# ============================================================
# CONFIGURE LIGHTNING TRAINER
# ============================================================

def build_trainer():
    print("\nCONFIGURING FULL TRAINING")
    print("-" * 65)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="tft-epoch{epoch:02d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=0.0,
        mode="min",
        verbose=True,
    )

    csv_logger = CSVLogger(
        save_dir=str(LOG_DIR),
        name="tft_training",
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
        logger=csv_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
        num_sanity_val_steps=2,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        log_every_n_steps=10,
        precision="32-true",
        fast_dev_run=FAST_DEV_RUN,
    )

    print("Training hardware: CPU")
    print("Maximum epochs:", MAX_EPOCHS)
    print(
        "Early stopping patience:",
        EARLY_STOPPING_PATIENCE,
    )
    print("Batch size:", BATCH_SIZE)
    print("Full training batches: ENABLED")
    print("Full validation batches: ENABLED")

    return (
        trainer,
        checkpoint_callback,
        csv_logger,
    )


# ============================================================
# METRIC CONVERSION
# ============================================================

def safe_float(value):
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None

        return float(
            value.detach().cpu().item()
        )

    if isinstance(value, (int, float)):
        return float(value)

    return None


# ============================================================
# SAVE TRAINING REPORT
# ============================================================

def save_training_report(
    trainer,
    checkpoint_callback,
    csv_logger,
    model,
    training,
    validation,
    testing,
    dataset_results,
    training_started_at,
):
    best_checkpoint = (
        checkpoint_callback.best_model_path
    )

    if not best_checkpoint:
        raise RuntimeError(
            "No best checkpoint was generated."
        )

    best_path = Path(best_checkpoint)

    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint is missing: {best_path}"
        )

    best_loss = safe_float(
        checkpoint_callback.best_model_score
    )

    if best_loss is None:
        raise RuntimeError(
            "Best validation loss is unavailable."
        )

    if not np.isfinite(best_loss):
        raise RuntimeError(
            "Best validation loss is not finite."
        )

    callback_metrics = {}

    for name, value in trainer.callback_metrics.items():
        converted = safe_float(value)

        if converted is not None and np.isfinite(
            converted
        ):
            callback_metrics[str(name)] = converted

    report = {
        "experiment": "tft_full_training",
        "status": "PASS",
        "model": "TemporalFusionTransformer",
        "started_at_utc": training_started_at,
        "finished_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "seed": SEED,
        "hardware": "CPU",
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders including "
            "unfulfilled orders"
        ),
        "grades": GRADES,
        "encoder_length": ENCODER_LENGTH,
        "forecast_horizon": PREDICTION_LENGTH,
        "quantiles": QUANTILES,
        "model_settings": {
            "hidden_size": HIDDEN_SIZE,
            "attention_heads": ATTENTION_HEADS,
            "hidden_continuous_size": (
                HIDDEN_CONTINUOUS_SIZE
            ),
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": (
                EARLY_STOPPING_PATIENCE
            ),
            "gradient_clip_val": (
                GRADIENT_CLIP_VAL
            ),
        },
        "epochs_completed": int(
            trainer.current_epoch
        ),
        "best_validation_quantile_loss": best_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": (
            checkpoint_callback.last_model_path
        ),
        "training_log_directory": str(
            csv_logger.log_dir
        ),
        "dataset_windows": {
            "train": len(training),
            "validation": len(validation),
            "test_reserved": len(testing),
        },
        "dataset_alignment": dataset_results,
        "known_future_features": (
            KNOWN_FUTURE_REALS
        ),
        "historical_only_features": (
            HISTORICAL_ONLY_REALS
        ),
        "model_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "callback_metrics": callback_metrics,
        "research_notes": [
            (
                "Only training data was used "
                "to fit TFT parameters."
            ),
            (
                "2024 validation data was used "
                "for checkpoint selection."
            ),
            (
                "The 2025-2026 test partition "
                "was not evaluated during training."
            ),
            (
                "High and Medium auction prices "
                "are historical quotation-derived "
                "proxies and remain source candidates."
            ),
            (
                "No future actual production, "
                "inventory, demand or auction prices "
                "are configured as known-future inputs."
            ),
            (
                "Quantile loss is not directly "
                "equivalent to MAE or RMSE."
            ),
        ],
    }

    RESULT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "PHASE 2 - STEP 11: FULL TFT MODEL TRAINING"
    )
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("Lightning:", pl.__version__)

    training_started_at = datetime.now(
        timezone.utc
    ).isoformat()

    configure_seed()
    prepare_directories()

    (
        training,
        validation,
        testing,
        dataset_results,
    ) = prepare_datasets()

    train_loader, val_loader = prepare_dataloaders(
        training,
        validation,
    )

    model = build_model(training)

    (
        trainer,
        checkpoint_callback,
        csv_logger,
    ) = build_trainer()

    print("\nSTARTING FULL TFT TRAINING")
    print("=" * 65)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\nSAVING TRAINING RESULTS")
    print("-" * 65)

    results = save_training_report(
        trainer=trainer,
        checkpoint_callback=checkpoint_callback,
        csv_logger=csv_logger,
        model=model,
        training=training,
        validation=validation,
        testing=testing,
        dataset_results=dataset_results,
        training_started_at=training_started_at,
    )

    print("\nFULL TFT TRAINING SUMMARY")
    print("=" * 65)

    print(
        "Epochs completed:",
        results["epochs_completed"],
    )

    print(
        "Best validation quantile loss:",
        results["best_validation_quantile_loss"],
    )

    print(
        "Best model checkpoint:",
        results["best_checkpoint"],
    )

    print(
        "Training windows:",
        results["dataset_windows"]["train"],
    )

    print(
        "Validation windows:",
        results["dataset_windows"]["validation"],
    )

    print(
        "Reserved test windows:",
        results["dataset_windows"]["test_reserved"],
    )

    print("\nOUTPUTS")
    print("Checkpoints:", CHECKPOINT_DIR)
    print(
        "CSV training logs:",
        results["training_log_directory"],
    )
    print("Training report:", RESULT_FILE)

    print("\nFULL TFT TRAINING: PASS")


if __name__ == "__main__":
    main()
