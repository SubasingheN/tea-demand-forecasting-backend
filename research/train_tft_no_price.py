
from pathlib import Path
from datetime import datetime, timezone
import json
import random

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from build_tft_datasets import (
    load_features,
    load_manifests,
    validate_datasets,
)


# ============================================================
# STEP 13 - AUCTION PRICE ABLATION
# TRAIN THE SAME TFT WITHOUT AUCTION PRICE FEATURES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_no_price"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_FILE = OUTPUT_DIR / "training_results.json"

SEED = 42

GRADES = ["BOP", "BOPF", "Dust", "OP", "Pekoe"]

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")

ENCODER_LENGTH = 52
PREDICTION_LENGTH = 4

BATCH_SIZE = 32
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 0.001
GRADIENT_CLIP_VAL = 0.1

HIDDEN_SIZE = 16
ATTENTION_HEADS = 2
HIDDEN_CONTINUOUS_SIZE = 8
DROPOUT = 0.1

QUANTILES = [0.1, 0.5, 0.9]

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
]

PRICE_COLUMNS = [
    "auction_price_high_lkr_per_kg_input",
    "auction_price_medium_lkr_per_kg_input",
    "auction_price_high_lkr_per_kg_available",
    "auction_price_medium_lkr_per_kg_available",
]

MODEL_COLUMNS = (
    ["week_start_date", "grade"]
    + KNOWN_FUTURE_REALS
    + HISTORICAL_ONLY_REALS
)


def configure_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    pl.seed_everything(
        SEED,
        workers=True,
    )


def create_directories():
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


def prepare_no_price_data():
    print("\nLOADING PROCESSED FEATURES")
    print("-" * 65)

    full_df = load_features()

    # Use exactly the same rows, dates, target,
    # and calendar/operational inputs as Model A.
    # Only auction price inputs and masks are removed.
    missing_columns = [
        column
        for column in MODEL_COLUMNS
        if column not in full_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing source columns: "
            + ", ".join(missing_columns)
        )

    df = full_df[MODEL_COLUMNS].copy()

    if any(
        column in df.columns
        for column in PRICE_COLUMNS
    ):
        raise ValueError(
            "Auction-price columns are still present."
        )

    if len(df) != 5680:
        raise ValueError(
            f"Expected 5680 records, found {len(df)}"
        )

    if set(df["grade"]) != set(GRADES):
        raise ValueError(
            "Unexpected tea grades."
        )

    if df.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate grade/week rows."
        )

    for column in (
        KNOWN_FUTURE_REALS
        + HISTORICAL_ONLY_REALS
    ):
        if df[column].isna().any():
            raise ValueError(
                f"Missing values in {column}"
            )

    df["demand_kg"] = (
        df["demand_kg"].astype("float32")
    )

    if df["demand_kg"].le(0).any():
        raise ValueError(
            "Demand must be positive for softplus."
        )

    df["time_idx"] = (
        df["time_idx"].astype("int64")
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Auction-price features: REMOVED")
    print("Target: demand_kg")
    print("Target type: float32")

    return df


def get_cutoff_index(df, cutoff):
    subset = (
        df[df["week_start_date"] >= cutoff]
        .sort_values("week_start_date")
    )

    if subset.empty:
        raise ValueError(
            f"No observations for cutoff {cutoff}"
        )

    return int(
        subset.iloc[0]["time_idx"]
    )


def create_no_price_datasets(df):
    print("\nCONSTRUCTING NO-PRICE TFT DATASETS")
    print("-" * 65)

    validation_start_idx = get_cutoff_index(
        df,
        VALIDATION_START,
    )

    test_start_idx = get_cutoff_index(
        df,
        TEST_START,
    )

    train_df = df[
        df["week_start_date"]
        < VALIDATION_START
    ].copy()

    validation_df = df[
        df["week_start_date"]
        < TEST_START
    ].copy()

    test_df = df.copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="demand_kg",
        group_ids=["grade"],

        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,

        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,

        static_categoricals=["grade"],

        time_varying_known_reals=KNOWN_FUTURE_REALS,

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

    validation = TimeSeriesDataSet.from_dataset(
        training,
        validation_df,
        min_prediction_idx=validation_start_idx,
        stop_randomization=True,
        predict=False,
    )

    testing = TimeSeriesDataSet.from_dataset(
        training,
        test_df,
        min_prediction_idx=test_start_idx,
        stop_randomization=True,
        predict=False,
    )

    return training, validation, testing


def check_datasets(df, datasets):
    print("\nCHECKING CHRONOLOGICAL SPLITS")
    print("-" * 65)

    manifests = load_manifests()

    results = validate_datasets(
        datasets,
        manifests,
        df,
    )

    train = datasets["train"]

    known = set(
        train.time_varying_known_reals
    )

    unknown = set(
        train.time_varying_unknown_reals
    )

    if known & unknown:
        raise ValueError(
            "Known and unknown model inputs overlap."
        )

    if "demand_kg" in known:
        raise ValueError(
            "Future target leakage detected."
        )

    if any(
        column in known or column in unknown
        for column in PRICE_COLUMNS
    ):
        raise ValueError(
            "Auction-price ablation failed."
        )

    if not set(
        HISTORICAL_ONLY_REALS
    ).issubset(unknown):
        raise ValueError(
            "Missing historical observed inputs."
        )

    print("\nKNOWN FUTURE INPUTS")
    print(sorted(known))

    print("\nHISTORICAL-ONLY INPUTS")
    print(sorted(unknown))

    print("\nAUCTION PRICE ABLATION: PASS")

    return results


def create_dataloaders(training, validation):
    train_loader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    print("\nDATALOADERS")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("All batches enabled: YES")

    return train_loader, val_loader


def create_model(training):
    print("\nCREATING NO-PRICE TFT MODEL")
    print("-" * 65)

    model = TemporalFusionTransformer.from_dataset(
        training,

        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,

        attention_head_size=ATTENTION_HEADS,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,

        dropout=DROPOUT,

        output_size=len(QUANTILES),
        loss=QuantileLoss(
            quantiles=QUANTILES
        ),

        log_interval=20,
        reduce_on_plateau_patience=3,
        optimizer="adam",
    )

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model: TemporalFusionTransformer")
    print("Model parameters:", parameters)
    print("Auction-price inputs: NONE")
    print("Quantiles:", QUANTILES)

    return model


def create_trainer():
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename=(
            "tft-no-price-epoch{epoch:02d}"
            "-val{val_loss:.4f}"
        ),
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

    logger = CSVLogger(
        save_dir=str(LOG_DIR),
        name="tft_no_price_training",
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

        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,

        deterministic=True,
        num_sanity_val_steps=2,

        limit_train_batches=1.0,
        limit_val_batches=1.0,

        log_every_n_steps=10,
        precision="32-true",
    )

    return trainer, checkpoint_callback, logger


def to_float(value):
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None

        return float(
            value.detach().cpu().item()
        )

    return float(value)


def save_report(
    trainer,
    checkpoint_callback,
    logger,
    model,
    datasets,
    split_results,
    started_at,
):
    best_checkpoint = (
        checkpoint_callback.best_model_path
    )

    if not best_checkpoint:
        raise RuntimeError(
            "Best checkpoint was not saved."
        )

    if not Path(best_checkpoint).exists():
        raise FileNotFoundError(
            best_checkpoint
        )

    best_loss = to_float(
        checkpoint_callback.best_model_score
    )

    if (
        best_loss is None
        or not np.isfinite(best_loss)
    ):
        raise ValueError(
            "Invalid best validation loss."
        )

    metrics = {}

    for name, value in trainer.callback_metrics.items():
        try:
            converted = to_float(value)
        except (TypeError, ValueError):
            continue

        if (
            converted is not None
            and np.isfinite(converted)
        ):
            metrics[str(name)] = converted

    report = {
        "experiment": "tft_no_price_ablation",
        "status": "PASS",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": "TemporalFusionTransformer",
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders including "
            "unfulfilled orders"
        ),

        "auction_price_features": [],
        "auction_price_ablation": True,

        "grades": GRADES,
        "encoder_length": ENCODER_LENGTH,
        "prediction_length": PREDICTION_LENGTH,

        "seed": SEED,
        "quantiles": QUANTILES,

        "training_settings": {
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
        },

        "epochs_completed": int(
            trainer.current_epoch
        ),

        "best_validation_quantile_loss": best_loss,

        "best_checkpoint": best_checkpoint,

        "training_log_directory": str(
            logger.log_dir
        ),

        "dataset_windows": {
            name: len(dataset)
            for name, dataset in datasets.items()
        },

        "split_validation": split_results,

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

        "callback_metrics": metrics,

        "test_partition_evaluated": False,

        "research_notes": [
            (
                "Auction price values and "
                "availability masks were removed."
            ),
            (
                "The same chronological split, "
                "forecast horizon, model hyperparameters "
                "and random seed were used."
            ),
            (
                "Validation data was used for "
                "model checkpoint selection."
            ),
            (
                "The held-out test partition "
                "was not evaluated."
            ),
            (
                "Forecast MAE and interval coverage "
                "must be compared separately using "
                "matched validation windows."
            ),
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


def main():
    print(
        "PHASE 2 - STEP 13: "
        "TFT AUCTION PRICE ABLATION TRAINING"
    )
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("Lightning:", pl.__version__)

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    configure_seed()
    create_directories()

    df = prepare_no_price_data()

    training, validation, testing = (
        create_no_price_datasets(df)
    )

    datasets = {
        "train": training,
        "validation": validation,
        "test": testing,
    }

    split_results = check_datasets(
        df,
        datasets,
    )

    train_loader, val_loader = (
        create_dataloaders(
            training,
            validation,
        )
    )

    model = create_model(training)

    trainer, checkpoint_callback, logger = (
        create_trainer()
    )

    print("\nSTARTING NO-PRICE TFT TRAINING")
    print("=" * 65)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    report = save_report(
        trainer,
        checkpoint_callback,
        logger,
        model,
        datasets,
        split_results,
        started_at,
    )

    print("\nNO-PRICE TFT TRAINING SUMMARY")
    print("=" * 65)

    print(
        "Epochs completed:",
        report["epochs_completed"]
    )

    print(
        "Best validation quantile loss:",
        report["best_validation_quantile_loss"]
    )

    print(
        "Best checkpoint:",
        report["best_checkpoint"]
    )

    print(
        "Training windows:",
        report["dataset_windows"]["train"]
    )

    print(
        "Validation windows:",
        report["dataset_windows"]["validation"]
    )

    print(
        "Reserved test windows:",
        report["dataset_windows"]["test"]
    )

    print("\nOUTPUTS")
    print("Checkpoints:", CHECKPOINT_DIR)
    print("Training logs:", logger.log_dir)
    print("Training report:", REPORT_FILE)

    print("\nNO-PRICE TFT TRAINING: PASS")


if __name__ == "__main__":
    main()
