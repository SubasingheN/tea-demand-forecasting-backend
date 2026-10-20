
"""
Optimize the nine-input Tea Demand Forecasting TFT.

Train two candidate TFT configurations and select
the best using chronological validation data.

Important:
- Nine historical features
- 52-week encoder
- Four-week forecast horizon
- Five tea grades
- No held-out test evaluation
- No deployment or modification of existing models

Outputs are stored in:
research/outputs/tft_nine_input/optimization/
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from build_eight_input_dataset import (
    load_eight_input_data,
    create_eight_input_datasets,
    check_dataset_integrity,
)

from tft_eight_input_config import (
    OUTPUT_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    QUANTILES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    HISTORICAL_ONLY_REALS,
    TEA_GRADES,
    validate_feature_configuration,
)


OPTIMIZATION_DIR = OUTPUT_DIR / "optimization"
RESULTS_FILE = OPTIMIZATION_DIR / "candidate_results.csv"
SUMMARY_FILE = OPTIMIZATION_DIR / "optimization_summary.json"

CANDIDATES = [
    {
        "name": "candidate_a",
        "hidden_size": 24,
        "hidden_continuous_size": 12,
        "attention_head_size": 4,
        "dropout": 0.15,
    },
    {
        "name": "candidate_b",
        "hidden_size": 32,
        "hidden_continuous_size": 16,
        "attention_head_size": 4,
        "dropout": 0.20,
    },
]

PATIENCE = 5
GRADIENT_CLIP_VAL = 0.1


def prepare():
    """Validate configuration and protect existing outputs."""

    validate_feature_configuration()

    if ENCODER_LENGTH != 52:
        raise ValueError("Expected 52 encoder weeks.")

    if PREDICTION_LENGTH != 4:
        raise ValueError("Expected four forecast weeks.")

    if len(HISTORICAL_ONLY_REALS) != 9:
        raise ValueError("Expected nine historical features.")

    if len(TEA_GRADES) != 5:
        raise ValueError("Expected five tea grades.")

    if list(QUANTILES) != [0.1, 0.5, 0.9]:
        raise ValueError("Expected P10, P50 and P90.")

    for path in [RESULTS_FILE, SUMMARY_FILE]:
        if path.exists():
            raise FileExistsError(
                f"Optimization output already exists: {path}"
            )

    for candidate in CANDIDATES:
        directory = (
            OPTIMIZATION_DIR
            / candidate["name"]
        )

        if directory.exists():
            raise FileExistsError(
                f"Candidate directory already exists: {directory}"
            )

    OPTIMIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n=== NINE-INPUT TFT OPTIMIZATION ===")
    print("Candidates:", len(CANDIDATES))
    print("Encoder:", ENCODER_LENGTH)
    print("Forecast horizon:", PREDICTION_LENGTH)
    print("Test data used for selection: NO")


def create_loaders(datasets):
    """Create chronological training and validation loaders."""

    train_loader = datasets["training"].to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = datasets["validation"].to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Training or validation loader is empty.")

    return train_loader, val_loader


def create_model(training_dataset, config):
    """Build a candidate TFT model."""

    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=config["hidden_size"],
        hidden_continuous_size=(
            config["hidden_continuous_size"]
        ),
        attention_head_size=config["attention_head_size"],
        dropout=config["dropout"],
        output_size=len(QUANTILES),
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=20,
        reduce_on_plateau_patience=3,
        optimizer="adam",
    )


def train_candidate(config, training_dataset, loaders):
    """Train one candidate and save its best checkpoint."""

    name = config["name"]
    candidate_dir = OPTIMIZATION_DIR / name
    checkpoint_dir = candidate_dir / "checkpoints"

    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )

    logger = CSVLogger(
        save_dir=str(candidate_dir),
        name="training_logs",
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        min_epochs=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    model = create_model(training_dataset, config)

    print(f"\n=== TRAINING {name.upper()} ===")
    print("Configuration:", config)

    trainer.fit(
        model,
        train_dataloaders=loaders[0],
        val_dataloaders=loaders[1],
    )

    if not checkpoint.best_model_path:
        raise RuntimeError(f"No checkpoint saved for {name}.")

    if checkpoint.best_model_score is None:
        raise RuntimeError(
            f"No validation score saved for {name}."
        )

    score = float(checkpoint.best_model_score)

    if not math.isfinite(score):
        raise ValueError(
            f"Invalid validation score for {name}."
        )

    return {
        "name": name,
        "config": config,
        "checkpoint": checkpoint.best_model_path,
        "validation_quantile_loss": score,
    }


def evaluate_candidate(checkpoint_path, validation_loader):
    """
    Score one frozen candidate on validation data.

    Reports P50 accuracy and raw crossing frequency.
    Rearranges quantiles for consistent output.
    """

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )

    model.eval()
    model.cpu()

    actual_values = []
    predicted_values = []
    covered_values = []
    crossing_values = []

    with torch.inference_mode():
        for x, y in validation_loader:
            output = model(x)

            raw = output.prediction.detach().cpu()

            target = (
                y[0] if isinstance(y, (tuple, list)) else y
            ).detach().cpu()

            if raw.ndim != 3 or raw.shape[-1] != 3:
                raise ValueError("Unexpected quantile output shape.")

            if raw.shape[:2] != target.shape:
                raise ValueError("Prediction and target mismatch.")

            if not torch.isfinite(raw).all():
                raise ValueError("Non-finite candidate predictions.")

            crossing = (
                (raw[..., 0] > raw[..., 1])
                | (raw[..., 1] > raw[..., 2])
            )

            ordered = torch.sort(raw, dim=-1).values

            actual_values.append(
                target.numpy().reshape(-1)
            )

            predicted_values.append(
                ordered[..., 1].numpy().reshape(-1)
            )

            covered_values.append(
                (
                    (target >= ordered[..., 0])
                    & (target <= ordered[..., 2])
                ).numpy().reshape(-1)
            )

            crossing_values.append(
                crossing.numpy().reshape(-1)
            )

    actual = np.concatenate(actual_values)
    predicted = np.concatenate(predicted_values)
    covered = np.concatenate(covered_values)
    crossing = np.concatenate(crossing_values)

    error = predicted - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    total_demand = float(np.sum(np.abs(actual)))

    wape = (
        100.0 * float(np.sum(np.abs(error))) / total_demand
        if total_demand > 0
        else None
    )

    return {
        "validation_forecast_count": int(actual.size),
        "validation_mae_kg": round(mae, 4),
        "validation_rmse_kg": round(rmse, 4),
        "validation_wape_percent": (
            round(wape, 4) if wape is not None else None
        ),
        "validation_interval_coverage_percent": round(
            float(covered.mean()) * 100.0, 4
        ),
        "validation_raw_crossing_percent": round(
            float(crossing.mean()) * 100.0, 4
        ),
    }


def main():
    prepare()

    dataframe = load_eight_input_data()

    datasets = create_eight_input_datasets(dataframe)

    check_dataset_integrity(dataframe, datasets)

    train_loader, val_loader = create_loaders(datasets)

    results = []

    for config in CANDIDATES:
        # Same seed gives each candidate a reproducible run.
        pl.seed_everything(RANDOM_SEED, workers=True)

        trained = train_candidate(
            config,
            datasets["training"],
            (train_loader, val_loader),
        )

        validation_metrics = evaluate_candidate(
            trained["checkpoint"],
            val_loader,
        )

        result = {
            "name": trained["name"],
            "checkpoint": trained["checkpoint"],
            "hidden_size": config["hidden_size"],
            "hidden_continuous_size": (
                config["hidden_continuous_size"]
            ),
            "attention_head_size": (
                config["attention_head_size"]
            ),
            "dropout": config["dropout"],
            "validation_quantile_loss": (
                trained["validation_quantile_loss"]
            ),
            **validation_metrics,
        }

        results.append(result)

        print(f"\n=== {config['name'].upper()} RESULTS ===")
        for key, value in result.items():
            print(f"{key}: {value}")

    results_frame = pd.DataFrame(results)

    # Select according to validation MAE.
    # Quantile loss is the tie-breaker.
    selected = sorted(
        results,
        key=lambda row: (
            row["validation_mae_kg"],
            row["validation_quantile_loss"],
        ),
    )[0]

    results_frame.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary = {
        "model_family": "nine_input_tft",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "encoder_weeks": ENCODER_LENGTH,
        "forecast_weeks": PREDICTION_LENGTH,
        "historical_features": HISTORICAL_ONLY_REALS,
        "selection_metric": "validation_mae_kg",
        "selected_candidate": selected,
        "all_candidates": results,
        "held_out_test_evaluated": False,
        "factory_model_replaced": False,
        "notes": [
            "Candidate selection uses validation data only.",
            "All nine historical features are retained.",
            "Raw quantile crossings are reported.",
            "The previous trained checkpoint is unchanged.",
            "Test performance is not assumed from validation results.",
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== OPTIMIZATION RESULTS ===")
    print(
        results_frame[
            [
                "name",
                "validation_mae_kg",
                "validation_rmse_kg",
                "validation_wape_percent",
                "validation_interval_coverage_percent",
                "validation_quantile_loss",
            ]
        ].to_string(index=False)
    )

    print("\n=== SELECTED CANDIDATE ===")
    print("Name:", selected["name"])
    print("Checkpoint:", selected["checkpoint"])
    print("Validation MAE:", selected["validation_mae_kg"])

    print("\nResults:", RESULTS_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Existing checkpoint modified: NO")
    print("Held-out test evaluated: NO")
    print("Factory model replaced: NO")
    print("\nNINE-INPUT TFT OPTIMIZATION: PASS")


if __name__ == "__main__":
    main()
