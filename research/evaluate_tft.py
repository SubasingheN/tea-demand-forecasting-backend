
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from build_tft_datasets import (
    GRADES,
    PREDICTION_LENGTH,
    load_features,
    load_manifests,
    create_datasets,
    validate_datasets,
    validate_feature_roles,
)


# ============================================================
# PHASE 2 - STEP 12
# TFT VALIDATION EVALUATION - CORRECTED
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "research"
    / "outputs"
    / "tft_full_training"
    / "checkpoints"
    / "tft-epoch09-val496.3011.ckpt"
)

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_validation_evaluation"
)

PREDICTIONS_FILE = OUTPUT_DIR / "validation_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "validation_summary.json"
GRADE_FILE = OUTPUT_DIR / "metrics_by_grade.csv"
HORIZON_FILE = OUTPUT_DIR / "metrics_by_horizon.csv"
DETAIL_FILE = OUTPUT_DIR / "metrics_by_grade_and_horizon.csv"

BATCH_SIZE = 32
QUANTILES = [0.1, 0.5, 0.9]


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def prepare_validation():
    print("\nLOADING VALIDATION DATA")
    print("-" * 65)

    model_df = load_features()
    manifests = load_manifests()

    training, validation, testing = create_datasets(
        model_df
    )

    validate_datasets(
        {
            "train": training,
            "validation": validation,
            "test": testing,
        },
        manifests,
        model_df,
    )

    validate_feature_roles(training)

    # A shared, validated mapping from integer
    # week index to the corresponding Monday.
    date_mapping = (
        model_df[
            ["time_idx", "week_start_date"]
        ]
        .drop_duplicates()
        .copy()
    )

    if date_mapping["time_idx"].duplicated().any():
        raise ValueError(
            "A time index maps to multiple dates."
        )

    index_to_date = {
        int(row.time_idx): pd.Timestamp(
            row.week_start_date
        )
        for row in date_mapping.itertuples(index=False)
    }

    print("Validation windows:", len(validation))
    print("Test data: NOT EVALUATED")

    return (
        validation,
        manifests["validation"],
        index_to_date,
    )


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    print("\nLOADING TRAINED TFT MODEL")
    print("-" * 65)

    # Only load a checkpoint produced by your
    # trusted local training process.
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
        weights_only=False,
    )

    model.eval()
    model.cpu()

    print("Checkpoint:", CHECKPOINT)
    print("Model:", type(model).__name__)

    return model


def calculate_metrics(frame, prediction_column):
    actual = frame["actual_demand_kg"].to_numpy(
        dtype=np.float64
    )

    predicted = frame[prediction_column].to_numpy(
        dtype=np.float64
    )

    error = actual - predicted

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    denominator = float(
        np.sum(np.abs(actual))
    )

    wape = (
        float(
            100 * np.sum(np.abs(error)) / denominator
        )
        if denominator > 0
        else None
    )

    return {
        "observations": int(len(frame)),
        "mae_kg": mae,
        "rmse_kg": rmse,
        "wape_percent": wape,
    }


def evaluate_predictions(
    model,
    validation,
    manifest,
    index_to_date,
):
    print("\nGENERATING TFT PREDICTIONS")
    print("-" * 65)

    loader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    result = model.predict(
        loader,
        mode="quantiles",
        return_x=True,
        return_y=True,
        return_index=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": True,
        },
    )

    quantiles = to_numpy(result.output)

    if (
        quantiles.ndim != 3
        or quantiles.shape[1] != PREDICTION_LENGTH
        or quantiles.shape[2] != 3
    ):
        raise ValueError(
            f"Unexpected quantile shape: {quantiles.shape}"
        )

    if not np.isfinite(quantiles).all():
        raise ValueError(
            "Nonfinite TFT quantile predictions."
        )

    actual_output = result.y

    if isinstance(actual_output, (list, tuple)):
        actual_output = actual_output[0]

    actual = to_numpy(actual_output)

    encoder_target = to_numpy(
        result.x["encoder_target"]
    )

    decoder_time_idx = to_numpy(
        result.x["decoder_time_idx"]
    )

    indices = result.index.copy()

    print("Prediction array shape:", quantiles.shape)
    print("Actual target shape:", actual.shape)
    print("Decoder time index shape:", decoder_time_idx.shape)
    print("Returned index columns:", list(indices.columns))

    expected_shape = quantiles.shape[:2]

    if actual.shape != expected_shape:
        raise ValueError(
            "Prediction and target shapes do not match."
        )

    if decoder_time_idx.shape != expected_shape:
        raise ValueError(
            "Decoder time index shape does not match."
        )

    if encoder_target.shape[0] != len(quantiles):
        raise ValueError(
            "Encoder sample count mismatch."
        )

    if encoder_target.shape[1] != 52:
        raise ValueError(
            "Expected 52 historical encoder weeks."
        )

    if "grade" not in indices.columns:
        raise ValueError(
            "Returned index is missing grade. "
            f"Columns: {list(indices.columns)}"
        )

    if len(indices) != len(quantiles):
        raise ValueError(
            "Prediction index row count mismatch."
        )

    expected_manifest_keys = set()

    for row in manifest.itertuples(index=False):
        expected_manifest_keys.add(
            (
                str(row.grade),
                pd.Timestamp(row.target_start).strftime(
                    "%Y-%m-%d"
                ),
            )
        )

    actual_window_keys = set()
    rows = []
    quantile_crossings = 0

    for sample in range(len(quantiles)):
        grade = str(
            indices.iloc[sample]["grade"]
        )

        if grade not in GRADES:
            raise ValueError(
                f"Unexpected predicted grade: {grade}"
            )

        decoder_indices = decoder_time_idx[
            sample
        ].astype(int)

        if not np.array_equal(
            np.diff(decoder_indices),
            np.ones(PREDICTION_LENGTH - 1, dtype=int),
        ):
            raise ValueError(
                "Nonconsecutive decoder weeks."
            )

        target_start_idx = int(
            decoder_indices[0]
        )

        if target_start_idx not in index_to_date:
            raise ValueError(
                "Unknown decoder time index: "
                f"{target_start_idx}"
            )

        target_start = index_to_date[
            target_start_idx
        ]

        key = (
            grade,
            target_start.strftime("%Y-%m-%d"),
        )

        if key in actual_window_keys:
            raise ValueError(
                f"Duplicate forecast window: {key}"
            )

        actual_window_keys.add(key)

        last_observed_demand = float(
            encoder_target[sample, -1]
        )

        for horizon in range(
            PREDICTION_LENGTH
        ):
            idx = int(
                decoder_indices[horizon]
            )

            if idx not in index_to_date:
                raise ValueError(
                    f"Unknown forecast time index: {idx}"
                )

            target_date = index_to_date[idx]

            q10, q50, q90 = (
                float(value)
                for value in quantiles[
                    sample, horizon
                ]
            )

            if q10 > q50 or q50 > q90:
                quantile_crossings += 1

            actual_kg = float(
                actual[sample, horizon]
            )

            rows.append({
                "grade": grade,
                "forecast_origin": (
                    target_start
                    - pd.Timedelta(weeks=1)
                ).strftime("%Y-%m-%d"),
                "target_start": target_start.strftime(
                    "%Y-%m-%d"
                ),
                "target_week": target_date.strftime(
                    "%Y-%m-%d"
                ),
                "horizon": horizon + 1,
                "actual_demand_kg": actual_kg,
                "tft_q10_kg": q10,
                "tft_q50_kg": q50,
                "tft_q90_kg": q90,
                "last_observation_baseline_kg": (
                    last_observed_demand
                ),
                "within_80_percent_interval": int(
                    q10 <= actual_kg <= q90
                ),
            })

    if actual_window_keys != expected_manifest_keys:
        missing = sorted(
            expected_manifest_keys - actual_window_keys
        )
        unexpected = sorted(
            actual_window_keys - expected_manifest_keys
        )

        raise ValueError(
            "Predictions do not match Step 8 manifest.\n"
            f"Missing: {missing[:5]}\n"
            f"Unexpected: {unexpected[:5]}"
        )

    frame = pd.DataFrame(rows)

    if len(frame) != len(manifest) * PREDICTION_LENGTH:
        raise ValueError(
            "Incorrect number of evaluation rows."
        )

    if frame["actual_demand_kg"].isna().any():
        raise ValueError(
            "Missing actual customer demand."
        )

    for column in [
        "actual_demand_kg",
        "tft_q10_kg",
        "tft_q50_kg",
        "tft_q90_kg",
        "last_observation_baseline_kg",
    ]:
        if not np.isfinite(
            frame[column].to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                f"Nonfinite values in {column}"
            )

    print("Matched validation windows:", len(actual_window_keys))
    print("Prediction observations:", len(frame))
    print("Quantile crossing observations:", quantile_crossings)

    return frame, quantile_crossings


def summarize_group(group):
    tft = calculate_metrics(
        group,
        "tft_q50_kg",
    )

    baseline = calculate_metrics(
        group,
        "last_observation_baseline_kg",
    )

    improvement = None

    if baseline["mae_kg"] > 0:
        improvement = float(
            100 * (
                baseline["mae_kg"] - tft["mae_kg"]
            ) / baseline["mae_kg"]
        )

    return {
        "observations": tft["observations"],
        "tft_mae_kg": tft["mae_kg"],
        "tft_rmse_kg": tft["rmse_kg"],
        "tft_wape_percent": tft["wape_percent"],
        "baseline_mae_kg": baseline["mae_kg"],
        "baseline_rmse_kg": baseline["rmse_kg"],
        "baseline_wape_percent": baseline["wape_percent"],
        "tft_mae_improvement_percent": improvement,
        "interval_80_coverage_percent": float(
            100 * group[
                "within_80_percent_interval"
            ].mean()
        ),
    }


def build_metric_tables(frame):
    grade_rows = []

    for grade in GRADES:
        subset = frame[
            frame["grade"] == grade
        ]

        grade_rows.append({
            "grade": grade,
            **summarize_group(subset),
        })

    horizon_rows = []

    for horizon in range(
        1, PREDICTION_LENGTH + 1
    ):
        subset = frame[
            frame["horizon"] == horizon
        ]

        horizon_rows.append({
            "horizon": horizon,
            **summarize_group(subset),
        })

    detail_rows = []

    for grade in GRADES:
        for horizon in range(
            1, PREDICTION_LENGTH + 1
        ):
            subset = frame[
                (frame["grade"] == grade)
                & (frame["horizon"] == horizon)
            ]

            detail_rows.append({
                "grade": grade,
                "horizon": horizon,
                **summarize_group(subset),
            })

    return (
        pd.DataFrame(grade_rows),
        pd.DataFrame(horizon_rows),
        pd.DataFrame(detail_rows),
        summarize_group(frame),
    )


def save_results(
    frame,
    grade_metrics,
    horizon_metrics,
    detail_metrics,
    overall,
    crossing_count,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    grade_metrics.to_csv(
        GRADE_FILE,
        index=False,
    )

    horizon_metrics.to_csv(
        HORIZON_FILE,
        index=False,
    )

    detail_metrics.to_csv(
        DETAIL_FILE,
        index=False,
    )

    report = {
        "status": "PASS",
        "partition": "2024_validation",
        "checkpoint": str(CHECKPOINT),
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders including "
            "unfulfilled orders"
        ),
        "forecast_horizon": PREDICTION_LENGTH,
        "quantiles": QUANTILES,
        "forecast_windows": int(
            len(frame) // PREDICTION_LENGTH
        ),
        "prediction_rows": int(len(frame)),
        "quantile_crossing_observations": int(
            crossing_count
        ),
        "overall_metrics": overall,
        "grade_metrics": grade_metrics.to_dict(
            orient="records"
        ),
        "horizon_metrics": horizon_metrics.to_dict(
            orient="records"
        ),
        "test_partition_evaluated": False,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "limitations": [
            (
                "Validation windows overlap and "
                "are not statistically independent."
            ),
            (
                "The naive baseline repeats the last "
                "observed demand for all four weeks."
            ),
            (
                "Forecast accuracy alone does not "
                "establish an auction-price effect."
            ),
            (
                "Auction prices are quotation-based "
                "historical candidate indicators."
            ),
            (
                "Test data was not evaluated."
            ),
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    print(
        "PHASE 2 - STEP 12: "
        "TFT VALIDATION FORECAST EVALUATION"
    )
    print("=" * 65)

    (
        validation,
        manifest,
        index_to_date,
    ) = prepare_validation()

    model = load_model()

    frame, crossing_count = evaluate_predictions(
        model,
        validation,
        manifest,
        index_to_date,
    )

    (
        grade_metrics,
        horizon_metrics,
        detail_metrics,
        overall,
    ) = build_metric_tables(frame)

    save_results(
        frame,
        grade_metrics,
        horizon_metrics,
        detail_metrics,
        overall,
        crossing_count,
    )

    print("\nOVERALL FORECAST ACCURACY")
    print("-" * 65)

    print("Validation windows:", len(manifest))
    print("Prediction rows:", len(frame))

    for name, value in overall.items():
        print(f"{name}: {value}")

    print("\nMETRICS BY TEA GRADE")
    print(
        grade_metrics[
            [
                "grade",
                "tft_mae_kg",
                "baseline_mae_kg",
                "tft_wape_percent",
                "interval_80_coverage_percent",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nMETRICS BY FORECAST HORIZON")
    print(
        horizon_metrics[
            [
                "horizon",
                "tft_mae_kg",
                "baseline_mae_kg",
                "interval_80_coverage_percent",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print(PREDICTIONS_FILE)
    print(SUMMARY_FILE)
    print(GRADE_FILE)
    print(HORIZON_FILE)
    print(DETAIL_FILE)

    print("\nTEST DATASET: NOT EVALUATED")
    print("VALIDATION FORECAST EVALUATION: PASS")


if __name__ == "__main__":
    main()
