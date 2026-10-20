
"""
Held-out test evaluation for the five-input TFT
auction-price ablation experiment.

Uses the same prepared Version 2 dataset and
chronological test period as the eight-input TFT.

Metrics:
    MAE
    RMSE
    sMAPE
    P10/P90 interval coverage
    P10/P90 interval width

Baselines:
    Last observed demand
    52-week seasonal demand

No model retraining.
No existing evaluation files are overwritten.

The source dataset remains provisional.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from build_tft_five_input_ablation_datasets import (
    load_five_input_ablation_data,
    create_five_input_ablation_datasets,
    check_five_input_ablation_integrity,
)

from tft_five_input_ablation_config import (
    OUTPUT_DIR,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    TEA_GRADES,
    TEST_START,
)


MODEL_NAME = "five_input_ablation_tft"

REPORT_PATH = (
    OUTPUT_DIR
    / "five_input_ablation_training_metadata.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "five_input_ablation_test_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "five_input_ablation_test_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "five_input_ablation_test_summary.json"
)


def load_frozen_model():
    """Load the validation-selected ablation checkpoint."""

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Training report missing: {REPORT_PATH}"
        )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    if report.get("model") != MODEL_NAME:
        raise ValueError(
            "Unexpected model in training report: "
            f"{report.get('model')}"
        )

    if report.get("test_evaluated") is not False:
        raise ValueError(
            "Expected training metadata with "
            "test_evaluated=False."
        )

    checkpoint_path = Path(report["checkpoint"])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    if (
        checkpoint_path.resolve().parent
        != (OUTPUT_DIR / "checkpoints").resolve()
    ):
        raise ValueError(
            "Checkpoint is outside the dedicated "
            "ablation checkpoint directory."
        )

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
    )

    model.eval()

    return model, checkpoint_path, report


def validate_predictions(predictions, actuals):
    """Validate shapes, numeric values and quantiles."""

    if predictions.ndim != 3:
        raise ValueError(
            "Predictions must have three dimensions."
        )

    expected = (
        PREDICTION_LENGTH,
        len(QUANTILES),
    )

    if tuple(predictions.shape[1:]) != expected:
        raise ValueError(
            "Unexpected prediction shape: "
            f"{tuple(predictions.shape)}"
        )

    if tuple(actuals.shape) != tuple(
        predictions.shape[:2]
    ):
        raise ValueError(
            "Prediction and actual target shapes differ."
        )

    if not torch.isfinite(predictions).all():
        raise ValueError(
            "Non-finite forecast predictions detected."
        )

    if not torch.isfinite(actuals).all():
        raise ValueError(
            "Non-finite actual demand values detected."
        )

    if not (
        (predictions[:, :, 0] <= predictions[:, :, 1])
        & (predictions[:, :, 1] <= predictions[:, :, 2])
    ).all():
        raise ValueError(
            "P10, P50 and P90 are incorrectly ordered."
        )


def generate_test_predictions(model, dataset):
    """Generate forecasts and historical-only baselines."""

    loader = dataset.to_dataloader(
        train=False,
        batch_size=32,
        num_workers=0,
    )

    rows = []

    with torch.no_grad():
        for x, y in loader:
            output = model(x)

            predictions = (
                output.prediction.detach().cpu()
            )

            actual_tensor = (
                y[0]
                if isinstance(y, (tuple, list))
                else y
            )

            actuals = actual_tensor.detach().cpu()

            validate_predictions(
                predictions,
                actuals,
            )

            encoder_targets = (
                x["encoder_target"].detach().cpu()
            )

            if encoder_targets.shape[1] != ENCODER_LENGTH:
                raise ValueError(
                    "Unexpected encoder length."
                )

            last_observed = encoder_targets[:, -1]

            prediction_index = dataset.x_to_index(x)

            if len(prediction_index) != len(predictions):
                raise ValueError(
                    "Prediction index length mismatch."
                )

            if not {
                "grade",
                "time_idx",
            }.issubset(prediction_index.columns):
                raise ValueError(
                    "Grade or time index missing."
                )

            for sample in range(len(predictions)):
                grade = str(
                    prediction_index.iloc[sample]["grade"]
                )

                if grade not in TEA_GRADES:
                    raise ValueError(
                        f"Unexpected tea grade: {grade}"
                    )

                prediction_start = int(
                    prediction_index.iloc[sample]["time_idx"]
                )

                for horizon in range(
                    1,
                    PREDICTION_LENGTH + 1,
                ):
                    index = horizon - 1

                    # Observed demand 52 weeks before
                    # the target date, inside the encoder.
                    seasonal_position = horizon - 53

                    if not (
                        -ENCODER_LENGTH
                        <= seasonal_position
                        < 0
                    ):
                        raise ValueError(
                            "Seasonal baseline position "
                            "falls outside encoder history."
                        )

                    seasonal_value = float(
                        encoder_targets[
                            sample,
                            seasonal_position,
                        ]
                    )

                    rows.append({
                        "grade": grade,
                        "target_time_idx": (
                            prediction_start + index
                        ),
                        "horizon": horizon,
                        "actual_demand_kg": float(
                            actuals[sample, index]
                        ),
                        "p10_kg": float(
                            predictions[sample, index, 0]
                        ),
                        "p50_kg": float(
                            predictions[sample, index, 1]
                        ),
                        "p90_kg": float(
                            predictions[sample, index, 2]
                        ),
                        "last_baseline_kg": float(
                            last_observed[sample]
                        ),
                        "seasonal_baseline_kg": (
                            seasonal_value
                        ),
                    })

    results = pd.DataFrame(rows)

    expected_rows = (
        len(dataset) * PREDICTION_LENGTH
    )

    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} forecasts, "
            f"received {len(results)}."
        )

    return results


def calculate_metrics(actual, forecast):
    """Calculate MAE, RMSE and sMAPE."""

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    if len(actual) == 0:
        raise ValueError(
            "Cannot evaluate an empty sample."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(forecast).all()
    ):
        raise ValueError(
            "Metric inputs must be finite."
        )

    error = forecast - actual

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    denominator = (
        np.abs(actual) + np.abs(forecast)
    )

    smape_terms = np.divide(
        200.0 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 0,
    )

    return {
        "mae_kg": round(float(mae), 4),
        "rmse_kg": round(float(rmse), 4),
        "smape_percent": round(
            float(np.mean(smape_terms)),
            4,
        ),
    }


def evaluate_group(group, grade, horizon):
    """Score TFT and baseline forecasts."""

    actual = group[
        "actual_demand_kg"
    ].to_numpy()

    model_columns = {
        MODEL_NAME: "p50_kg",
        "last_observed": "last_baseline_kg",
        "seasonal_52w": "seasonal_baseline_kg",
    }

    results = []

    for model_name, column in model_columns.items():
        metric = calculate_metrics(
            actual,
            group[column].to_numpy(),
        )

        row = {
            "grade": grade,
            "horizon": horizon,
            "model": model_name,
            "sample_count": len(group),
            **metric,
            "interval_coverage_percent": None,
            "mean_interval_width_kg": None,
        }

        if model_name == MODEL_NAME:
            lower = group[
                "p10_kg"
            ].to_numpy()

            upper = group[
                "p90_kg"
            ].to_numpy()

            covered = (
                (actual >= lower)
                & (actual <= upper)
            )

            row["interval_coverage_percent"] = round(
                100.0 * float(np.mean(covered)),
                4,
            )

            row["mean_interval_width_kg"] = round(
                float(np.mean(upper - lower)),
                4,
            )

        results.append(row)

    return results


def build_metrics_table(predictions):
    """Calculate overall, grade and horizon scores."""

    results = []

    results.extend(
        evaluate_group(
            predictions,
            grade="ALL",
            horizon="ALL",
        )
    )

    for grade in TEA_GRADES:
        grade_rows = predictions.loc[
            predictions["grade"] == grade
        ]

        results.extend(
            evaluate_group(
                grade_rows,
                grade=grade,
                horizon="ALL",
            )
        )

        for horizon in range(
            1,
            PREDICTION_LENGTH + 1,
        ):
            group = grade_rows.loc[
                grade_rows["horizon"] == horizon
            ]

            results.extend(
                evaluate_group(
                    group,
                    grade=grade,
                    horizon=horizon,
                )
            )

    for horizon in range(
        1,
        PREDICTION_LENGTH + 1,
    ):
        group = predictions.loc[
            predictions["horizon"] == horizon
        ]

        results.extend(
            evaluate_group(
                group,
                grade="ALL",
                horizon=horizon,
            )
        )

    return pd.DataFrame(results)


def main():
    print(
        "\n=== FIVE-INPUT TFT ABLATION HELD-OUT EVALUATION ==="
    )

    # Preserve previous research results.
    for output_file in [
        PREDICTIONS_FILE,
        METRICS_FILE,
        SUMMARY_FILE,
    ]:
        if output_file.exists():
            raise FileExistsError(
                "Refusing to overwrite: "
                f"{output_file}"
            )

    model, checkpoint, training_report = (
        load_frozen_model()
    )

    df = load_five_input_ablation_data()

    datasets = create_five_input_ablation_datasets(
        df
    )

    check_five_input_ablation_integrity(
        df,
        datasets,
    )

    test_dataset = datasets["testing"]

    print("\nModel:", MODEL_NAME)
    print("Checkpoint:", checkpoint.name)
    print("Test windows:", len(test_dataset))
    print("Test evaluation only: YES")

    predictions = generate_test_predictions(
        model,
        test_dataset,
    )

    # Use actual reporting dates for comparison,
    # not time_idx from separate dataset versions.
    calendar = (
        df[["time_idx", "week_start_date"]]
        .drop_duplicates()
    )

    predictions = predictions.merge(
        calendar,
        left_on="target_time_idx",
        right_on="time_idx",
        how="left",
        validate="many_to_one",
    )

    if predictions["week_start_date"].isna().any():
        raise ValueError(
            "Missing forecast target dates."
        )

    if (
        predictions["week_start_date"] < TEST_START
    ).any():
        raise ValueError(
            "Test predictions contain pre-test dates."
        )

    predictions["week_start_date"] = (
        predictions["week_start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    comparison_keys = [
        "grade",
        "week_start_date",
        "horizon",
    ]

    if predictions.duplicated(
        comparison_keys
    ).any():
        raise ValueError(
            "Duplicate matched forecast keys."
        )

    metrics = build_metrics_table(
        predictions
    )

    overall = metrics.loc[
        (metrics["grade"] == "ALL")
        & (metrics["horizon"] == "ALL")
    ]

    print("\n=== OVERALL TEST METRICS ===")

    print(
        overall[
            [
                "model",
                "sample_count",
                "mae_kg",
                "rmse_kg",
                "smape_percent",
                "interval_coverage_percent",
            ]
        ].to_string(index=False)
    )

    print("\n=== FORECAST HORIZON METRICS ===")

    horizon_metrics = metrics.loc[
        (metrics["grade"] == "ALL")
        & (metrics["horizon"] != "ALL")
    ]

    print(
        horizon_metrics[
            [
                "horizon",
                "model",
                "mae_kg",
                "rmse_kg",
            ]
        ].to_string(index=False)
    )

    summary = {
        "model": MODEL_NAME,
        "evaluation": "held_out_test",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "checkpoint": str(checkpoint),
        "test_windows": len(test_dataset),
        "prediction_rows": len(predictions),
        "test_start": str(TEST_START.date()),
        "overall_metrics": [
            {
                key: (
                    None
                    if pd.isna(value)
                    else value
                )
                for key, value in record.items()
            }
            for record in overall.to_dict(
                orient="records"
            )
        ],
        "historical_only_features": (
            training_report.get(
                "historical_only_features",
                [],
            )
        ),
        "excluded_auction_features": (
            training_report.get(
                "excluded_auction_features",
                [],
            )
        ),
        "auction_features_used_by_model": False,
        "dataset_approval_status": "provisional",
        "notes": [
            "Evaluation uses the frozen ablation checkpoint.",
            "No model retraining was performed.",
            "Test data was held out during training.",
            "Seasonal and last-observed baselines use encoder history only.",
            "Overlapping test windows are scored separately.",
            "Overlapping forecasts are not independent observations.",
            "Compare with eight-input TFT using matched grade, target date and horizon.",
            "The source dataset retains unresolved auction audit flags.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== EVALUATION OUTPUT FILES ===")
    print(PREDICTIONS_FILE)
    print(METRICS_FILE)
    print(SUMMARY_FILE)

    print(
        "\nFIVE-INPUT TFT ABLATION TEST EVALUATION: PASS"
    )
    print("Model retrained during evaluation: NO")


if __name__ == "__main__":
    main()
