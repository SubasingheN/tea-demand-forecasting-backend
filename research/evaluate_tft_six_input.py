
"""
Held-out evaluation for the six-input TFT.

Evaluates the frozen checkpoint on chronological
test windows without retraining or modifying
existing datasets or checkpoints.

Metrics:
- MAE
- RMSE
- sMAPE
- P10/P90 interval coverage
- P10/P90 interval width

Baselines:
- Last observed weekly demand
- Corresponding week in the previous year
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TemporalFusionTransformer,
)

from build_tft_six_input_datasets import (
    load_six_input_data,
    create_six_input_datasets,
)

from tft_six_input_config import (
    OUTPUT_DIR,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    TEA_GRADES,
    TEST_START,
)


REPORT_PATH = (
    OUTPUT_DIR / "six_input_training_metadata.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "six_input_test_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR / "six_input_test_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "six_input_test_summary.json"
)


def load_frozen_model():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Training report missing: {REPORT_PATH}"
        )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    checkpoint_path = Path(
        report["checkpoint"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint_path}"
        )

    model = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
        )
    )

    model.eval()

    return model, checkpoint_path


def validate_predictions(predictions, actuals):
    if predictions.ndim != 3:
        raise ValueError(
            "Expected three-dimensional predictions."
        )

    if predictions.shape[1:] != (
        PREDICTION_LENGTH,
        len(QUANTILES),
    ):
        raise ValueError(
            "Unexpected prediction dimensions."
        )

    if actuals.shape != predictions.shape[:2]:
        raise ValueError(
            "Actual and predicted shapes do not match."
        )

    if not torch.isfinite(predictions).all():
        raise ValueError(
            "Non-finite predictions detected."
        )

    if not torch.isfinite(actuals).all():
        raise ValueError(
            "Non-finite actual targets detected."
        )

    lower = predictions[:, :, 0]
    median = predictions[:, :, 1]
    upper = predictions[:, :, 2]

    if not (
        (lower <= median) &
        (median <= upper)
    ).all():
        raise ValueError(
            "Quantile predictions are not ordered."
        )


def generate_test_predictions(model, dataset):
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

            actuals = (
                y[0] if isinstance(y, (tuple, list))
                else y
            ).detach().cpu()

            validate_predictions(
                predictions,
                actuals,
            )

            encoder_targets = (
                x["encoder_target"]
                .detach()
                .cpu()
            )

            if encoder_targets.shape[1] != ENCODER_LENGTH:
                raise ValueError(
                    "Unexpected encoder length."
                )

            # Historical observations only:
            # no decoder targets are used by baselines.
            last_observed = (
                encoder_targets[:, -1]
            )

            # A separate seasonal baseline is
            # computed for each forecast horizon
            # using the observed value 52 weeks
            # before its target date.
            prediction_index = dataset.x_to_index(x)

            if len(prediction_index) != len(predictions):
                raise ValueError(
                    "Prediction index length mismatch."
                )

            if "grade" not in prediction_index.columns:
                raise ValueError(
                    "Tea-grade index missing."
                )

            if "time_idx" not in prediction_index.columns:
                raise ValueError(
                    "Prediction start index missing."
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

                    seasonal_encoder_position = (
                        horizon - 52 - 1
                    )

                    seasonal_value = float(
                        encoder_targets[
                            sample,
                            seasonal_encoder_position
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
            "Incomplete test prediction results."
        )

    return results


def calculate_metrics(actual, prediction):
    actual = np.asarray(
        actual, dtype=float
    )

    prediction = np.asarray(
        prediction, dtype=float
    )

    error = prediction - actual

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    denominator = (
        np.abs(actual) + np.abs(prediction)
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
    actual = group[
        "actual_demand_kg"
    ].to_numpy()

    models = {
        "six_input_tft": "p50_kg",
        "last_observed": "last_baseline_kg",
        "seasonal_52w": "seasonal_baseline_kg",
    }

    results = []

    for model_name, column in models.items():
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

        if model_name == "six_input_tft":
            lower = group[
                "p10_kg"
            ].to_numpy()

            upper = group[
                "p90_kg"
            ].to_numpy()

            covered = (
                (actual >= lower) &
                (actual <= upper)
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
    results = []

    results.extend(
        evaluate_group(
            predictions,
            grade="ALL",
            horizon="ALL",
        )
    )

    for grade in TEA_GRADES:
        grade_rows = predictions[
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
            group = grade_rows[
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
        group = predictions[
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
        "\n=== SIX-INPUT TFT HELD-OUT EVALUATION ==="
    )

    model, checkpoint = load_frozen_model()

    df = load_six_input_data()

    datasets = create_six_input_datasets(
        df
    )

    test_dataset = datasets["testing"]

    print("Checkpoint:", checkpoint.name)
    print("Held-out test windows:", len(test_dataset))

    predictions = generate_test_predictions(
        model,
        test_dataset,
    )

    # Convert integer time indices to actual
    # calendar weeks using the cleaned dataset.
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
            "Some forecast weeks are missing."
        )

    if (
        predictions["week_start_date"] < TEST_START
    ).any():
        raise ValueError(
            "Test predictions include pre-test targets."
        )

    predictions["week_start_date"] = (
        predictions["week_start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    # Each grade, forecast target and horizon
    # must have an unambiguous matched prediction.
    if predictions.duplicated(
        subset=[
            "grade",
            "target_time_idx",
            "horizon",
        ]
    ).any():
        raise ValueError(
            "Duplicate matched test predictions."
        )

    metrics = build_metrics_table(
        predictions
    )

    overall = metrics[
        (metrics["grade"] == "ALL") &
        (metrics["horizon"] == "ALL")
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Dedicated filenames; existing research
    # evaluation outputs are left unchanged.
    for output_file in [
        PREDICTIONS_FILE,
        METRICS_FILE,
        SUMMARY_FILE,
    ]:
        if output_file.exists():
            raise FileExistsError(
                "Refusing to overwrite existing "
                f"evaluation file: {output_file}"
            )

    summary = {
        "model": "six_input_tft",
        "evaluation": "held_out_test",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "checkpoint": str(checkpoint),
        "test_windows": len(test_dataset),
        "prediction_rows": len(predictions),
        "test_start": str(
            TEST_START.date()
        ),
        "overall_metrics": (
            overall.where(
                pd.notna(overall),
                None,
            )
            .to_dict(orient="records")
        ),
        "notes": [
            "Test targets were not used for training.",
            "Baselines use encoder observations only.",
            "Overlapping windows are evaluated separately.",
            "Results must not be interpreted as independent samples.",
            "No retraining was performed.",
        ],
    }

    PREDICTIONS_FILE.parent.mkdir(
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

    print("\nSIX-INPUT TFT TEST EVALUATION: PASS")
    print("Model retrained: NO")


if __name__ == "__main__":
    main()
