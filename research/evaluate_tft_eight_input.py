
"""
Evaluate the trained nine-input TFT on held-out test data.

Features:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_monday_rate
    usd_lkr_sunday_rate
    festival_flag
    auction_price_high_lkr
    auction_price_medium_lkr
    auction_price_low_lkr

Forecast horizon: four weeks.
Historical encoder: 52 weeks.

Metrics:
    MAE, RMSE, WAPE, sMAPE
    P10-P90 interval coverage and width
    Raw quantile crossing frequency

Baselines:
    Last observed demand
    Corresponding demand 52 weeks before target

Crossed quantiles are rearranged in ascending order.
The original crossing frequency is reported explicitly.

No retraining or checkpoint modification is performed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from build_eight_input_dataset import (
    load_eight_input_data,
    create_eight_input_datasets,
)

from tft_eight_input_config import (
    OUTPUT_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    TEA_GRADES,
    TEST_START,
    HISTORICAL_ONLY_REALS,
    validate_feature_configuration,
)


REPORT_PATH = OUTPUT_DIR / "nine_input_training_metadata.json"

PREDICTIONS_FILE = OUTPUT_DIR / "nine_input_test_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "nine_input_test_metrics.csv"
SUMMARY_FILE = OUTPUT_DIR / "nine_input_test_summary.json"


def load_frozen_model():
    """Load the checkpoint selected during training."""

    validate_feature_configuration()

    if MODEL_NAME != "nine_input_tft":
        raise ValueError("Expected nine_input_tft configuration.")

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Training metadata missing: {REPORT_PATH}"
        )

    report = json.loads(
        REPORT_PATH.read_text(encoding="utf-8")
    )

    if report.get("model") != MODEL_NAME:
        raise ValueError("Training report model does not match.")

    if report.get("status") != "trained":
        raise ValueError("Training is not marked as completed.")

    if report.get("historical_only_features") != HISTORICAL_ONLY_REALS:
        raise ValueError(
            "Training and evaluation historical features differ."
        )

    checkpoint = Path(report["checkpoint"])

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint missing: {checkpoint}"
        )

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )

    model.eval()
    model.cpu()

    checkpoint_quantiles = getattr(
        model.loss, "quantiles", None
    )

    if checkpoint_quantiles is not None:
        if not np.allclose(
            np.asarray(checkpoint_quantiles, dtype=float),
            np.asarray(QUANTILES, dtype=float),
        ):
            raise ValueError(
                "Checkpoint quantiles differ from configuration."
            )

    return model, checkpoint, report


def validate_and_rearrange_predictions(predictions, actuals):
    """
    Validate raw TFT output and rearrange crossed quantiles.

    Returns:
        corrected: Tensor [batch, horizon, quantiles]
        crossings: Boolean Tensor [batch, horizon]

    Rearrangement sorts within each individual forecast.
    It never mixes different weeks or tea grades.
    """

    if predictions.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional prediction tensor."
        )

    expected_tail = (PREDICTION_LENGTH, len(QUANTILES))

    if tuple(predictions.shape[1:]) != expected_tail:
        raise ValueError(
            f"Unexpected prediction shape: {tuple(predictions.shape)}"
        )

    if tuple(actuals.shape) != tuple(predictions.shape[:2]):
        raise ValueError(
            "Actual and predicted demand dimensions differ."
        )

    if not torch.isfinite(predictions).all():
        raise ValueError("Non-finite model predictions detected.")

    if not torch.isfinite(actuals).all():
        raise ValueError("Non-finite actual demand detected.")

    if list(QUANTILES) != [0.1, 0.5, 0.9]:
        raise ValueError(
            "This evaluator expects P10, P50 and P90."
        )

    crossings = (
        (predictions[:, :, 0] > predictions[:, :, 1])
        | (predictions[:, :, 1] > predictions[:, :, 2])
    )

    corrected = torch.sort(
        predictions,
        dim=-1,
    ).values

    if not (
        (corrected[:, :, 0] <= corrected[:, :, 1])
        & (corrected[:, :, 1] <= corrected[:, :, 2])
    ).all():
        raise ValueError(
            "Quantile rearrangement failed."
        )

    return corrected, crossings


def generate_test_predictions(model, dataset):
    """Generate TFT predictions and historical baselines."""

    loader = dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    rows = []

    with torch.inference_mode():
        for x, y in loader:
            output = model(x)

            raw_predictions = (
                output.prediction.detach().cpu()
            )

            target_tensor = (
                y[0] if isinstance(y, (tuple, list)) else y
            )

            actuals = target_tensor.detach().cpu()

            predictions, crossings = (
                validate_and_rearrange_predictions(
                    raw_predictions, actuals
                )
            )

            encoder_targets = (
                x["encoder_target"].detach().cpu()
            )

            if encoder_targets.shape[1] != ENCODER_LENGTH:
                raise ValueError(
                    "Unexpected encoder history length."
                )

            prediction_index = dataset.x_to_index(x)

            if len(prediction_index) != len(predictions):
                raise ValueError(
                    "Prediction index length mismatch."
                )

            for required in ["grade", "time_idx"]:
                if required not in prediction_index.columns:
                    raise ValueError(
                        f"Prediction index missing: {required}"
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

                last_observed = float(
                    encoder_targets[sample, -1]
                )

                for horizon in range(
                    1, PREDICTION_LENGTH + 1
                ):
                    step = horizon - 1

                    # For target week origin+horizon,
                    # the corresponding week 52 weeks
                    # earlier is at horizon-53 in
                    # the 52-position encoder.
                    seasonal_position = horizon - 53

                    if not (
                        -ENCODER_LENGTH
                        <= seasonal_position
                        < 0
                    ):
                        raise ValueError(
                            "Invalid seasonal baseline index."
                        )

                    seasonal_value = float(
                        encoder_targets[
                            sample, seasonal_position
                        ]
                    )

                    rows.append({
                        "grade": grade,
                        "target_time_idx": (
                            prediction_start + step
                        ),
                        "horizon": horizon,
                        "actual_demand_kg": float(
                            actuals[sample, step]
                        ),
                        "p10_kg": float(
                            predictions[sample, step, 0]
                        ),
                        "p50_kg": float(
                            predictions[sample, step, 1]
                        ),
                        "p90_kg": float(
                            predictions[sample, step, 2]
                        ),
                        "raw_p10_kg": float(
                            raw_predictions[sample, step, 0]
                        ),
                        "raw_p50_kg": float(
                            raw_predictions[sample, step, 1]
                        ),
                        "raw_p90_kg": float(
                            raw_predictions[sample, step, 2]
                        ),
                        "quantile_crossing": bool(
                            crossings[sample, step]
                        ),
                        "last_baseline_kg": last_observed,
                        "seasonal_baseline_kg": seasonal_value,
                    })

    results = pd.DataFrame(rows)

    expected_rows = len(dataset) * PREDICTION_LENGTH

    if len(results) != expected_rows:
        raise ValueError(
            f"Incomplete predictions: "
            f"{len(results)} != {expected_rows}"
        )

    return results


def calculate_metrics(actual, prediction):
    """Calculate point-forecast accuracy in kg."""

    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if actual.size == 0 or actual.shape != prediction.shape:
        raise ValueError(
            "Empty or mismatched metric arrays."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(prediction).all()
    ):
        raise ValueError(
            "Non-finite values in metric calculation."
        )

    difference = prediction - actual
    absolute_error = np.abs(difference)

    mae = float(np.mean(absolute_error))

    rmse = float(
        np.sqrt(np.mean(difference ** 2))
    )

    actual_total = float(np.abs(actual).sum())

    wape = (
        float(100.0 * absolute_error.sum() / actual_total)
        if actual_total > 0
        else None
    )

    denominator = np.abs(actual) + np.abs(prediction)

    smape = float(
        np.mean(
            np.divide(
                200.0 * absolute_error,
                denominator,
                out=np.zeros_like(absolute_error),
                where=denominator > 0,
            )
        )
    )

    return {
        "mae_kg": round(mae, 4),
        "rmse_kg": round(rmse, 4),
        "wape_percent": (
            round(wape, 4) if wape is not None else None
        ),
        "smape_percent": round(smape, 4),
    }


def evaluate_group(group, grade, horizon):
    """Evaluate all three forecasting approaches."""

    if group.empty:
        raise ValueError(
            "Cannot evaluate an empty prediction group."
        )

    actual = group["actual_demand_kg"].to_numpy(
        dtype=float
    )

    model_columns = {
        MODEL_NAME: "p50_kg",
        "last_observed": "last_baseline_kg",
        "seasonal_52w": "seasonal_baseline_kg",
    }

    rows = []

    for name, prediction_column in model_columns.items():
        metrics = calculate_metrics(
            actual,
            group[prediction_column].to_numpy(
                dtype=float
            ),
        )

        record = {
            "grade": grade,
            "horizon": horizon,
            "model": name,
            "sample_count": len(group),
            **metrics,
            "interval_coverage_percent": None,
            "mean_interval_width_kg": None,
            "raw_crossing_percent": None,
        }

        if name == MODEL_NAME:
            lower = group["p10_kg"].to_numpy(
                dtype=float
            )
            upper = group["p90_kg"].to_numpy(
                dtype=float
            )

            covered = (
                (actual >= lower) & (actual <= upper)
            )

            record["interval_coverage_percent"] = round(
                100.0 * float(np.mean(covered)),
                4,
            )

            record["mean_interval_width_kg"] = round(
                float(np.mean(upper - lower)),
                4,
            )

            record["raw_crossing_percent"] = round(
                100.0
                * float(group["quantile_crossing"].mean()),
                4,
            )

        rows.append(record)

    return rows


def build_metrics_table(predictions):
    """Calculate overall, per-grade and per-horizon results."""

    rows = []

    rows.extend(
        evaluate_group(predictions, "ALL", "ALL")
    )

    for grade in TEA_GRADES:
        grade_rows = predictions.loc[
            predictions["grade"] == grade
        ]

        rows.extend(
            evaluate_group(grade_rows, grade, "ALL")
        )

        for horizon in range(
            1, PREDICTION_LENGTH + 1
        ):
            subset = grade_rows.loc[
                grade_rows["horizon"] == horizon
            ]

            rows.extend(
                evaluate_group(subset, grade, horizon)
            )

    for horizon in range(
        1, PREDICTION_LENGTH + 1
    ):
        subset = predictions.loc[
            predictions["horizon"] == horizon
        ]

        rows.extend(
            evaluate_group(subset, "ALL", horizon)
        )

    return pd.DataFrame(rows)


def clean_json_records(frame):
    """Return JSON-safe DataFrame records."""

    records = []

    for record in frame.to_dict(orient="records"):
        result = {}

        for key, value in record.items():
            if pd.isna(value):
                result[key] = None
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value

        records.append(result)

    return records


def main():
    print(
        "\n=== NINE-INPUT TFT HELD-OUT EVALUATION ==="
    )

    # Avoid accidentally replacing old evaluation results.
    for path in [
        PREDICTIONS_FILE,
        METRICS_FILE,
        SUMMARY_FILE,
    ]:
        if path.exists():
            raise FileExistsError(
                f"Evaluation output already exists: {path}"
            )

    model, checkpoint, training_report = (
        load_frozen_model()
    )

    dataframe = load_eight_input_data()

    datasets = create_eight_input_datasets(
        dataframe
    )

    testing = datasets["testing"]

    print("Model:", MODEL_NAME)
    print("Checkpoint:", checkpoint.name)
    print("Held-out test windows:", len(testing))

    predictions = generate_test_predictions(
        model, testing
    )

    calendar = (
        dataframe[["time_idx", "week_start_date"]]
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
            "One or more forecast dates are missing."
        )

    if (
        predictions["week_start_date"] < TEST_START
    ).any():
        raise ValueError(
            "Test forecasts contain dates before cutoff."
        )

    predictions["week_start_date"] = (
        predictions["week_start_date"].dt.strftime(
            "%Y-%m-%d"
        )
    )

    unique_keys = [
        "grade",
        "week_start_date",
        "horizon",
    ]

    if predictions.duplicated(unique_keys).any():
        raise ValueError(
            "Duplicate grade/date/horizon forecasts."
        )

    metrics = build_metrics_table(predictions)

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
                "wape_percent",
                "smape_percent",
                "interval_coverage_percent",
                "raw_crossing_percent",
            ]
        ].to_string(index=False)
    )

    horizons = metrics.loc[
        (metrics["grade"] == "ALL")
        & (metrics["horizon"] != "ALL")
    ]

    print("\n=== FORECAST HORIZON METRICS ===")
    print(
        horizons[
            [
                "horizon",
                "model",
                "mae_kg",
                "rmse_kg",
                "wape_percent",
            ]
        ].to_string(index=False)
    )

    grade_metrics = metrics.loc[
        (metrics["grade"] != "ALL")
        & (metrics["horizon"] == "ALL")
        & (metrics["model"] == MODEL_NAME)
    ]

    print("\n=== TFT PERFORMANCE BY TEA GRADE ===")
    print(
        grade_metrics[
            [
                "grade",
                "mae_kg",
                "rmse_kg",
                "wape_percent",
                "interval_coverage_percent",
                "raw_crossing_percent",
            ]
        ].to_string(index=False)
    )

    crossing_count = int(
        predictions["quantile_crossing"].sum()
    )

    crossing_percent = round(
        100.0 * crossing_count / len(predictions),
        4,
    )

    print("\n=== QUANTILE CROSSING DIAGNOSTICS ===")
    print("Total forecasts:", len(predictions))
    print("Raw crossing count:", crossing_count)
    print("Raw crossing percentage:", crossing_percent)
    print("Correction method: ascending rearrangement")

    summary = {
        "model": MODEL_NAME,
        "evaluation": "held_out_test",
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "checkpoint": str(checkpoint),
        "test_start": str(TEST_START.date()),
        "test_windows": len(testing),
        "prediction_rows": len(predictions),
        "encoder_weeks": ENCODER_LENGTH,
        "forecast_horizon_weeks": PREDICTION_LENGTH,
        "historical_features": HISTORICAL_ONLY_REALS,
        "quantile_rearrangement": {
            "method": "ascending_sort",
            "raw_crossing_count": crossing_count,
            "raw_crossing_percent": crossing_percent,
        },
        "overall_metrics": clean_json_records(overall),
        "grade_metrics": clean_json_records(
            grade_metrics
        ),
        "notes": [
            "The trained checkpoint was not modified.",
            "Original quantile crossings are recorded.",
            "Metrics use rearranged quantiles for the TFT.",
            "The held-out test partition was not used for checkpoint selection.",
            "Overlapping forecast windows are not independent observations.",
            "Baselines use historical encoder demand.",
            "Factory deployment requires separate integration.",
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

    print("\nNINE-INPUT TFT TEST EVALUATION: PASS")
    print("Model retrained: NO")
    print("Factory model replaced: NO")


if __name__ == "__main__":
    main()
