
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer

from build_tft_datasets import (
    GRADES,
    PREDICTION_LENGTH,
    load_manifests,
    validate_datasets,
)

from train_tft_no_price import (
    prepare_no_price_data,
    create_no_price_datasets,
)

from evaluate_tft import evaluate_predictions


# ============================================================
# PHASE 3 - STEP 16
# FINAL HELD-OUT TEST EVALUATION
#
# Frozen no-price TFT vs:
# 1. Last-observation baseline
# 2. Seasonal-naive baseline (52 weeks earlier)
#
# No retraining or hyperparameter tuning.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "research"
    / "outputs"
    / "tft_no_price"
    / "checkpoints"
    / "tft-no-price-epoch09-val470.2676.ckpt"
)

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_final_test"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "test_predictions.csv"
)

GRADE_FILE = (
    OUTPUT_DIR / "metrics_by_grade.csv"
)

HORIZON_FILE = (
    OUTPUT_DIR / "metrics_by_horizon.csv"
)

DETAIL_FILE = (
    OUTPUT_DIR / "metrics_by_grade_and_horizon.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "test_summary.json"
)

EXPECTED_TEST_WINDOWS = 445
EXPECTED_PREDICTIONS = (
    EXPECTED_TEST_WINDOWS * PREDICTION_LENGTH
)

SEASONAL_LAG_WEEKS = 52

BASELINES = {
    "tft": "tft_q50_kg",
    "last_observation": (
        "last_observation_baseline_kg"
    ),
    "seasonal_naive": (
        "seasonal_naive_baseline_kg"
    ),
}


# ============================================================
# LOAD THE FROZEN, SELECTED CHECKPOINT
# ============================================================

def load_selected_model():
    print("\nLOADING SELECTED NO-PRICE TFT")
    print("-" * 65)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # Only deserialize locally trained,
    # trusted checkpoint files.
    model = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            str(CHECKPOINT),
            map_location="cpu",
            weights_only=False,
        )
    )

    model.eval()
    model.cpu()

    expected_unknown = {
        "demand_kg",
        "production_kg",
        "inventory_kg",
    }

    actual_unknown = set(
        model.hparams.time_varying_reals_encoder
    ) if (
        hasattr(
            model.hparams,
            "time_varying_reals_encoder"
        )
    ) else None

    print("Checkpoint:", CHECKPOINT)
    print("Model:", type(model).__name__)

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    # Dataset feature-role checks are performed
    # separately. No checkpoint is modified.
    return model


# ============================================================
# REBUILD TEST DATASET WITHOUT REFITTING
# ============================================================

def prepare_test_dataset():
    print("\nREBUILDING THE NO-PRICE TEST DATASET")
    print("-" * 65)

    df = prepare_no_price_data()
    manifests = load_manifests()

    training, validation, testing = (
        create_no_price_datasets(df)
    )

    validate_datasets(
        {
            "train": training,
            "validation": validation,
            "test": testing,
        },
        manifests,
        df,
    )

    if len(testing) != EXPECTED_TEST_WINDOWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_WINDOWS} "
            f"test windows; found {len(testing)}"
        )

    if set(
        testing.time_varying_unknown_reals
    ) != {
        "demand_kg",
        "production_kg",
        "inventory_kg",
    }:
        raise ValueError(
            "Test dataset has unexpected "
            "historical-only model variables."
        )

    forbidden = {
        "demand_kg",
        "production_kg",
        "inventory_kg",
    }

    if forbidden.intersection(
        testing.time_varying_known_reals
    ):
        raise ValueError(
            "Future-observed variable leakage."
        )

    date_pairs = df[
        ["time_idx", "week_start_date"]
    ].drop_duplicates()

    if date_pairs["time_idx"].duplicated().any():
        raise ValueError(
            "One time index maps to multiple dates."
        )

    index_to_date = {
        int(row.time_idx): pd.Timestamp(
            row.week_start_date
        )
        for row in date_pairs.itertuples(index=False)
    }

    print("Training windows:", len(training))
    print("Validation windows:", len(validation))
    print("Reserved test windows:", len(testing))

    return (
        df,
        testing,
        manifests["test"],
        index_to_date,
    )


# ============================================================
# GENERATE MATCHED TEST FORECASTS
# ============================================================

def generate_test_predictions(
    model,
    testing,
    manifest,
    index_to_date,
):
    print("\nGENERATING FROZEN TEST PREDICTIONS")
    print("-" * 65)

    # Uses the already-validated Step 12
    # prediction routine, with the test
    # manifest and test dataset.
    frame, quantile_crossings = (
        evaluate_predictions(
            model,
            testing,
            manifest,
            index_to_date,
        )
    )

    if len(frame) != EXPECTED_PREDICTIONS:
        raise ValueError(
            f"Expected {EXPECTED_PREDICTIONS} "
            f"test prediction rows; "
            f"found {len(frame)}"
        )

    if frame["target_start"].duplicated().all():
        raise ValueError(
            "Unexpected prediction date structure."
        )

    print("Test prediction rows:", len(frame))
    print(
        "Quantile crossing observations:",
        quantile_crossings,
    )

    return frame, quantile_crossings


# ============================================================
# ADD 52-WEEK SEASONAL NAIVE BENCHMARK
# ============================================================

def add_seasonal_naive(
    frame,
    historical_data,
):
    print("\nCALCULATING SEASONAL NAIVE BASELINE")
    print("-" * 65)

    history = historical_data[
        ["grade", "week_start_date", "demand_kg"]
    ].copy()

    history["week_start_date"] = pd.to_datetime(
        history["week_start_date"]
    )

    if history.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate grade/week historical demand."
        )

    history = history.rename(
        columns={
            "week_start_date": (
                "seasonal_reference_week"
            ),
            "demand_kg": (
                "seasonal_naive_baseline_kg"
            ),
        }
    )

    frame = frame.copy()

    frame["target_week"] = pd.to_datetime(
        frame["target_week"]
    )

    frame["forecast_origin"] = pd.to_datetime(
        frame["forecast_origin"]
    )

    frame["seasonal_reference_week"] = (
        frame["target_week"]
        - pd.Timedelta(
            weeks=SEASONAL_LAG_WEEKS
        )
    )

    # A seasonal value is valid only if it was
    # observable before the forecast was issued.
    if (
        frame["seasonal_reference_week"]
        > frame["forecast_origin"]
    ).any():
        raise ValueError(
            "Seasonal-naive baseline uses future data."
        )

    original_rows = len(frame)

    frame = frame.merge(
        history,
        on=[
            "grade",
            "seasonal_reference_week",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(frame) != original_rows:
        raise ValueError(
            "Seasonal baseline merge changed row count."
        )

    if frame[
        "seasonal_naive_baseline_kg"
    ].isna().any():
        raise ValueError(
            "Missing 52-week seasonal baseline values."
        )

    frame["target_week"] = (
        frame["target_week"]
        .dt.strftime("%Y-%m-%d")
    )

    frame["forecast_origin"] = (
        frame["forecast_origin"]
        .dt.strftime("%Y-%m-%d")
    )

    frame["seasonal_reference_week"] = (
        frame["seasonal_reference_week"]
        .dt.strftime("%Y-%m-%d")
    )

    print(
        "Seasonal baseline predictions:",
        frame[
            "seasonal_naive_baseline_kg"
        ].notna().sum(),
    )

    return frame


# ============================================================
# CHECK TEST TARGETS AGAINST SOURCE OBSERVATIONS
# ============================================================

def verify_actual_targets(frame, historical_data):
    source = historical_data[
        ["grade", "week_start_date", "demand_kg"]
    ].copy()

    source["week_start_date"] = (
        pd.to_datetime(
            source["week_start_date"]
        ).dt.strftime("%Y-%m-%d")
    )

    source = source.rename(
        columns={
            "week_start_date": "target_week",
            "demand_kg": "source_actual_demand_kg",
        }
    )

    checked = frame.merge(
        source,
        on=["grade", "target_week"],
        how="left",
        validate="many_to_one",
    )

    if len(checked) != len(frame):
        raise ValueError(
            "Target verification changed row count."
        )

    if checked[
        "source_actual_demand_kg"
    ].isna().any():
        raise ValueError(
            "Unmatched test target observations."
        )

    if not np.allclose(
        checked[
            "actual_demand_kg"
        ].to_numpy(dtype=np.float64),
        checked[
            "source_actual_demand_kg"
        ].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.01,
    ):
        raise ValueError(
            "Predicted targets do not match "
            "the original test observations."
        )

    return checked.drop(
        columns=["source_actual_demand_kg"]
    )


# ============================================================
# CALCULATE FORECAST ACCURACY
# ============================================================

def metrics_for(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Nonfinite values in forecast metrics."
        )

    errors = actual - predicted

    mae = float(
        np.mean(np.abs(errors))
    )

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    denominator = float(
        np.sum(np.abs(actual))
    )

    wape = (
        float(
            100.0
            * np.sum(np.abs(errors))
            / denominator
        )
        if denominator > 0
        else None
    )

    return mae, rmse, wape


def summarize_group(frame):
    actual = frame[
        "actual_demand_kg"
    ].to_numpy(dtype=np.float64)

    result = {
        "prediction_rows": int(len(frame)),
    }

    for model_name, column in BASELINES.items():
        mae, rmse, wape = metrics_for(
            actual,
            frame[column],
        )

        result[f"{model_name}_mae_kg"] = mae
        result[f"{model_name}_rmse_kg"] = rmse
        result[f"{model_name}_wape_percent"] = wape

    coverage = float(
        100
        * frame[
            "within_80_percent_interval"
        ].mean()
    )

    result[
        "tft_interval_80_coverage_percent"
    ] = coverage

    naive_mae = result[
        "last_observation_mae_kg"
    ]

    seasonal_mae = result[
        "seasonal_naive_mae_kg"
    ]

    tft_mae = result[
        "tft_mae_kg"
    ]

    result[
        "tft_mae_improvement_over_last_percent"
    ] = (
        float(
            100.0
            * (naive_mae - tft_mae)
            / naive_mae
        )
        if naive_mae > 0
        else None
    )

    result[
        "tft_mae_improvement_over_seasonal_percent"
    ] = (
        float(
            100.0
            * (seasonal_mae - tft_mae)
            / seasonal_mae
        )
        if seasonal_mae > 0
        else None
    )

    return result


def create_metric_tables(frame):
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
        1,
        PREDICTION_LENGTH + 1,
    ):
        subset = frame[
            frame["horizon"] == horizon
        ]

        horizon_rows.append({
            "horizon": horizon,
            **summarize_group(subset),
        })

    detailed_rows = []

    for grade in GRADES:
        for horizon in range(
            1,
            PREDICTION_LENGTH + 1,
        ):
            subset = frame[
                (frame["grade"] == grade)
                & (frame["horizon"] == horizon)
            ]

            detailed_rows.append({
                "grade": grade,
                "horizon": horizon,
                **summarize_group(subset),
            })

    overall = summarize_group(frame)

    return (
        pd.DataFrame(grade_rows),
        pd.DataFrame(horizon_rows),
        pd.DataFrame(detailed_rows),
        overall,
    )


# ============================================================
# SAVE FINAL TEST OUTPUTS
# ============================================================

def save_outputs(
    predictions,
    grade_metrics,
    horizon_metrics,
    detailed_metrics,
    overall,
    crossing_count,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
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

    detailed_metrics.to_csv(
        DETAIL_FILE,
        index=False,
    )

    report = {
        "status": "PASS",
        "evaluation": "final_held_out_test",
        "partition": "2025_2026_test",
        "model": "no_price_tft",
        "checkpoint": str(CHECKPOINT),
        "target": "demand_kg",
        "target_definition": (
            "Total customer orders including "
            "unfulfilled orders"
        ),
        "grades": GRADES,
        "encoder_length": 52,
        "prediction_length": PREDICTION_LENGTH,
        "forecast_windows": int(
            len(predictions)
            // PREDICTION_LENGTH
        ),
        "prediction_rows": int(
            len(predictions)
        ),
        "quantile_crossing_observations": int(
            crossing_count
        ),
        "nominal_interval_coverage_percent": 80.0,
        "overall_metrics": overall,
        "grade_metrics": grade_metrics.to_dict(
            orient="records"
        ),
        "horizon_metrics": horizon_metrics.to_dict(
            orient="records"
        ),
        "model_selected_using": (
            "2024 validation only"
        ),
        "no_model_retraining": True,
        "training_data_modified": False,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "limitations": [
            (
                "Test windows are overlapping "
                "rolling-origin forecasts."
            ),
            (
                "Reported errors are not "
                "statistically independent."
            ),
            (
                "The no-price TFT does not "
                "use auction-price covariates."
            ),
            (
                "Seasonal-naive uses observed "
                "demand 52 weeks earlier."
            ),
            (
                "Forecast interval coverage "
                "is calculated from the "
                "10th and 90th quantiles."
            ),
            (
                "Test outcomes must not be "
                "used for iterative model selection."
            ),
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "PHASE 3 - STEP 16: FINAL TFT TEST EVALUATION"
    )

    print("=" * 65)

    (
        source_df,
        testing,
        test_manifest,
        index_to_date,
    ) = prepare_test_dataset()

    model = load_selected_model()

    predictions, crossing_count = (
        generate_test_predictions(
            model,
            testing,
            test_manifest,
            index_to_date,
        )
    )

    predictions = add_seasonal_naive(
        predictions,
        source_df,
    )

    predictions = verify_actual_targets(
        predictions,
        source_df,
    )

    (
        grade_metrics,
        horizon_metrics,
        detailed_metrics,
        overall,
    ) = create_metric_tables(
        predictions
    )

    save_outputs(
        predictions,
        grade_metrics,
        horizon_metrics,
        detailed_metrics,
        overall,
        crossing_count,
    )

    print("\nFINAL TEST ACCURACY")
    print("-" * 65)

    print(
        "Forecast windows:",
        len(test_manifest),
    )

    print(
        "Prediction rows:",
        len(predictions),
    )

    for name, value in overall.items():
        print(f"{name}: {value}")

    print("\nMETRICS BY TEA GRADE")

    print(
        grade_metrics[
            [
                "grade",
                "tft_mae_kg",
                "last_observation_mae_kg",
                "seasonal_naive_mae_kg",
                "tft_interval_80_coverage_percent",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nMETRICS BY FORECAST HORIZON")

    print(
        horizon_metrics[
            [
                "horizon",
                "tft_mae_kg",
                "last_observation_mae_kg",
                "seasonal_naive_mae_kg",
                "tft_interval_80_coverage_percent",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nOUTPUT FILES")

    print(PREDICTIONS_FILE)
    print(GRADE_FILE)
    print(HORIZON_FILE)
    print(DETAIL_FILE)
    print(SUMMARY_FILE)

    print("\nMODEL RETRAINING: NONE")
    print("FINAL HELD-OUT TEST EVALUATION: PASS")


if __name__ == "__main__":
    main()
