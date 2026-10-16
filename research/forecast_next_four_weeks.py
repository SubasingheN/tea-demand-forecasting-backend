
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

from train_tft_no_price import (
    prepare_no_price_data,
    create_no_price_datasets,
    GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
)


# ============================================================
# STEP 19 - FOUR-WEEK OPERATIONAL DEMAND FORECASTING
#
# CORRECTED CALENDAR ENCODING
#
# Week encoding:
#   ISO week number / 52.1775
#
# Month encoding:
#   Calendar month / 12
#
# Selected model:
#   Frozen no-price TFT, epoch 9
#
# Historical input:
#   52 completed weekly observations per grade
#
# Output:
#   4 future weeks x 5 tea grades = 20 forecasts
#
# No retraining.
# No future observed demand, production or inventory.
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
    / "tft_operational_forecasts"
)

CSV_FILE = (
    OUTPUT_DIR
    / "next_four_weeks_forecasts.csv"
)

JSON_FILE = (
    OUTPUT_DIR
    / "next_four_weeks_forecasts.json"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "forecast_generation_report.json"
)

EXPECTED_LATEST_WEEK = pd.Timestamp("2026-10-05")

WEEK_PERIOD = 52.1775
MONTH_PERIOD = 12.0

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

EXPECTED_FORECASTS = (
    len(GRADES) * PREDICTION_LENGTH
)


# ============================================================
# 1. CALENDAR FEATURE ENGINEERING
# ============================================================

def calculate_calendar_features(date):
    date = pd.Timestamp(date)

    iso_week = int(
        date.isocalendar().week
    )

    month = int(date.month)

    week_angle = (
        2.0
        * np.pi
        * iso_week
        / WEEK_PERIOD
    )

    month_angle = (
        2.0
        * np.pi
        * month
        / MONTH_PERIOD
    )

    return {
        "week_sin": float(
            np.sin(week_angle)
        ),
        "week_cos": float(
            np.cos(week_angle)
        ),
        "month_sin": float(
            np.sin(month_angle)
        ),
        "month_cos": float(
            np.cos(month_angle)
        ),
    }


def verify_calendar_encoding(df):
    print("\nVERIFYING ORIGINAL CALENDAR ENCODING")
    print("-" * 65)

    unique_weeks = (
        df.drop_duplicates(
            subset=["week_start_date"]
        )
        .sort_values("week_start_date")
    )

    maximum_error = 0.0

    for row in unique_weeks.itertuples(index=False):
        calculated = calculate_calendar_features(
            row.week_start_date
        )

        for column in [
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
        ]:
            actual = float(
                getattr(row, column)
            )

            expected = calculated[column]

            difference = abs(
                actual - expected
            )

            maximum_error = max(
                maximum_error,
                difference,
            )

    print("Calendar weeks checked:", len(unique_weeks))
    print("Week period:", WEEK_PERIOD)
    print("Month period:", MONTH_PERIOD)
    print("Maximum absolute error:", maximum_error)

    if maximum_error > 1e-4:
        raise ValueError(
            "Future calendar formula does not match "
            "the original historical features. "
            f"Maximum error: {maximum_error}"
        )

    print("ORIGINAL CALENDAR ENCODING: PASS")


# ============================================================
# 2. LOAD AND VALIDATE FACTORY HISTORY
# ============================================================

def load_historical_data():
    print("\nLOADING HISTORICAL FACTORY DATA")
    print("-" * 65)

    df = prepare_no_price_data()

    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        errors="raise",
    ).dt.normalize()

    df = df.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)

    if df.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate grade/week records."
        )

    latest_by_grade = df.groupby(
        "grade"
    )["week_start_date"].max()

    if set(latest_by_grade.index) != set(GRADES):
        raise ValueError(
            "Unexpected tea grades."
        )

    if latest_by_grade.nunique() != 1:
        raise ValueError(
            "Tea grades have different latest dates."
        )

    latest = pd.Timestamp(
        latest_by_grade.iloc[0]
    )

    if latest != EXPECTED_LATEST_WEEK:
        raise ValueError(
            f"Expected latest week "
            f"{EXPECTED_LATEST_WEEK.date()}, "
            f"found {latest.date()}."
        )

    last_indices = df.groupby(
        "grade"
    )["time_idx"].max()

    if last_indices.nunique() != 1:
        raise ValueError(
            "Latest time indices differ by grade."
        )

    latest_time_idx = int(
        last_indices.iloc[0]
    )

    for grade in GRADES:
        history = (
            df[df["grade"] == grade]
            .tail(ENCODER_LENGTH)
        )

        if len(history) != ENCODER_LENGTH:
            raise ValueError(
                f"Insufficient history for {grade}."
            )

        expected_dates = pd.date_range(
            end=latest,
            periods=ENCODER_LENGTH,
            freq="W-MON",
        )

        actual_dates = pd.DatetimeIndex(
            history["week_start_date"]
        )

        if not actual_dates.equals(expected_dates):
            raise ValueError(
                f"Missing or discontinuous weeks: {grade}"
            )

        indices = history[
            "time_idx"
        ].to_numpy(dtype=int)

        if not np.array_equal(
            np.diff(indices),
            np.ones(
                ENCODER_LENGTH - 1,
                dtype=int,
            ),
        ):
            raise ValueError(
                f"Nonconsecutive time indices: {grade}"
            )

        if int(indices[-1]) != latest_time_idx:
            raise ValueError(
                f"Latest time index mismatch: {grade}"
            )

    print("Latest observation:", latest.date())
    print("Latest time index:", latest_time_idx)
    print("Tea grades:", ", ".join(GRADES))
    print("Encoder weeks:", ENCODER_LENGTH)

    verify_calendar_encoding(df)

    return df, latest, latest_time_idx


# ============================================================
# 3. CONSTRUCT FUTURE CALENDAR ROWS
# ============================================================

def build_future_data(
    historical,
    latest,
    latest_time_idx,
):
    print("\nCREATING FUTURE CALENDAR FEATURES")
    print("-" * 65)

    rows = []

    for grade in GRADES:
        grade_history = historical[
            historical["grade"] == grade
        ].sort_values("week_start_date")

        last_observation = grade_history.iloc[-1]

        for horizon in range(
            1,
            PREDICTION_LENGTH + 1,
        ):
            forecast_date = (
                latest
                + pd.Timedelta(weeks=horizon)
            )

            calendar = calculate_calendar_features(
                forecast_date
            )

            # The TimeSeriesDataSet needs values
            # for every declared unknown real.
            #
            # These are technical placeholders.
            # They are NOT actual future values.
            #
            # The TFT decoder must not use them.
            row = {
                "week_start_date": forecast_date,
                "grade": grade,
                "time_idx": (
                    latest_time_idx + horizon
                ),
                "demand_kg": float(
                    last_observation["demand_kg"]
                ),
                "production_kg": float(
                    last_observation["production_kg"]
                ),
                "inventory_kg": float(
                    last_observation["inventory_kg"]
                ),
            }

            row.update(calendar)
            rows.append(row)

    future = pd.DataFrame(rows)

    if len(future) != EXPECTED_FORECASTS:
        raise ValueError(
            "Incorrect number of future records."
        )

    if future.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate future grade/week records."
        )

    print("Future records:", len(future))
    print(
        "First forecast week:",
        future["week_start_date"].min().date(),
    )
    print(
        "Last forecast week:",
        future["week_start_date"].max().date(),
    )

    print("\nFUTURE CALENDAR PREVIEW")

    preview = (
        future[
            future["grade"] == GRADES[0]
        ][
            [
                "week_start_date",
                "time_idx",
                "week_sin",
                "week_cos",
                "month_sin",
                "month_cos",
            ]
        ]
    )

    print(
        preview.round(5).to_string(
            index=False
        )
    )

    return future


# ============================================================
# 4. PREPARE TFT INFERENCE DATASET
# ============================================================

def prepare_inference_dataset(
    historical,
    future,
):
    print("\nPREPARING TFT INFERENCE DATASET")
    print("-" * 65)

    # Reuse the training dataset's fitted
    # categorical encoders and target scaler.
    #
    # Do not fit preprocessing transformations
    # on future data.
    training, _, _ = (
        create_no_price_datasets(historical)
    )

    history_context = (
        historical.groupby(
            "grade",
            group_keys=False,
        )
        .tail(ENCODER_LENGTH)
        .copy()
    )

    combined = pd.concat(
        [
            history_context,
            future,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    combined["demand_kg"] = (
        combined["demand_kg"].astype("float32")
    )

    inference = TimeSeriesDataSet.from_dataset(
        training,
        combined,
        predict=True,
        stop_randomization=True,
    )

    if len(inference) != len(GRADES):
        raise ValueError(
            "Expected five inference windows; "
            f"found {len(inference)}."
        )

    if set(
        inference.time_varying_known_reals
    ) != set(KNOWN_FUTURE_REALS):
        raise ValueError(
            "Unexpected known-future features."
        )

    if set(
        inference.time_varying_unknown_reals
    ) != set(HISTORICAL_ONLY_REALS):
        raise ValueError(
            "Unexpected historical-only features."
        )

    print("Inference windows:", len(inference))
    print("Decoder weeks:", PREDICTION_LENGTH)

    return inference


# ============================================================
# 5. LOAD FROZEN MODEL
# ============================================================

def load_selected_model():
    print("\nLOADING SELECTED TFT MODEL")
    print("-" * 65)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    # Only deserialize a locally generated,
    # trusted model checkpoint.
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
        weights_only=False,
    )

    model.eval()
    model.cpu()

    forbidden_decoder = (
        set(model.decoder_variables)
        & set(HISTORICAL_ONLY_REALS)
    )

    if forbidden_decoder:
        raise ValueError(
            "Historical-only features present "
            "in the TFT decoder: "
            f"{sorted(forbidden_decoder)}"
        )

    print("Model:", type(model).__name__)
    print("Checkpoint:", CHECKPOINT)
    print("Decoder leakage check: PASS")

    return model


# ============================================================
# 6. GENERATE QUANTILE FORECASTS
# ============================================================

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def run_forecasting(
    model,
    inference,
    future,
):
    print("\nGENERATING TFT DEMAND FORECASTS")
    print("-" * 65)

    loader = inference.to_dataloader(
        train=False,
        batch_size=len(GRADES),
        num_workers=0,
    )

    prediction = model.predict(
        loader,
        mode="quantiles",
        return_x=True,
        return_index=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "logger": False,
            "enable_progress_bar": True,
        },
    )

    values = to_numpy(
        prediction.output
    )

    decoder_indices = to_numpy(
        prediction.x["decoder_time_idx"]
    )

    indices = prediction.index.copy()

    expected_shape = (
        len(GRADES),
        PREDICTION_LENGTH,
        len(QUANTILES),
    )

    if values.shape != expected_shape:
        raise ValueError(
            "Unexpected quantile shape: "
            f"{values.shape}"
        )

    if decoder_indices.shape != (
        len(GRADES),
        PREDICTION_LENGTH,
    ):
        raise ValueError(
            "Unexpected decoder indices shape."
        )

    if "grade" not in indices.columns:
        raise ValueError(
            "Predicted grades are missing."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite TFT predictions."
        )

    lookup = {
        (
            str(row.grade),
            int(row.time_idx),
        ): pd.Timestamp(row.week_start_date)
        for row in future.itertuples(index=False)
    }

    rows = []

    for sample in range(len(GRADES)):
        grade = str(
            indices.iloc[sample]["grade"]
        )

        if grade not in GRADES:
            raise ValueError(
                f"Unexpected grade: {grade}"
            )

        for horizon in range(
            PREDICTION_LENGTH
        ):
            time_idx = int(
                decoder_indices[sample, horizon]
            )

            key = (
                grade,
                time_idx,
            )

            if key not in lookup:
                raise ValueError(
                    "Decoder prediction does not "
                    "match expected future dates: "
                    f"{key}"
                )

            q10, q50, q90 = (
                float(v)
                for v in values[sample, horizon]
            )

            if not (
                q10 <= q50 <= q90
            ):
                raise ValueError(
                    f"Quantile crossing: {key}"
                )

            if q10 < 0:
                raise ValueError(
                    f"Negative forecast: {key}"
                )

            rows.append({
                "grade": grade,
                "forecast_week": lookup[key].strftime(
                    "%Y-%m-%d"
                ),
                "forecast_horizon": horizon + 1,
                "lower_demand_kg": q10,
                "median_demand_kg": q50,
                "upper_demand_kg": q90,
            })

    result = pd.DataFrame(rows)

    if len(result) != EXPECTED_FORECASTS:
        raise ValueError(
            f"Expected {EXPECTED_FORECASTS} "
            f"forecasts, found {len(result)}."
        )

    if result.duplicated(
        ["grade", "forecast_week"]
    ).any():
        raise ValueError(
            "Duplicate grade/week forecasts."
        )

    if set(result["grade"]) != set(GRADES):
        raise ValueError(
            "Not all tea grades were forecast."
        )

    result = result.sort_values(
        [
            "grade",
            "forecast_horizon",
        ]
    ).reset_index(drop=True)

    print("Generated forecasts:", len(result))
    print("Forecast quantiles:", QUANTILES)
    print("Forecast validation: PASS")

    return result


# ============================================================
# 7. SAVE CSV AND JSON OUTPUTS
# ============================================================

def save_forecasts(
    forecasts,
    origin,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        CSV_FILE,
        index=False,
        float_format="%.4f",
    )

    grouped_forecasts = {}

    for grade in GRADES:
        subset = forecasts[
            forecasts["grade"] == grade
        ].sort_values("forecast_horizon")

        grouped_forecasts[grade] = [
            {
                "week": row.forecast_week,
                "horizon": int(
                    row.forecast_horizon
                ),
                "lower_demand_kg": float(
                    row.lower_demand_kg
                ),
                "median_demand_kg": float(
                    row.median_demand_kg
                ),
                "upper_demand_kg": float(
                    row.upper_demand_kg
                ),
            }
            for row in subset.itertuples(index=False)
        ]

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    payload = {
        "status": "PASS",
        "model": "no_price_tft",
        "forecast_type": "dataset_origin_forecast",
        "forecast_origin": origin.strftime(
            "%Y-%m-%d"
        ),
        "target": "demand_kg",
        "unit": "kg",
        "encoder_weeks": ENCODER_LENGTH,
        "forecast_horizon_weeks": (
            PREDICTION_LENGTH
        ),
        "quantiles": QUANTILES,
        "nominal_interval_coverage_percent": 80,
        "forecast_count": int(
            len(forecasts)
        ),
        "forecasts": grouped_forecasts,
        "checkpoint": str(CHECKPOINT),
        "generated_at_utc": generated_at,
        "notes": [
            "Forecast origin is the latest week "
            "available in the factory dataset.",
            "Forecasts are not automatically "
            "updated with live factory data.",
            "Future demand, production and "
            "inventory are not decoder inputs.",
            "Median predictions represent P50.",
            "Intervals are model-estimated "
            "P10 to P90 uncertainty ranges.",
        ],
    }

    JSON_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = {
        "status": "PASS",
        "step": 19,
        "model": "no_price_tft",
        "forecast_origin": origin.strftime(
            "%Y-%m-%d"
        ),
        "forecast_count": int(
            len(forecasts)
        ),
        "historical_input_weeks": (
            ENCODER_LENGTH
        ),
        "forecast_horizon_weeks": (
            PREDICTION_LENGTH
        ),
        "week_encoding_period": (
            WEEK_PERIOD
        ),
        "month_encoding_period": (
            MONTH_PERIOD
        ),
        "future_observed_inputs_used": False,
        "model_retraining": False,
        "outputs": {
            "csv": str(CSV_FILE),
            "json": str(JSON_FILE),
        },
        "generated_at_utc": generated_at,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print(
        "PHASE 3 - STEP 19: "
        "FOUR-WEEK OPERATIONAL DEMAND FORECAST"
    )
    print("=" * 65)

    historical, latest, latest_idx = (
        load_historical_data()
    )

    future = build_future_data(
        historical,
        latest,
        latest_idx,
    )

    inference = prepare_inference_dataset(
        historical,
        future,
    )

    model = load_selected_model()

    forecasts = run_forecasting(
        model,
        inference,
        future,
    )

    save_forecasts(
        forecasts,
        latest,
    )

    print("\nFOUR-WEEK TEA DEMAND FORECASTS")
    print("-" * 65)

    print(
        forecasts.round(2).to_string(
            index=False
        )
    )

    print("\nOUTPUT FILES")
    print(CSV_FILE)
    print(JSON_FILE)
    print(REPORT_FILE)

    print("\nMODEL RETRAINING: NONE")
    print("UNKNOWN FUTURE OBSERVATIONS USED: NO")
    print("FOUR-WEEK DEMAND FORECAST: PASS")


if __name__ == "__main__":
    main()
