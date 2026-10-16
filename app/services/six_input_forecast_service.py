
"""
Six-input TFT inference service.

Produces four-week P10/P50/P90 demand forecasts
from the frozen six-input TFT checkpoint.

Supports:
- Historical replay
- Latest completed factory week
- One to five tea grades
- Six historical model inputs
- Four future calendar weeks

Does not train models, save forecasts,
or modify research datasets/checkpoints.
"""

from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

from app.schemas.six_input_weekly import (
    SixInputWeeklyBatch,
)

from app.services.six_input_history_service import (
    prepare_six_input_history,
)


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from tft_six_input_config import (
    CHECKPOINT_DIR,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    HISTORICAL_ONLY_REALS,
    KNOWN_FUTURE_REALS,
)


CHECKPOINT = (
    CHECKPOINT_DIR
    / "six-input-epoch08-val303.8794.ckpt"
)


def load_six_input_tft():
    """
    Load and verify the frozen six-input TFT.
    """

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            str(CHECKPOINT),
            map_location="cpu",
        )
    )

    model.eval()
    model.cpu()

    parameters = model.dataset_parameters

    if parameters is None:
        raise ValueError(
            "Checkpoint dataset parameters missing."
        )

    required_parameters = {
        "max_encoder_length": ENCODER_LENGTH,
        "max_prediction_length": PREDICTION_LENGTH,
        "group_ids": ["grade"],
    }

    for name, expected in required_parameters.items():
        if parameters.get(name) != expected:
            raise ValueError(
                f"Checkpoint configuration mismatch: {name}"
            )

    if set(
        parameters.get("time_varying_known_reals", [])
    ) != set(KNOWN_FUTURE_REALS):
        raise ValueError(
            "Future-known features do not match training."
        )

    if set(
        parameters.get("time_varying_unknown_reals", [])
    ) != set(HISTORICAL_ONLY_REALS):
        raise ValueError(
            "Historical inputs do not match training."
        )

    if set(model.decoder_variables).intersection(
        HISTORICAL_ONLY_REALS
    ):
        raise ValueError(
            "Historical-only inputs found in decoder."
        )

    learned_quantiles = getattr(
        model.loss,
        "quantiles",
        None,
    )

    if (
        learned_quantiles is None
        or not np.allclose(
            np.asarray(learned_quantiles, dtype=float),
            np.asarray(QUANTILES, dtype=float),
        )
    ):
        raise ValueError(
            "Checkpoint quantile configuration mismatch."
        )

    return model


def calculate_future_calendar(week_date):
    """
    Match original six-input TFT preprocessing.
    """

    timestamp = pd.Timestamp(week_date)

    iso_week = int(timestamp.isocalendar().week)
    month = int(timestamp.month)

    return {
        "week_sin": float(
            np.sin(2.0 * np.pi * iso_week / 52.1775)
        ),
        "week_cos": float(
            np.cos(2.0 * np.pi * iso_week / 52.1775)
        ),
        "month_sin": float(
            np.sin(2.0 * np.pi * month / 12.0)
        ),
        "month_cos": float(
            np.cos(2.0 * np.pi * month / 12.0)
        ),
    }


def build_six_input_future(
    history,
    selected_grades,
):
    """
    Build the four future calendar rows.

    Placeholder values in historical-only fields
    exist solely for dataset construction.
    They are not future observations and are not
    used as future-known decoder inputs.
    """

    rows = []

    for grade in selected_grades:
        group = history.loc[
            history["grade"] == grade
        ].sort_values("time_idx")

        if len(group) != ENCODER_LENGTH:
            raise ValueError(
                f"{grade}: invalid encoder length."
            )

        last = group.iloc[-1]

        last_date = pd.Timestamp(
            last["week_start_date"]
        )

        last_idx = int(last["time_idx"])

        for horizon in range(
            1,
            PREDICTION_LENGTH + 1,
        ):
            future_date = (
                last_date
                + timedelta(weeks=horizon)
            )

            row = {
                "week_start_date": future_date,
                "grade": grade,
                "time_idx": last_idx + horizon,
                **calculate_future_calendar(
                    future_date
                ),
            }

            for field in HISTORICAL_ONLY_REALS:
                row[field] = float(last[field])

            rows.append(row)

    return pd.DataFrame(rows)


def build_six_input_inference_dataset(
    model,
    history,
    future,
    selected_grades,
):
    """
    Reuse checkpoint-fitted dataset parameters,
    categorical encoders, and normalizers.
    """

    combined = pd.concat(
        [history, future],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    numeric_fields = list(
        set(
            HISTORICAL_ONLY_REALS
            + KNOWN_FUTURE_REALS
        )
        - {"time_idx"}
    )

    for field in numeric_fields:
        combined[field] = (
            combined[field].astype("float32")
        )

    for grade in selected_grades:
        group = combined.loc[
            combined["grade"] == grade
        ].sort_values("time_idx")

        expected_rows = (
            ENCODER_LENGTH + PREDICTION_LENGTH
        )

        if len(group) != expected_rows:
            raise ValueError(
                f"{grade}: expected {expected_rows} rows."
            )

        indices = group[
            "time_idx"
        ].to_numpy(dtype=np.int64)

        if not np.all(np.diff(indices) == 1):
            raise ValueError(
                f"{grade}: nonconsecutive time indices."
            )

    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters,
        combined,
        predict=True,
        stop_randomization=True,
    )

    if len(dataset) != len(selected_grades):
        raise ValueError(
            "Incorrect number of prediction windows."
        )

    return dataset


def generate_six_input_forecast(
    batch_input: SixInputWeeklyBatch,
):
    """
    Generate four-week P10/P50/P90 forecasts
    for the selected tea grades.
    """

    prepared = prepare_six_input_history(
        batch_input
    )

    history = prepared["historical_dataframe"]
    selected_grades = prepared["selected_grades"]
    forecast_mode = prepared["forecast_mode"]

    model = load_six_input_tft()

    future = build_six_input_future(
        history,
        selected_grades,
    )

    dataset = build_six_input_inference_dataset(
        model,
        history,
        future,
        selected_grades,
    )

    loader = dataset.to_dataloader(
        train=False,
        batch_size=len(selected_grades),
        num_workers=0,
    )

    with torch.inference_mode():
        prediction = model.predict(
            loader,
            mode="quantiles",
            return_index=True,
            return_x=True,
            trainer_kwargs={
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )

    output = prediction.output

    if isinstance(output, torch.Tensor):
        values = (
            output.detach().cpu().numpy()
        )
    else:
        values = np.asarray(output)

    expected_shape = (
        len(selected_grades),
        PREDICTION_LENGTH,
        len(QUANTILES),
    )

    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected prediction shape: {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite model predictions."
        )

    decoder_indices = (
        prediction.x["decoder_time_idx"]
        .detach().cpu().numpy()
    )

    if decoder_indices.shape != (
        len(selected_grades),
        PREDICTION_LENGTH,
    ):
        raise ValueError(
            "Unexpected decoder time indices."
        )

    prediction_index = prediction.index

    if len(prediction_index) != len(selected_grades):
        raise ValueError(
            "Incorrect prediction index length."
        )

    calendar_lookup = {
        (
            str(row.grade),
            int(row.time_idx),
        ): pd.Timestamp(row.week_start_date)
        for row in future.itertuples(index=False)
    }

    forecasts = {}

    for sample in range(len(prediction_index)):
        grade = str(
            prediction_index.iloc[sample]["grade"]
        )

        if grade not in selected_grades:
            raise ValueError(
                f"Unexpected predicted grade: {grade}"
            )

        if grade in forecasts:
            raise ValueError(
                f"Duplicate forecast for {grade}"
            )

        weekly = []

        for position in range(PREDICTION_LENGTH):
            time_idx = int(
                decoder_indices[sample, position]
            )

            key = (grade, time_idx)

            if key not in calendar_lookup:
                raise ValueError(
                    f"Forecast calendar mismatch: {key}"
                )

            p10, p50, p90 = [
                float(number)
                for number in values[
                    sample, position
                ]
            ]

            if not (0 <= p10 <= p50 <= p90):
                raise ValueError(
                    f"Invalid forecast interval: {key}"
                )

            week_date = calendar_lookup[key]

            weekly.append({
                "week": week_date.strftime(
                    "%Y-%m-%d"
                ),
                "horizon": position + 1,
                "lower_demand_kg": round(p10, 2),
                "median_demand_kg": round(p50, 2),
                "upper_demand_kg": round(p90, 2),
            })

        forecasts[grade] = weekly

    if set(forecasts) != set(selected_grades):
        raise ValueError(
            "Missing selected-grade predictions."
        )

    if forecast_mode == "latest_completed_week_forecast":
        message = (
            "Forecast generated using the latest "
            "completed week in the factory dataset. "
            "Current-period accuracy has not been "
            "independently verified."
        )
    else:
        message = (
            "Historical-replay forecast generated "
            "using the frozen six-input TFT."
        )

    return {
        "status": "success",
        "model": "six_input_tft",
        "forecast_type": forecast_mode,
        "forecast_origin": prepared[
            "forecast_origin"
        ],
        "forecast_start_date": prepared[
            "forecast_start_date"
        ],
        "forecast_end_date": prepared[
            "forecast_end_date"
        ],
        "dataset_latest_week": prepared[
            "dataset_latest_week"
        ],
        "historical_input_weeks": ENCODER_LENGTH,
        "forecast_horizon_weeks": PREDICTION_LENGTH,
        "selected_grades": selected_grades,
        "tea_grade_count": len(selected_grades),
        "forecast_count": (
            len(selected_grades) * PREDICTION_LENGTH
        ),
        "target": "demand_kg",
        "unit": "kg",
        "quantiles": {
            "lower": 0.1,
            "median": 0.5,
            "upper": 0.9,
        },
        "input_features": list(
            HISTORICAL_ONLY_REALS
        ),
        "forecasts": forecasts,
        "saved": False,
        "message": message,
    }
