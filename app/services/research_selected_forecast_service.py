
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

from app.schemas.research_weekly_input import (
    ResearchWeeklyBatchInput,
)

from app.services.research_selected_history_service import (
    prepare_selected_history,
)

from app.services.research_selected_history_service import (
    calculate_calendar_features,
)


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    ROOT
    / "research"
    / "outputs"
    / "tft_no_price"
    / "checkpoints"
    / "tft-no-price-epoch09-val470.2676.ckpt"
)

ENCODER_WEEKS = 52
FORECAST_WEEKS = 4

QUANTILES = [0.1, 0.5, 0.9]

HISTORICAL_FIELDS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
]

FUTURE_FIELDS = [
    "time_idx",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
]

ALL_GRADES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]


def load_frozen_tft():
    """
    Load the previously validated research
    checkpoint without retraining.
    """
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"TFT checkpoint missing: {CHECKPOINT}"
        )

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

    parameters = model.dataset_parameters

    if parameters is None:
        raise ValueError(
            "Checkpoint dataset parameters missing."
        )

    if parameters.get(
        "max_encoder_length"
    ) != ENCODER_WEEKS:
        raise ValueError(
            "Unexpected model encoder length."
        )

    if parameters.get(
        "max_prediction_length"
    ) != FORECAST_WEEKS:
        raise ValueError(
            "Unexpected model prediction length."
        )

    if parameters.get("group_ids") != ["grade"]:
        raise ValueError(
            "Unexpected model grouping."
        )

    if set(
        parameters.get(
            "time_varying_known_reals", []
        )
    ) != set(FUTURE_FIELDS):
        raise ValueError(
            "Unexpected future-known variables."
        )

    if set(
        parameters.get(
            "time_varying_unknown_reals", []
        )
    ) != set(HISTORICAL_FIELDS):
        raise ValueError(
            "Unexpected historical variables."
        )

    if set(model.decoder_variables) & set(
        HISTORICAL_FIELDS
    ):
        raise ValueError(
            "Historical-only variables found "
            "in the decoder."
        )

    return model


def build_selected_future_rows(
    history,
    selected_grades,
):
    """
    Generate calendar rows for the next
    four forecast weeks.

    Historical-only quantities use technical
    placeholders, never future observations.
    """
    future_rows = []

    for grade in selected_grades:
        grade_history = (
            history[history["grade"] == grade]
            .sort_values("time_idx")
        )

        if len(grade_history) != ENCODER_WEEKS:
            raise ValueError(
                f"{grade}: invalid encoder history."
            )

        latest = grade_history.iloc[-1]

        latest_date = pd.Timestamp(
            latest["week_start_date"]
        )

        latest_idx = int(
            latest["time_idx"]
        )

        for horizon in range(
            1,
            FORECAST_WEEKS + 1,
        ):
            future_date = (
                latest_date
                + timedelta(weeks=horizon)
            )

            calendar = (
                calculate_calendar_features(
                    future_date
                )
            )

            future_rows.append({
                "week_start_date": future_date,
                "grade": grade,
                "time_idx": (
                    latest_idx + horizon
                ),
                **calendar,

                # Required to construct the
                # TimeSeriesDataSet. These are
                # NOT real future observations.
                "demand_kg": float(
                    latest["demand_kg"]
                ),
                "production_kg": float(
                    latest["production_kg"]
                ),
                "inventory_kg": float(
                    latest["inventory_kg"]
                ),
            })

    return pd.DataFrame(future_rows)


def build_selected_inference_dataset(
    model,
    history,
    future,
    selected_grades,
):
    """
    Reuse the checkpoint's existing
    categorical encoders and normalizers.
    """
    combined = pd.concat(
        [history, future],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    combined["demand_kg"] = (
        combined["demand_kg"]
        .astype("float32")
    )

    for grade in selected_grades:
        rows = combined[
            combined["grade"] == grade
        ]

        if len(rows) != (
            ENCODER_WEEKS + FORECAST_WEEKS
        ):
            raise ValueError(
                f"{grade}: incorrect inference "
                "window length."
            )

        indices = rows[
            "time_idx"
        ].to_numpy(dtype=np.int64)

        if not np.all(
            np.diff(indices) == 1
        ):
            raise ValueError(
                f"{grade}: discontinuous "
                "inference time indices."
            )

    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters,
        combined,
        predict=True,
        stop_randomization=True,
    )

    if len(dataset) != len(selected_grades):
        raise ValueError(
            "Expected one prediction window "
            "per selected grade."
        )

    return dataset


def generate_selected_tft_forecast(
    batch_input: ResearchWeeklyBatchInput,
):
    """
    Generate fresh four-week P10/P50/P90
    forecasts for selected tea grades.

    No retraining or file/database writes.
    """
    prepared = prepare_selected_history(
        batch_input
    )

    history = prepared[
        "historical_dataframe"
    ]

    selected_grades = prepared[
        "selected_grades"
    ]

    model = load_frozen_tft()

    future = build_selected_future_rows(
        history,
        selected_grades,
    )

    dataset = (
        build_selected_inference_dataset(
            model,
            history,
            future,
            selected_grades,
        )
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
            },
        )

    output = prediction.output

    if isinstance(output, torch.Tensor):
        values = (
            output.detach()
            .cpu()
            .numpy()
        )
    else:
        values = np.asarray(output)

    expected_shape = (
        len(selected_grades),
        FORECAST_WEEKS,
        len(QUANTILES),
    )

    if values.shape != expected_shape:
        raise ValueError(
            "Unexpected forecast output shape: "
            f"{values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite TFT predictions."
        )

    decoder_time_idx = (
        prediction.x["decoder_time_idx"]
        .detach()
        .cpu()
        .numpy()
    )

    prediction_index = prediction.index

    if decoder_time_idx.shape != (
        len(selected_grades),
        FORECAST_WEEKS,
    ):
        raise ValueError(
            "Unexpected decoder time indices."
        )

    lookup = {
        (
            str(row.grade),
            int(row.time_idx),
        ): pd.Timestamp(
            row.week_start_date
        )
        for row in future.itertuples(
            index=False
        )
    }

    forecasts = {}

    for sample in range(
        len(prediction_index)
    ):
        grade = str(
            prediction_index.iloc[sample][
                "grade"
            ]
        )

        if grade not in selected_grades:
            raise ValueError(
                f"Unexpected predicted grade: {grade}"
            )

        if grade in forecasts:
            raise ValueError(
                f"Duplicate prediction: {grade}"
            )

        weekly = []

        for position in range(
            FORECAST_WEEKS
        ):
            idx = int(
                decoder_time_idx[
                    sample, position
                ]
            )

            key = (grade, idx)

            if key not in lookup:
                raise ValueError(
                    "Prediction does not match "
                    f"future calendar: {key}"
                )

            q10, q50, q90 = [
                float(value)
                for value in values[
                    sample, position
                ]
            ]

            if not (
                0 <= q10 <= q50 <= q90
            ):
                raise ValueError(
                    "Invalid forecast interval: "
                    f"{grade}, {idx}"
                )

            week_date = lookup[
                key
            ]

            weekly.append({
                "week": (
                    week_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "horizon": position + 1,
                "lower_demand_kg": round(
                    q10, 2
                ),
                "median_demand_kg": round(
                    q50, 2
                ),
                "upper_demand_kg": round(
                    q90, 2
                ),
            })

        forecasts[grade] = weekly

    if set(forecasts) != set(
        selected_grades
    ):
        raise ValueError(
            "Not all selected grades "
            "received predictions."
        )

    return {
        "status": "success",
        "model": "no_price_tft",
        "forecast_type": (
            "selected_operational_forecast"
        ),
        "forecast_origin": prepared[
            "forecast_origin"
        ],
        "forecast_start_date": prepared[
            "forecast_start_date"
        ],
        "forecast_end_date": prepared[
            "forecast_end_date"
        ],
        "historical_input_weeks": (
            ENCODER_WEEKS
        ),
        "forecast_horizon_weeks": (
            FORECAST_WEEKS
        ),
        "selected_grades": selected_grades,
        "tea_grade_count": len(
            selected_grades
        ),
        "forecast_count": (
            len(selected_grades)
            * FORECAST_WEEKS
        ),
        "target": "demand_kg",
        "unit": "kg",
        "quantiles": {
            "lower": 0.1,
            "median": 0.5,
            "upper": 0.9,
        },
        "forecasts": forecasts,
        "saved": False,
        "message": (
            "Selective operational TFT "
            "forecast generated without "
            "retraining or modifying "
            "research artifacts."
        ),
    }
