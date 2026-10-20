
"""
Production nine-input tea demand forecasting service.

Uses the frozen production Temporal Fusion Transformer
and the preprocessing parameters fitted on the approved
historical dataset.

Historical inputs:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_monday_rate
    usd_lkr_sunday_rate
    festival_flag
    auction_price_high_lkr
    auction_price_medium_lkr
    auction_price_low_lkr

Forecast output:
    Four consecutive future weeks.
    P10, P50 and P90 demand estimates in kg.

Forecasting modes:
    1. Normal saved-data forecasting.
    2. Latest-week edited-input preview forecasting.

The preview:
    - Uses the submitted latest-week inputs in memory.
    - Does not update MongoDB.
    - Does not save or finalize forecasts.
    - Does not retrain the model.

The original public function name
generate_eight_input_forecast() is preserved.

No database writes or model training occur here.
"""

import json
import sys

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_history_service import (
    prepare_eight_input_history,
    prepare_eight_input_history_preview,
)


ROOT = Path(__file__).resolve().parents[2]

RESEARCH_DIR = ROOT / "research"

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from tft_eight_input_config import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    TEA_GRADES,
    HISTORICAL_ONLY_REALS,
    KNOWN_FUTURE_REALS,
)


MODEL_NAME = "nine_input_tft"
MODEL_VERSION = "production_v1"

PRODUCTION_DIR = (
    RESEARCH_DIR
    / "outputs"
    / "tft_nine_input"
    / "production"
)

CHECKPOINT = (
    PRODUCTION_DIR
    / "checkpoints"
    / "nine_input_tft_production.ckpt"
)

DATASET_PARAMETERS_FILE = (
    PRODUCTION_DIR
    / "dataset_parameters.pt"
)

PRODUCTION_METADATA_FILE = (
    PRODUCTION_DIR
    / "production_metadata.json"
)

EXPECTED_FEATURES = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
    "festival_flag",
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]

EXPECTED_QUANTILES = [0.1, 0.5, 0.9]


def validate_production_metadata(metadata):
    """
    Validate the identity and configuration
    of the production model artifacts.
    """

    if metadata.get("model") != MODEL_NAME:
        raise ValueError(
            "Production metadata model identity mismatch."
        )

    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Production model version mismatch."
        )

    if metadata.get("status") not in {
        "trained_not_deployed",
        "deployed",
    }:
        raise ValueError(
            "Production metadata has an invalid status."
        )

    forecasting = metadata.get("forecasting", {})

    if forecasting.get("encoder_weeks") != ENCODER_LENGTH:
        raise ValueError(
            "Production encoder length mismatch."
        )

    if forecasting.get("forecast_weeks") != PREDICTION_LENGTH:
        raise ValueError(
            "Production forecast horizon mismatch."
        )

    if forecasting.get("target") != "demand_kg":
        raise ValueError(
            "Production target mismatch."
        )

    if forecasting.get("historical_only_features") != (
        EXPECTED_FEATURES
    ):
        raise ValueError(
            "Production historical input features mismatch."
        )

    if forecasting.get("known_future_features") != (
        list(KNOWN_FUTURE_REALS)
    ):
        raise ValueError(
            "Production known-future features mismatch."
        )

    if forecasting.get("group_ids") != ["grade"]:
        raise ValueError(
            "Production grouping configuration mismatch."
        )

    if not np.allclose(
        forecasting.get("quantiles", []),
        EXPECTED_QUANTILES,
    ):
        raise ValueError(
            "Production quantile configuration mismatch."
        )

    if list(HISTORICAL_ONLY_REALS) != EXPECTED_FEATURES:
        raise ValueError(
            "Current backend feature configuration "
            "differs from the trained production model."
        )

    if list(QUANTILES) != EXPECTED_QUANTILES:
        raise ValueError(
            "Current backend quantiles do not match "
            "the trained production model."
        )

    artifact_paths = metadata.get("artifacts", {})

    if not artifact_paths.get("checkpoint"):
        raise ValueError(
            "Production checkpoint metadata is missing."
        )

    if not artifact_paths.get("dataset_parameters"):
        raise ValueError(
            "Production preprocessing metadata is missing."
        )

    if (
        Path(artifact_paths["checkpoint"]).resolve()
        != CHECKPOINT.resolve()
    ):
        raise ValueError(
            "Production checkpoint path mismatch."
        )

    if (
        Path(
            artifact_paths["dataset_parameters"]
        ).resolve()
        != DATASET_PARAMETERS_FILE.resolve()
    ):
        raise ValueError(
            "Production preprocessing path mismatch."
        )


def validate_saved_parameters(parameters):
    """
    Verify the saved TFT dataset configuration.
    """

    if parameters.get("max_encoder_length") != (
        ENCODER_LENGTH
    ):
        raise ValueError(
            "Saved encoder length mismatch."
        )

    if parameters.get("max_prediction_length") != (
        PREDICTION_LENGTH
    ):
        raise ValueError(
            "Saved prediction length mismatch."
        )

    if parameters.get("group_ids") != ["grade"]:
        raise ValueError(
            "Saved grouping configuration mismatch."
        )

    if parameters.get("target") != "demand_kg":
        raise ValueError(
            "Saved target configuration mismatch."
        )

    if parameters.get("time_varying_unknown_reals") != (
        EXPECTED_FEATURES
    ):
        raise ValueError(
            "Saved historical feature configuration mismatch."
        )

    if parameters.get("time_varying_known_reals") != (
        list(KNOWN_FUTURE_REALS)
    ):
        raise ValueError(
            "Saved known-future feature configuration mismatch."
        )


@lru_cache(maxsize=1)
def load_eight_input_tft():
    """
    Load the frozen production model and
    its trusted preprocessing parameters.
    """

    for path in [
        CHECKPOINT,
        DATASET_PARAMETERS_FILE,
        PRODUCTION_METADATA_FILE,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Production model artifact missing: {path}"
            )

    metadata = json.loads(
        PRODUCTION_METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    validate_production_metadata(metadata)

    # Load only trusted artifacts generated by
    # this project's model-training pipeline.
    parameters = torch.load(
        str(DATASET_PARAMETERS_FILE),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(parameters, dict):
        raise ValueError(
            "Invalid production preprocessing parameters."
        )

    validate_saved_parameters(parameters)

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
    )

    model.cpu()
    model.eval()

    checkpoint_parameters = model.dataset_parameters

    if checkpoint_parameters is None:
        raise ValueError(
            "Checkpoint dataset parameters are missing."
        )

    validate_saved_parameters(checkpoint_parameters)

    if set(model.decoder_variables).intersection(
        EXPECTED_FEATURES
    ):
        raise ValueError(
            "Historical-only features are incorrectly "
            "included in decoder inputs."
        )

    learned_quantiles = getattr(
        model.loss,
        "quantiles",
        None,
    )

    if learned_quantiles is None or not np.allclose(
        np.asarray(learned_quantiles, dtype=float),
        EXPECTED_QUANTILES,
    ):
        raise ValueError(
            "Production checkpoint quantiles mismatch."
        )

    return model, parameters


def calculate_future_calendar(week_date):
    """
    Reproduce the calendar-feature formulas
    used during model training.
    """

    timestamp = pd.Timestamp(week_date)

    iso_week = int(
        timestamp.isocalendar().week
    )

    month = int(timestamp.month)

    return {
        "week_sin": float(
            np.sin(
                2.0 * np.pi * iso_week / 52.1775
            )
        ),
        "week_cos": float(
            np.cos(
                2.0 * np.pi * iso_week / 52.1775
            )
        ),
        "month_sin": float(
            np.sin(
                2.0 * np.pi * month / 12.0
            )
        ),
        "month_cos": float(
            np.cos(
                2.0 * np.pi * month / 12.0
            )
        ),
    }


def validate_prepared_history(history, selected_grades):
    """
    Validate the 52 consecutive observed weeks
    supplied by the history preparation service.
    """

    if not isinstance(history, pd.DataFrame):
        raise ValueError(
            "Historical observations must be a DataFrame."
        )

    if not selected_grades:
        raise ValueError(
            "At least one tea grade is required."
        )

    if len(selected_grades) != len(set(selected_grades)):
        raise ValueError(
            "Duplicate tea grades were requested."
        )

    if not set(selected_grades).issubset(set(TEA_GRADES)):
        raise ValueError(
            "An unsupported tea grade was requested."
        )

    required_columns = {
        "week_start_date",
        "grade",
        "time_idx",
        *EXPECTED_FEATURES,
        *KNOWN_FUTURE_REALS,
    }

    missing_columns = sorted(
        required_columns - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "History preparation must provide the "
            "nine-input production features. "
            f"Missing: {missing_columns}"
        )

    if history.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Duplicate historical grade/week records."
        )

    reference_dates = None
    reference_indices = None

    for grade in selected_grades:
        group = (
            history.loc[
                history["grade"] == grade
            ]
            .sort_values("time_idx")
            .copy()
        )

        if len(group) != ENCODER_LENGTH:
            raise ValueError(
                f"{grade}: expected {ENCODER_LENGTH} "
                "historical weeks."
            )

        dates = pd.to_datetime(
            group["week_start_date"]
        )

        if not (dates.dt.dayofweek == 0).all():
            raise ValueError(
                f"{grade}: reporting weeks must start Monday."
            )

        date_values = dates.to_numpy(
            dtype="datetime64[D]"
        )

        date_differences = np.diff(
            date_values
        ).astype("timedelta64[D]").astype(int)

        if not np.all(date_differences == 7):
            raise ValueError(
                f"{grade}: historical dates are not consecutive."
            )

        indices = group["time_idx"].to_numpy(
            dtype=np.int64
        )

        if not np.all(np.diff(indices) == 1):
            raise ValueError(
                f"{grade}: historical time indices "
                "are not consecutive."
            )

        for feature in EXPECTED_FEATURES:
            values = pd.to_numeric(
                group[feature],
                errors="raise",
            ).to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{grade}: invalid values in {feature}."
                )

            if feature in {
                "demand_kg",
                "usd_lkr_monday_rate",
                "usd_lkr_sunday_rate",
                "auction_price_high_lkr",
                "auction_price_medium_lkr",
                "auction_price_low_lkr",
            } and (values <= 0).any():
                raise ValueError(
                    f"{grade}: {feature} must be positive."
                )

            if feature in {
                "production_kg",
                "inventory_kg",
            } and (values < 0).any():
                raise ValueError(
                    f"{grade}: {feature} cannot be negative."
                )

            if feature == "festival_flag":
                if not np.isin(values, [0, 1]).all():
                    raise ValueError(
                        f"{grade}: festival_flag must be 0 or 1."
                    )

        if reference_dates is None:
            reference_dates = date_values
            reference_indices = indices
        else:
            if not np.array_equal(
                reference_dates,
                date_values,
            ):
                raise ValueError(
                    "Historical reporting dates differ "
                    "between selected tea grades."
                )

            if not np.array_equal(
                reference_indices,
                indices,
            ):
                raise ValueError(
                    "Historical time indices differ "
                    "between selected tea grades."
                )


def build_eight_input_future(
    history,
    selected_grades,
):
    """
    Construct the next four calendar weeks.

    Historical-only fields receive structural
    placeholders because TimeSeriesDataSet
    requires complete rows.

    Those fields are not decoder inputs.
    """

    rows = []

    for grade in selected_grades:
        group = (
            history.loc[
                history["grade"] == grade
            ]
            .sort_values("time_idx")
        )

        last = group.iloc[-1]

        last_date = pd.Timestamp(
            last["week_start_date"]
        )

        last_idx = int(
            last["time_idx"]
        )

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

            for feature in EXPECTED_FEATURES:
                row[feature] = float(
                    last[feature]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_eight_input_inference_dataset(
    parameters,
    history,
    future,
    selected_grades,
):
    """
    Build inference windows using the fitted
    production encoders and scalers.
    """

    combined = pd.concat(
        [history, future],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    numeric_fields = (
        set(EXPECTED_FEATURES)
        | set(KNOWN_FUTURE_REALS)
    ) - {"time_idx"}

    for field in numeric_fields:
        combined[field] = pd.to_numeric(
            combined[field],
            errors="raise",
        ).astype("float32")

        if not np.isfinite(
            combined[field].to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                f"Invalid inference feature: {field}"
            )

    combined["grade"] = combined[
        "grade"
    ].astype(str)

    for grade in selected_grades:
        group = combined.loc[
            combined["grade"] == grade
        ].sort_values("time_idx")

        expected_rows = (
            ENCODER_LENGTH + PREDICTION_LENGTH
        )

        if len(group) != expected_rows:
            raise ValueError(
                f"{grade}: expected {expected_rows} "
                "inference rows."
            )

        indices = group["time_idx"].to_numpy(
            dtype=np.int64
        )

        if not np.all(np.diff(indices) == 1):
            raise ValueError(
                f"{grade}: nonconsecutive inference indices."
            )

    dataset = TimeSeriesDataSet.from_parameters(
        parameters,
        combined,
        predict=True,
        stop_randomization=True,
    )

    if len(dataset) != len(selected_grades):
        raise ValueError(
            "Incorrect number of forecasting windows."
        )

    return dataset


def generate_eight_input_forecast(
    batch_input: EightInputWeeklyBatch,
    *,
    preview_latest_week=False,
):
    """
    Generate four-week forecasts using the
    production nine-input TFT model.

    preview_latest_week=False:
        Normal strict saved-data forecasting.

    preview_latest_week=True:
        Forecast using edited latest-week inputs
        without writing those inputs to MongoDB.

    Neither mode writes forecasts to MongoDB.
    """

    if not isinstance(
        preview_latest_week,
        bool,
    ):
        raise ValueError(
            "preview_latest_week must be a boolean."
        )

    if preview_latest_week:
        prepared = (
            prepare_eight_input_history_preview(
                batch_input
            )
        )
    else:
        prepared = prepare_eight_input_history(
            batch_input
        )

    history = prepared[
        "historical_dataframe"
    ]

    selected_grades = prepared[
        "selected_grades"
    ]

    forecast_mode = prepared[
        "forecast_mode"
    ]

    validate_prepared_history(
        history,
        selected_grades,
    )

    model, parameters = load_eight_input_tft()

    future = build_eight_input_future(
        history,
        selected_grades,
    )

    dataset = build_eight_input_inference_dataset(
        parameters,
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
            output.detach()
            .cpu()
            .numpy()
        )
    else:
        values = np.asarray(
            output,
            dtype=float,
        )

    expected_shape = (
        len(selected_grades),
        PREDICTION_LENGTH,
        len(QUANTILES),
    )

    if values.shape != expected_shape:
        raise ValueError(
            "Unexpected production TFT output shape: "
            f"{values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Production model generated "
            "non-finite forecasts."
        )

    # Quantile predictions may cross.
    # Sorting matches the existing evaluation policy.
    values = np.sort(
        values,
        axis=-1,
    )

    values = np.maximum(
        values,
        0.0,
    )

    decoder_indices = (
        prediction.x["decoder_time_idx"]
        .detach()
        .cpu()
        .numpy()
    )

    if decoder_indices.shape != (
        len(selected_grades),
        PREDICTION_LENGTH,
    ):
        raise ValueError(
            "Unexpected decoder time-index dimensions."
        )

    prediction_index = prediction.index

    if len(prediction_index) != len(
        selected_grades
    ):
        raise ValueError(
            "Prediction index length mismatch."
        )

    calendar_lookup = {
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
            prediction_index.iloc[
                sample
            ]["grade"]
        )

        if grade not in selected_grades:
            raise ValueError(
                f"Unexpected forecast grade: {grade}"
            )

        if grade in forecasts:
            raise ValueError(
                f"Duplicate prediction for {grade}."
            )

        weekly_forecasts = []

        for position in range(
            PREDICTION_LENGTH
        ):
            time_idx = int(
                decoder_indices[
                    sample,
                    position,
                ]
            )

            key = (
                grade,
                time_idx,
            )

            if key not in calendar_lookup:
                raise ValueError(
                    "Forecast calendar mismatch: "
                    f"{key}"
                )

            p10, p50, p90 = [
                float(value)
                for value in values[
                    sample,
                    position,
                ]
            ]

            week_date = calendar_lookup[key]

            weekly_forecasts.append(
                {
                    "week": week_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "horizon": position + 1,
                    "lower_demand_kg": round(
                        p10,
                        2,
                    ),
                    "median_demand_kg": round(
                        p50,
                        2,
                    ),
                    "upper_demand_kg": round(
                        p90,
                        2,
                    ),
                }
            )

        forecasts[grade] = weekly_forecasts

    if set(forecasts) != set(
        selected_grades
    ):
        raise ValueError(
            "Some selected tea grades "
            "are missing forecast results."
        )

    if (
        forecast_mode
        == "latest_completed_week_forecast"
    ):
        message = (
            "Four-week demand forecasts generated "
            "using the latest completed week's "
            "observations and the production TFT."
        )
    else:
        message = (
            "Historical forecast generated using "
            "the production TFT. A backfilled "
            "forecast must not be represented as "
            "a prediction issued in the past."
        )

    return {
        "status": "success",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
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
        "historical_input_weeks": (
            ENCODER_LENGTH
        ),
        "forecast_horizon_weeks": (
            PREDICTION_LENGTH
        ),
        "selected_grades": (
            selected_grades
        ),
        "tea_grade_count": len(
            selected_grades
        ),
        "forecast_count": (
            len(selected_grades)
            * PREDICTION_LENGTH
        ),
        "target": "demand_kg",
        "unit": "kg",
        "quantiles": {
            "lower": 0.1,
            "median": 0.5,
            "upper": 0.9,
        },
        "input_features": list(
            EXPECTED_FEATURES
        ),
        "forecasts": forecasts,
        "saved": False,
        "preview_latest_week": (
            preview_latest_week
        ),
        "message": message,
    }


def generate_eight_input_forecast_preview(
    batch_input: EightInputWeeklyBatch,
):
    """
    Convenience function for the controlled
    latest-week edited-input forecasting workflow.

    This function only calculates predictions.
    It does not modify MongoDB observations or
    store finalized forecast revisions.
    """

    return generate_eight_input_forecast(
        batch_input,
        preview_latest_week=True,
    )
