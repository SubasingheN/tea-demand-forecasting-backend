
"""
Explainability for the frozen eight-input TFT.

Uses genuine TFT interpretation outputs:
- Encoder variable-selection weights
- Decoder variable-selection weights
- Static variable-selection weights
- Historical temporal attention

No training.
No checkpoint modifications.
No database writes.
No fabricated interpretation scores.

The analysis applies to the selected input
window(s), not necessarily to all validation
or test samples.
"""

import math

import numpy as np
import torch

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_history_service import (
    prepare_eight_input_history,
)

from app.services.eight_input_forecast_service import (
    load_eight_input_tft,
    build_eight_input_future,
    build_eight_input_inference_dataset,
)

from tft_eight_input_config import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    HISTORICAL_ONLY_REALS,
    KNOWN_FUTURE_REALS,
)


MODEL_NAME = "eight_input_tft"


def to_numpy(value):
    """Safely convert model outputs to NumPy."""

    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(value)


def get_interpretation_item(
    interpretation,
    name,
):
    """Read an item from the interpretation."""

    if isinstance(interpretation, dict):
        return interpretation.get(name)

    return getattr(
        interpretation,
        name,
        None,
    )


def to_vector(value, name):
    """
    Convert a reduced interpretation value
    into a one-dimensional numeric vector.

    Ambiguous multidimensional shapes are
    rejected instead of silently averaged.
    """

    if value is None:
        return None

    array = np.asarray(
        to_numpy(value),
        dtype=np.float64,
    ).squeeze()

    if array.ndim == 0:
        array = array.reshape(1)

    if array.ndim != 1:
        raise ValueError(
            f"{name}: expected a reduced "
            f"one-dimensional vector, "
            f"received shape {array.shape}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}: interpretation contains "
            "non-finite values."
        )

    return array


def normalize_variable_weights(
    values,
    variable_names,
    group_name,
):
    """
    Convert learned variable-selection weights
    into within-group relative percentages.

    These are model weights, not causal effects.
    """

    vector = to_vector(
        values,
        group_name,
    )

    if vector is None:
        return []

    names = list(variable_names)

    if len(vector) != len(names):
        raise ValueError(
            f"{group_name}: expected "
            f"{len(names)} feature weights, "
            f"received {len(vector)}."
        )

    if np.any(vector < -1e-8):
        raise ValueError(
            f"{group_name}: negative "
            "variable-selection weights."
        )

    vector = np.maximum(vector, 0.0)

    total = float(vector.sum())

    if total <= 0:
        return []

    result = []

    for feature, weight in zip(
        names,
        vector,
    ):
        percentage = (
            float(weight) / total * 100.0
        )

        result.append({
            "feature": str(feature),
            "relative_importance_percent": round(
                percentage,
                4,
            ),
            "is_eight_input_feature": (
                feature in HISTORICAL_ONLY_REALS
            ),
        })

    result.sort(
        key=lambda item:
            item["relative_importance_percent"],
        reverse=True,
    )

    return result


def normalize_temporal_attention(values):
    """
    Interpret attention over the 52 historical
    positions for prediction horizon one.

    The newest historical observation is
    zero weeks before the forecast origin.
    """

    vector = to_vector(
        values,
        "historical_attention",
    )

    if vector is None:
        return []

    if len(vector) != ENCODER_LENGTH:
        raise ValueError(
            "Temporal attention length does "
            "not match the expected "
            f"{ENCODER_LENGTH}-week encoder. "
            f"Received {len(vector)} positions."
        )

    if np.any(vector < -1e-8):
        raise ValueError(
            "Temporal attention contains "
            "negative weights."
        )

    vector = np.maximum(
        vector,
        0.0,
    )

    total = float(vector.sum())

    if total <= 0:
        return []

    result = []

    for position, weight in enumerate(
        vector
    ):
        weeks_before_latest = (
            ENCODER_LENGTH - position - 1
        )

        percentage = (
            float(weight) / total * 100.0
        )

        result.append({
            "encoder_position": position,
            "weeks_before_latest_observation": (
                weeks_before_latest
            ),
            "relative_attention_percent": round(
                percentage,
                4,
            ),
        })

    return result


def generate_eight_input_explanation(
    batch_input: EightInputWeeklyBatch,
):
    """
    Extract genuine interpretation information
    for the selected forecast origin and grades.

    Returns only quantities calculated from
    the checkpoint's raw output.
    """

    prepared = prepare_eight_input_history(
        batch_input
    )

    history = prepared[
        "historical_dataframe"
    ]

    selected_grades = prepared[
        "selected_grades"
    ]

    model = load_eight_input_tft()

    future = build_eight_input_future(
        history,
        selected_grades,
    )

    dataset = (
        build_eight_input_inference_dataset(
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

    model.eval()

    with torch.inference_mode():
        prediction = model.predict(
            loader,
            mode="raw",
            return_index=True,
            trainer_kwargs={
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )

        interpretation = (
            model.interpret_output(
                prediction.output,
                reduction="sum",
                attention_prediction_horizon=0,
            )
        )

    encoder_weights = (
        normalize_variable_weights(
            get_interpretation_item(
                interpretation,
                "encoder_variables",
            ),
            model.encoder_variables,
            "encoder_variables",
        )
    )

    decoder_weights = (
        normalize_variable_weights(
            get_interpretation_item(
                interpretation,
                "decoder_variables",
            ),
            model.decoder_variables,
            "decoder_variables",
        )
    )

    static_weights = (
        normalize_variable_weights(
            get_interpretation_item(
                interpretation,
                "static_variables",
            ),
            model.static_variables,
            "static_variables",
        )
    )

    temporal_attention = (
        normalize_temporal_attention(
            get_interpretation_item(
                interpretation,
                "attention",
            )
        )
    )

    if not any([
        encoder_weights,
        decoder_weights,
        static_weights,
        temporal_attention,
    ]):
        raise ValueError(
            "The frozen checkpoint returned "
            "no usable interpretation outputs."
        )

    return {
        "status": "success",
        "model": MODEL_NAME,
        "explanation_source": (
            "frozen_checkpoint_raw_output"
        ),
        "forecast_origin": prepared[
            "forecast_origin"
        ],
        "forecast_type": prepared[
            "forecast_mode"
        ],
        "selected_grades": selected_grades,
        "sample_count": len(
            selected_grades
        ),
        "encoder_history_weeks": (
            ENCODER_LENGTH
        ),
        "forecast_horizon_weeks": (
            PREDICTION_LENGTH
        ),
        "historical_input_features": list(
            HISTORICAL_ONLY_REALS
        ),
        "known_future_features": list(
            KNOWN_FUTURE_REALS
        ),
        "static_feature_importance": (
            static_weights
        ),
        "encoder_feature_importance": (
            encoder_weights
        ),
        "decoder_feature_importance": (
            decoder_weights
        ),
        "historical_attention": (
            temporal_attention
        ),
        "attention_prediction_horizon": 1,
        "aggregation": (
            "Sum across selected inference "
            "windows, normalized within each "
            "feature group."
        ),
        "limitations": [
            (
                "Variable-selection weights "
                "describe model behavior, not "
                "causal effects."
            ),
            (
                "Feature importance percentages "
                "are normalized separately "
                "within each feature group."
            ),
            (
                "Historical attention describes "
                "the selected prediction horizon, "
                "not all four horizons."
            ),
            (
                "The explanation applies only "
                "to the selected input windows."
            ),
            (
                "The source dataset remains "
                "provisional and historical "
                "auction publication timestamps "
                "are not fully verified."
            ),
        ],
        "saved": False,
    }
