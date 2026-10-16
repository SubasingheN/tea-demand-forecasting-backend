
import json
import math
from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.weekly_input import (
    WeeklyBatchInput,
)

from app.services.demo_forecast_service import (
    create_demo_prediction_dataset,
)

from app.services.explainability_service import (
    generate_tft_interpretation,
    get_explainability_summary,
    get_explainability_summary_from_dataset,
)


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESEARCH_EXPLANATION_FILE = (
    PROJECT_ROOT
    / "research"
    / "outputs"
    / "tft_explainability"
    / "explanation_summary.json"
)


# ======================================================
# EXISTING EXPLAINABILITY ENDPOINTS
# ======================================================

@router.get(
    "/explainability/test"
)
def test_tft_explainability():
    try:
        result = (
            generate_tft_interpretation()
        )

        interpretation = result[
            "interpretation"
        ]

        return {
            "status": "success",
            "message": (
                "TFT interpretation generated "
                "successfully."
            ),
            "available_keys": list(
                interpretation.keys()
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate TFT "
                "interpretation: "
                f"{str(error)}"
            ),
        )


@router.get(
    "/explainability/summary"
)
def get_tft_explainability_summary():
    try:
        summary = (
            get_explainability_summary()
        )

        return {
            "status": "success",
            "message": (
                "TFT explainability summary "
                "generated successfully."
            ),
            "summary": summary,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate TFT "
                "explainability summary: "
                f"{str(error)}"
            ),
        )


@router.post(
    "/explainability/from-weekly-input"
)
def get_explainability_from_weekly_input(
    batch_input: WeeklyBatchInput,
):
    try:
        dataset_result = (
            create_demo_prediction_dataset(
                batch_input
            )
        )

        future_dataset = dataset_result[
            "future_dataset"
        ]

        summary = (
            get_explainability_summary_from_dataset(
                future_dataset
            )
        )

        return {
            "status": "success",

            "message": (
                "TFT explainability generated "
                "successfully from the latest "
                "weekly input."
            ),

            "input_week_start_date": str(
                batch_input.week_start_date
            ),

            "tea_type_count": len(
                batch_input.records
            ),

            "summary": summary,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate TFT "
                "explainability from weekly "
                "input: "
                f"{str(error)}"
            ),
        )


# ======================================================
# FROZEN RESEARCH EXPLAINABILITY
# ======================================================

def _validate_feature_importance(
    values,
    field_name,
):
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"{field_name} must be a non-empty list."
        )

    for item in values:
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid item in {field_name}."
            )

        feature = item.get("feature")
        percentage = item.get(
            "relative_importance_percent"
        )

        if (
            not isinstance(feature, str)
            or not feature
            or isinstance(percentage, bool)
            or not isinstance(
                percentage,
                (int, float),
            )
            or not math.isfinite(percentage)
            or percentage < 0
            or percentage > 100
        ):
            raise ValueError(
                f"Invalid feature importance in "
                f"{field_name}."
            )


def _validate_attention(values):
    if not isinstance(values, list) or not values:
        raise ValueError(
            "Historical attention must be "
            "a non-empty list."
        )

    for item in values:
        if not isinstance(item, dict):
            raise ValueError(
                "Invalid historical attention item."
            )

        offset = item.get(
            "weeks_before_latest_observation"
        )
        percentage = item.get(
            "relative_attention_percent"
        )

        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset >= 52
            or isinstance(percentage, bool)
            or not isinstance(
                percentage,
                (int, float),
            )
            or not math.isfinite(percentage)
            or percentage < 0
            or percentage > 100
        ):
            raise ValueError(
                "Invalid historical attention value."
            )


def _load_frozen_research_explanation():
    if not RESEARCH_EXPLANATION_FILE.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Saved research explainability "
                "results are unavailable."
            ),
        )

    try:
        with RESEARCH_EXPLANATION_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Research explanation must "
                "be a JSON object."
            )

        expected = {
            "status": "PASS",
            "model": "no_price_tft",
            "interpretation_partition": (
                "2024_validation"
            ),
            "forecast_horizon_weeks": 4,
            "encoder_history_weeks": 52,
            "validation_forecast_windows": 250,
            "auction_price_inputs_in_model": False,
        }

        for key, expected_value in expected.items():
            if (
                key not in data
                or type(data[key]) is not type(
                    expected_value
                )
                or data[key] != expected_value
            ):
                raise ValueError(
                    f"Unexpected research metadata: "
                    f"{key}."
                )

        for field in (
            "static_feature_importance",
            "encoder_feature_importance",
            "decoder_feature_importance",
        ):
            _validate_feature_importance(
                data.get(field),
                field,
            )

        _validate_attention(
            data.get(
                "top_historical_attention_positions"
            )
        )

        limitations = data.get("limitations")

        if (
            not isinstance(limitations, list)
            or not all(
                isinstance(item, str)
                for item in limitations
            )
        ):
            raise ValueError(
                "Research limitations are invalid."
            )

        return data

    except (
        json.JSONDecodeError,
        OSError,
        ValueError,
        TypeError,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Saved research explanation "
                f"failed validation: {error}"
            ),
        ) from error


@router.get(
    "/explainability/research/summary"
)
def get_research_tft_explainability():
    """
    Serve saved validation explanations from
    the selected no-price research TFT.

    This endpoint does not:
    - Retrain the model
    - Recalculate interpretation
    - Read future observed demand
    - Modify frozen test evaluation
    """

    data = _load_frozen_research_explanation()

    return {
        "status": "success",

        "model": data["model"],

        "interpretation_partition": data[
            "interpretation_partition"
        ],

        "forecast_horizon_weeks": data[
            "forecast_horizon_weeks"
        ],

        "encoder_history_weeks": data[
            "encoder_history_weeks"
        ],

        "validation_forecast_windows": data[
            "validation_forecast_windows"
        ],

        "auction_price_inputs_in_model": data[
            "auction_price_inputs_in_model"
        ],

        "static_feature_importance": data[
            "static_feature_importance"
        ],

        "encoder_feature_importance": data[
            "encoder_feature_importance"
        ],

        "decoder_feature_importance": data[
            "decoder_feature_importance"
        ],

        "top_historical_attention_positions": data[
            "top_historical_attention_positions"
        ],

        "limitations": data["limitations"],

        "generated_at_utc": data.get(
            "generated_at_utc"
        ),

        "message": (
            "Saved TFT explanations computed "
            "on 2024 validation windows. "
            "These are aggregate research "
            "explanations, not explanations "
            "generated specifically for the "
            "October 2026 forecasts."
        ),
    }
