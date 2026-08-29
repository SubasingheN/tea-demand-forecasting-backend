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

            "input_week_start_date":
                str(
                    batch_input.week_start_date
                ),

            "tea_type_count":
                len(
                    batch_input.records
                ),

            "summary":
                summary,
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