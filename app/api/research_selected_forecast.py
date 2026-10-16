
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from app.schemas.research_weekly_input import (
    ResearchWeeklyBatchInput,
)

from app.services.research_selected_forecast_service import (
    generate_selected_tft_forecast,
)

from app.db.research_selected_forecast_repository import (
    save_selected_forecast_run,
)


router = APIRouter(
    prefix="/forecast/research",
    tags=["Selective Research TFT Forecasting"],
)


class SelectedForecastSaveRequest(BaseModel):
    """
    Request to generate and save a research
    forecast from selected weekly inputs.
    """

    model_config = ConfigDict(extra="forbid")

    weekly_input: ResearchWeeklyBatchInput


def verify_completed_week(batch_input):
    """
    Ensure the input week has finished
    before allowing operational forecasting.
    """
    week_end = (
        batch_input.week_start_date
        + timedelta(days=6)
    )

    return week_end < date.today()


@router.post("/selected")
def generate_selected_research_forecast(
    batch_input: ResearchWeeklyBatchInput,
):
    """
    Generate predictions without saving.
    """
    try:
        return generate_selected_tft_forecast(
            batch_input
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Required research model or "
                "dataset file is unavailable."
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Selected research forecast "
                "could not be generated."
            ),
        ) from error


@router.post("/selected/save")
def generate_and_save_selected_forecast(
    request: SelectedForecastSaveRequest,
):
    """
    Generate a selected-grade TFT forecast
    and save it as an explicitly labeled
    test run.

    Operational saving will be enabled
    after verified factory input persistence
    is implemented.
    """
    try:
        batch_input = request.weekly_input

        # Generate on the server using the
        # trusted frozen research checkpoint.
        forecast = generate_selected_tft_forecast(
            batch_input
        )

        input_records = [
            record.model_dump()
            for record in batch_input.records
        ]

        saved = save_selected_forecast_run(
            forecast_result=forecast,
            input_source="test_data",
            input_records=input_records,
        )

        return {
            "status": "success",
            "message": (
                "Selective TFT forecast generated "
                "and saved as a test-data run."
            ),
            "run_id": saved["run_id"],
            "input_source": "test_data",
            "forecast_origin": saved[
                "forecast_origin"
            ],
            "selected_grades": saved[
                "selected_grades"
            ],
            "forecast_count": saved[
                "forecast_count"
            ],
            "forecasts": saved["forecasts"],
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Required research model or "
                "dataset file is unavailable."
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate and save "
                "the selective research forecast."
            ),
        ) from error
