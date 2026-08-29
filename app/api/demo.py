from datetime import date, timedelta

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.db.weekly_input_repository import (
    upsert_weekly_input,
    get_latest_weekly_input,
)

from app.schemas.weekly_input import (
    WeeklyBatchInput,
)

from app.services.demo_forecast_service import (
    generate_demo_forecast,
)


router = APIRouter()


def get_current_monday():
    """
    Return the Monday of the
    current ongoing week.
    """

    today = date.today()

    return today - timedelta(
        days=today.weekday()
    )


def get_previous_monday():
    """
    Return the Monday of the most
    recently completed week.

    Weekly factory data is entered
    only after that week has ended.
    """

    current_monday = (
        get_current_monday()
    )

    return (
        current_monday -
        timedelta(weeks=1)
    )


@router.get(
    "/demo/latest-week"
)
def get_demo_latest_week():
    today = date.today()

    current_week = (
        get_current_monday()
    )

    completed_week = (
        get_previous_monday()
    )

    completed_week_end = (
        completed_week +
        timedelta(days=6)
    )

    forecast_end_week = (
        current_week +
        timedelta(weeks=3)
    )


    return {
        "status":
            "success",

        "mode":
            "operational",

        "today":
            today.isoformat(),

        "completed_week_start_date":
            completed_week.isoformat(),

        "completed_week_end_date":
            completed_week_end.isoformat(),

        "current_week_start_date":
            current_week.isoformat(),

        "forecast_start_date":
            current_week.isoformat(),

        "forecast_end_week_start_date":
            forecast_end_week.isoformat(),

        "tea_types": [
            "BOP",
            "BOPF",
            "Dust",
            "OP",
            "Pekoe",
        ],

        "forecast_horizon":
            4,
    }


@router.get(
    "/demo/weekly-input/latest"
)
def get_latest_saved_weekly_input():
    try:
        weekly_input = (
            get_latest_weekly_input()
        )


        if weekly_input is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No operational weekly "
                    "input has been saved yet."
                ),
            )


        return {
            "status":
                "success",

            "mode":
                "operational",

            "weekly_input":
                weekly_input,
        }


    except HTTPException:
        raise


    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load the latest "
                "weekly input from MongoDB: "
                f"{str(error)}"
            ),
        )


@router.post(
    "/demo/weekly-input"
)
def test_demo_weekly_input(
    batch_input: WeeklyBatchInput,
):
    try:
        save_result = (
            upsert_weekly_input(
                batch_input
            )
        )


        return {
            "status":
                "success",

            "mode":
                "operational",

            "message": (
                "Weekly batch saved "
                "successfully to MongoDB. "
                "Research dataset was not "
                "modified."
            ),

            "week_start_date":
                batch_input
                .week_start_date,

            "record_count":
                len(
                    batch_input.records
                ),

            "records":
                batch_input.records,

            "database_save":
                save_result,
        }


    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save weekly "
                "input to MongoDB: "
                f"{str(error)}"
            ),
        )


@router.post(
    "/demo/forecast"
)
def generate_demo_demand_forecast(
    batch_input: WeeklyBatchInput,
):
    try:
        forecast_result = (
            generate_demo_forecast(
                batch_input
            )
        )


        return {
            "status":
                "success",

            "mode":
                "demo",

            "message": (
                "4-week demand forecast "
                "generated successfully "
                "using the trained TFT "
                "model."
            ),

            **forecast_result,
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
                "Unable to generate demo "
                "demand forecast: "
                f"{str(error)}"
            ),
        )