from fastapi import (
    APIRouter,
    HTTPException,
)

from fastapi.responses import (
    FileResponse,
)

from pydantic import BaseModel


from app.db.forecast_repository import (
    get_latest_operational_forecast,
)

from app.services.weekly_forecast_service import (
    format_future_forecasts,
)

from app.services.forecast_history_service import (
    save_forecast_history,
    get_forecast_history,
)

from app.services.comparison_service import (
    compare_forecast_with_actuals,
)

from app.services.report_service import (
    generate_weekly_forecast_pdf,
    generate_operational_forecast_pdf,
)


router = APIRouter()


class GeneratedForecastSaveRequest(
    BaseModel
):
    input_week_start_date: str
    forecasts: list[dict]


class GeneratedForecastReportRequest(
    BaseModel
):
    input_week_start_date: str
    forecasts: list[dict]


@router.get(
    "/forecast/weekly"
)
def get_weekly_demand_forecast():
    try:
        forecasts = (
            format_future_forecasts()
        )

        return {
            "status": "success",
            "forecast_horizon_weeks": 4,
            "tea_type_count": len(
                forecasts
            ),
            "forecasts": forecasts,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate weekly "
                "demand forecast: "
                f"{str(error)}"
            ),
        )


@router.post(
    "/forecast/weekly/save"
)
def save_weekly_demand_forecast():
    try:
        forecasts = (
            format_future_forecasts()
        )

        saved_forecast = (
            save_forecast_history(
                forecasts
            )
        )

        return {
            "status": "success",
            "message": (
                "Weekly demand forecast "
                "saved successfully."
            ),
            "forecast":
                saved_forecast,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save weekly "
                "demand forecast: "
                f"{str(error)}"
            ),
        )


@router.post(
    "/forecast/save-from-generated"
)
def save_generated_forecast(
    request:
        GeneratedForecastSaveRequest,
):
    try:
        if not request.forecasts:
            raise ValueError(
                "Forecast data is required."
            )

        saved_forecast = (
            save_forecast_history(
                forecasts=
                    request.forecasts,

                input_week_start_date=
                    request
                    .input_week_start_date,

                source="operational",
            )
        )

        return {
            "status": "success",

            "message": (
                "Generated operational "
                "forecast saved "
                "successfully."
            ),

            "forecast":
                saved_forecast,
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
                "Unable to save generated "
                "operational forecast: "
                f"{str(error)}"
            ),
        )


@router.get(
    "/forecast/history"
)
def get_saved_forecast_history():
    try:
        history = (
            get_forecast_history()
        )

        return {
            "status": "success",
            "forecast_count":
                len(history),
            "history":
                history,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load forecast "
                "history: "
                f"{str(error)}"
            ),
        )


@router.get(
    "/forecast/latest-operational"
)
def get_latest_saved_operational_forecast():
    try:
        forecast = (
            get_latest_operational_forecast()
        )

        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No operational forecast "
                    "has been saved yet."
                ),
            )

        return {
            "status": "success",
            "forecast":
                forecast,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load the latest "
                "operational forecast: "
                f"{str(error)}"
            ),
        )


@router.get(
    "/forecast/history/"
    "{forecast_id}/comparison"
)
def compare_saved_forecast(
    forecast_id: str,
):
    try:
        comparison = (
            compare_forecast_with_actuals(
                forecast_id
            )
        )

        return {
            "status": "success",

            "message": (
                "Forecast comparison "
                "generated successfully."
            ),

            "comparison":
                comparison,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to compare forecast "
                "with actual demand: "
                f"{str(error)}"
            ),
        )


@router.get(
    "/forecast/report/download"
)
def download_weekly_forecast_report():
    try:
        report_path = (
            generate_weekly_forecast_pdf()
        )

        return FileResponse(
            path=str(
                report_path
            ),
            media_type=
                "application/pdf",
            filename=
                report_path.name,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate forecast "
                "report: "
                f"{str(error)}"
            ),
        )


@router.post(
    "/forecast/report/"
    "download-from-generated"
)
def download_generated_forecast_report(
    request:
        GeneratedForecastReportRequest,
):
    try:
        if not request.forecasts:
            raise ValueError(
                "Forecast data is required "
                "to generate the report."
            )

        report_path = (
            generate_operational_forecast_pdf(
                forecasts=
                    request.forecasts,

                input_week_start_date=
                    request
                    .input_week_start_date,
            )
        )

        return FileResponse(
            path=str(
                report_path
            ),
            media_type=
                "application/pdf",
            filename=
                report_path.name,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate "
                "operational forecast report: "
                f"{str(error)}"
            ),
        )