from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.demand_alert_service import (
    generate_demand_alerts,
    generate_alerts_from_forecasts,
)


router = APIRouter()


class ForecastAlertRequest(BaseModel):
    forecasts: list[dict]


@router.get("/alerts/demand")
def get_demand_alerts():
    try:
        alerts = generate_demand_alerts()

        return {
            "status": "success",
            "message": (
                "Demand alerts generated "
                "successfully."
            ),
            "tea_type_count": len(alerts),
            "alerts": alerts,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate demand "
                "alerts: "
                f"{str(error)}"
            ),
        )


@router.post("/alerts/demand/from-forecast")
def get_alerts_from_forecast(
    request: ForecastAlertRequest,
):
    try:
        if not request.forecasts:
            raise ValueError(
                "Forecast data is required."
            )

        alerts = (
            generate_alerts_from_forecasts(
                request.forecasts
            )
        )

        return {
            "status": "success",
            "message": (
                "Demand alerts generated "
                "successfully from the "
                "provided TFT forecast."
            ),
            "tea_type_count":
                len(alerts),
            "alerts":
                alerts,
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
                "Unable to generate demand "
                "alerts from forecast: "
                f"{str(error)}"
            ),
        )