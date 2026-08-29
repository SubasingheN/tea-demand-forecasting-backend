from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.demand_alert_service import (
    generate_alerts_from_forecasts,
)

from app.services.recommendation_service import (
    generate_demand_recommendations,
    generate_recommendations_from_alerts,
)


router = APIRouter()


class ForecastRecommendationRequest(
    BaseModel
):
    forecasts: list[dict]


@router.get(
    "/recommendations/demand"
)
def get_demand_recommendations():
    try:
        recommendations = (
            generate_demand_recommendations()
        )

        return {
            "status": "success",
            "message": (
                "Demand recommendations "
                "generated successfully."
            ),
            "tea_type_count": len(
                recommendations
            ),
            "recommendations":
                recommendations,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate demand "
                f"recommendations: {str(error)}"
            ),
        )


@router.post(
    "/recommendations/demand/from-forecast"
)
def get_recommendations_from_forecast(
    request: ForecastRecommendationRequest,
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

        recommendations = (
            generate_recommendations_from_alerts(
                alerts
            )
        )

        return {
            "status": "success",
            "message": (
                "Demand recommendations "
                "generated successfully "
                "from the provided "
                "TFT forecast."
            ),
            "tea_type_count": len(
                recommendations
            ),
            "recommendations":
                recommendations,
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
                "recommendations from "
                "forecast: "
                f"{str(error)}"
            ),
        )