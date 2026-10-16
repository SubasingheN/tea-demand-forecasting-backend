
"""
FastAPI endpoints for six-input TFT forecasting.

Supports historical replay and forecasts from
the latest completed factory observation.

Does not retrain models, modify datasets,
or save forecast results.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.six_input_weekly import (
    SixInputWeeklyBatch,
)

from app.services.six_input_forecast_service import (
    generate_six_input_forecast,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/forecast/six-input",
    tags=["Six-Input TFT Forecasting"],
)


@router.post("/selected")
def forecast_six_input_selected(
    batch_input: SixInputWeeklyBatch,
):
    """
    Generate four-week P10/P50/P90 forecasts
    for one to five selected tea grades.

    Historical inputs:
    - demand_kg
    - production_kg
    - inventory_kg
    - usd_lkr_rate
    - festival_flag
    - colombo_auction_price_lkr
    """

    try:
        return generate_six_input_forecast(
            batch_input
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        logger.error(
            "Six-input model resource missing: %s",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Required six-input model or "
                "factory dataset is unavailable."
            ),
        ) from error

    except Exception as error:
        logger.exception(
            "Six-input forecast generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate the six-input "
                "TFT forecast. Check backend logs."
            ),
        ) from error
