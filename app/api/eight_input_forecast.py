
"""
FastAPI endpoint for the frozen eight-input TFT.

Generates four-week tea demand forecasts for
BOP, BOPF, Dust, OP and Pekoe.

Historical inputs:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_rate
    festival_flag
    auction_price_high_lkr
    auction_price_medium_lkr
    auction_price_low_lkr

Returns lower, median and upper demand estimates.

No retraining or database writes are performed.
"""

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_forecast_service import (
    generate_eight_input_forecast,
)


router = APIRouter(
    prefix="/research/eight-input",
    tags=["Eight-Input Tea Demand Forecasting"],
)


@router.post("/forecast")
def forecast_eight_input(
    batch: EightInputWeeklyBatch,
):
    """
    Generate four-week demand forecasts.

    Submitted records must match existing
    historical observations for the requested
    completed reporting week.
    """

    try:
        result = generate_eight_input_forecast(
            batch
        )

        return result

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception:
        # Do not expose database connection strings
        # or internal filesystem details to clients.
        raise HTTPException(
            status_code=500,
            detail=(
                "The eight-input forecasting service "
                "encountered an internal error."
            ),
        )
