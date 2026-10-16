
from fastapi import (
    APIRouter,
    HTTPException,
)

from app.db.research_selected_forecast_repository import (
    get_selected_forecast_run,
)

from app.services.research_forecast_revision_service import (
    compare_forecast_runs,
)


router = APIRouter(
    prefix="/forecast/research",
    tags=["Research Forecast Revisions"],
)


@router.get("/compare")
def compare_research_forecast_versions(
    previous_run_id: str,
    new_run_id: str,
):
    """
    Compare two saved research TFT forecast runs.

    Matching is based on tea grade and
    forecast calendar date, not horizon.
    """
    if previous_run_id == new_run_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Previous and new forecast IDs "
                "must be different."
            ),
        )

    try:
        previous = get_selected_forecast_run(
            previous_run_id
        )

        current = get_selected_forecast_run(
            new_run_id
        )

        if previous is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Previous forecast run "
                    "was not found."
                ),
            )

        if current is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "New forecast run "
                    "was not found."
                ),
            )

        result = compare_forecast_runs(
            previous,
            current,
        )

        return {
            **result,
            "previous_run_id": previous_run_id,
            "new_run_id": new_run_id,
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to compare the selected "
                "research forecasts."
            ),
        ) from error
