
"""
FastAPI endpoints for eight-input TFT explainability.

The service uses the frozen eight-input TFT
checkpoint and recorded historical observations.

POST /research/eight-input/explain

No retraining.
No MongoDB writes.
No modification of the original dataset.
"""

import logging

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_explainability_service import (
    generate_eight_input_explanation,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/research/eight-input",
    tags=["Eight-Input TFT Explainability"],
)


@router.post("/explain")
def explain_eight_input_forecast(
    batch: EightInputWeeklyBatch,
):
    """
    Generate model-derived explanations
    for selected tea grades and a completed
    historical reporting week.

    The same input validation and historical
    preprocessing used by the forecasting
    service are applied.
    """

    try:
        result = generate_eight_input_explanation(
            batch
        )

        if result.get("model") != "eight_input_tft":
            raise ValueError(
                "Unexpected explanation model."
            )

        if (
            result.get("explanation_source")
            != "frozen_checkpoint_raw_output"
        ):
            raise ValueError(
                "Explanation provenance could "
                "not be verified."
            )

        return result

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Eight-input TFT explanation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to extract interpretation "
                "outputs from the frozen eight-input "
                "TFT checkpoint. Check backend logs "
                "for diagnostic details."
            ),
        ) from exc
