from fastapi import APIRouter, HTTPException

from app.schemas.weekly_input import WeeklyBatchInput
from app.services.weekly_input_service import (
    prepare_weekly_batch,
    save_weekly_batch,
)

router = APIRouter()


@router.post("/weekly-input/test")
def test_weekly_batch(
    batch_input: WeeklyBatchInput,
):
    try:
        prepared_records = prepare_weekly_batch(
            batch_input
        )

        return {
            "status": "success",
            "message": (
                "Weekly batch prepared successfully."
            ),
            "record_count": len(
                prepared_records
            ),
            "prepared_records": prepared_records,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


@router.post("/weekly-input/save")
def save_weekly_data(
    batch_input: WeeklyBatchInput,
):
    try:
        saved_records = save_weekly_batch(
            batch_input
        )

        return {
            "status": "success",
            "message": (
                "Weekly batch saved successfully."
            ),
            "record_count": len(
                saved_records
            ),
            "saved_records": saved_records,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )