
"""
Read-only historical observations API.

Retrieves six-input factory observations from:
- The protected factory CSV
- New completed-week MongoDB observations

Missing grades are reported but never invented.

No database writes or model modifications occur.
"""

import logging
from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException
from pymongo.errors import PyMongoError

from app.services.six_input_history_service import (
    load_six_input_history,
)

from app.services.six_input_observation_storage_service import (
    get_factory_history_bounds,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/factory/six-input",
    tags=["Six-Input Factory History"],
)

TEA_GRADES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]

INPUT_FIELDS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_rate",
    "festival_flag",
    "colombo_auction_price_lkr",
]


@router.get("/history/{week_start_date}")
def get_six_input_factory_history(
    week_start_date: date,
):
    """
    Retrieve available recorded observations for
    a completed Monday-to-Sunday reporting week.

    Returns an empty records list when the
    requested completed week has no observations.
    """

    if week_start_date.weekday() != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "The reporting week must start "
                "on Monday."
            ),
        )

    week_end = (
        week_start_date + timedelta(days=6)
    )

    if week_end >= date.today():
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected reporting week "
                "has not yet been completed."
            ),
        )

    try:
        bounds = get_factory_history_bounds()

        requested = week_start_date.isoformat()

        if requested < bounds["first_week"]:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The selected reporting week "
                    "precedes the factory dataset."
                ),
            )

        history = load_six_input_history()

        selected = history.loc[
            history["week_start_date"]
            == pd.Timestamp(week_start_date)
        ]

        source = (
            "protected_factory_csv"
            if requested <= bounds["latest_week"]
            else "mongodb_factory_observations"
        )

        records = []

        for grade in TEA_GRADES:
            grade_rows = selected.loc[
                selected["grade"] == grade
            ]

            if grade_rows.empty:
                continue

            if len(grade_rows) != 1:
                raise ValueError(
                    "Duplicate historical observations "
                    f"for {grade} on {requested}."
                )

            row = grade_rows.iloc[0]

            record = {
                "tea_type": grade,
            }

            for field in INPUT_FIELDS:
                if field == "festival_flag":
                    record[field] = int(row[field])
                else:
                    record[field] = float(row[field])

            records.append(record)

        available = [
            record["tea_type"]
            for record in records
        ]

        missing = [
            grade
            for grade in TEA_GRADES
            if grade not in available
        ]

        return {
            "status": "success",
            "week_start_date": requested,
            "week_end_date": week_end.isoformat(),
            "source": source,
            "selected_grades": available,
            "record_count": len(records),
            "missing_grades": missing,
            "records": records,
            "read_only": True,
        }

    except HTTPException:
        raise

    except (
        FileNotFoundError,
        PyMongoError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        logger.exception(
            "Failed to retrieve historical "
            "observations for %s.",
            week_start_date,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Factory historical observations "
                "could not be retrieved."
            ),
        ) from error

    except Exception as error:
        logger.exception(
            "Unexpected historical API error."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected historical data "
                "processing error."
            ),
        ) from error
