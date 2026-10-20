
"""
Historical factory observation API for the eight-input TFT.

Endpoint:
    GET /factory/eight-input/history/{week_start_date}

Reads existing observations from the protected
Version 2 factory dataset and MongoDB extension.

No database writes are performed.
"""

from datetime import date, timedelta

import math

from fastapi import APIRouter, HTTPException
from pymongo.errors import PyMongoError

from app.services.eight_input_history_service import (
    load_eight_input_history,
)

from app.services.eight_input_observation_storage_service import (
    EIGHT_INPUT_FIELDS,
    TEA_GRADES,
    get_factory_history_bounds,
)


router = APIRouter(
    prefix="/factory/eight-input",
    tags=["Eight-Input Factory History"],
)


def valid_number(value):
    """Convert a finite numerical value to JSON-safe float."""

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            "Historical observation contains "
            "a non-finite numerical value."
        )

    return number


@router.get("/history/{week_start_date}")
def get_eight_input_historical_week(
    week_start_date: date,
):
    """
    Return the five tea-grade observations
    for one completed reporting week.
    """

    if week_start_date.weekday() != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Reporting week must start on Monday."
            ),
        )

    week_end = (
        week_start_date + timedelta(days=6)
    )

    if week_end >= date.today():
        raise HTTPException(
            status_code=422,
            detail=(
                "Only completed reporting weeks "
                "can be retrieved."
            ),
        )

    requested_week = week_start_date.isoformat()

    try:
        bounds = get_factory_history_bounds()

        history = load_eight_input_history()

        matches = history.loc[
            history["week_start_date"].dt.date
            == week_start_date
        ]

        if matches.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No factory observations found "
                    "for the selected reporting week."
                ),
            )

        records = []

        for grade in TEA_GRADES:
            grade_rows = matches.loc[
                matches["grade"] == grade
            ]

            if grade_rows.empty:
                continue

            if len(grade_rows) != 1:
                raise ValueError(
                    f"Duplicate historical records "
                    f"found for {grade}."
                )

            row = grade_rows.iloc[0]

            record = {
                "tea_type": grade,
            }

            for field in EIGHT_INPUT_FIELDS:
                value = valid_number(
                    row[field]
                )

                if field == "festival_flag":
                    if value not in (0.0, 1.0):
                        raise ValueError(
                            "Invalid historical festival flag."
                        )

                    record[field] = int(value)
                else:
                    record[field] = value

            records.append(record)

        source = (
            "protected_factory_csv"
            if requested_week <= bounds["latest_week"]
            else "mongodb_factory_observations"
        )

        return {
            "status": "success",
            "model": "eight_input_tft",
            "week_start_date": requested_week,
            "week_end_date": (
                week_end.isoformat()
            ),
            "source": source,
            "source_dataset_approval": "provisional",
            "auction_publication_timestamps_verified": False,
            "selected_grades": [
                record["tea_type"]
                for record in records
            ],
            "tea_grade_count": len(records),
            "records": records,
            "read_only": True,
            "write_performed": False,
        }

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc
