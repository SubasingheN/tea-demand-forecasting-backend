
"""
Forecast finalization and versioned history API for the
production nine-input Tea Demand Forecasting system.

Existing endpoints:
    POST /research/eight-input/finalize
    GET  /research/eight-input/history
    GET  /research/eight-input/history/previous/{grade}

Additional endpoints:
    GET  /research/eight-input/history/latest
    GET  /research/eight-input/history/revisions

Original finalized forecasts are stored in:
    eight_input_final_forecasts

Regenerated forecasts are stored in:
    eight_input_forecast_revisions

Revision conventions:
    Revision 0: Original finalized forecast.
    Revision 1+: Regenerated forecasts.

The original forecasts are never overwritten.

History retrieval is read-only.

This module does not retrain the TFT model or modify
the original historical dataset.
"""

import logging

from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_observation_storage_service import (
    TEA_GRADES,
    get_factory_history_bounds,
)

from app.services.eight_input_auto_forecast_service import (
    generate_and_finalize_saved_week,
)

from app.services.eight_input_forecast_storage_service import (
    COLLECTION_NAME,
    MODEL_NAME,
    MODEL_VERSION,
    compare_forecasts,
    get_forecast_collection,
    parse_week_date,
    serialize_forecast_document,
)

from app.services.eight_input_forecast_revision_service import (
    REVISION_COLLECTION_NAME,
    get_revision_collection,
    get_current_forecasts,
    get_complete_original_set,
    list_forecast_revisions,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/research/eight-input",
    tags=["Tea Demand Forecast History"],
)


def validate_operational_origin(batch):
    """
    Restrict public finalization to complete factory
    observations after the protected historical CSV.
    """

    if not isinstance(batch, EightInputWeeklyBatch):
        raise ValueError(
            "A valid factory weekly batch is required."
        )

    grades = [
        record.tea_type
        for record in batch.records
    ]

    if (
        len(grades) != len(TEA_GRADES)
        or set(grades) != set(TEA_GRADES)
    ):
        raise ValueError(
            "Forecast finalization requires "
            "all five tea-grade observations."
        )

    origin = batch.week_start_date.isoformat()

    bounds = get_factory_history_bounds()

    if origin <= bounds["latest_week"]:
        raise ValueError(
            "Historical reporting weeks cannot be "
            "finalized through this public endpoint. "
            "Historical forecast initialization "
            "is a separate backend operation."
        )

    return origin


def validate_history_origin(origin):
    """
    Validate a reporting-week origin without
    changing any stored records.
    """

    parsed = parse_week_date(
        origin,
        "forecast_origin",
    )

    return parsed.isoformat()


def serialize_history_record(
    document,
    revision,
    source,
):
    """
    Normalize an original or regenerated forecast
    for the combined history API.
    """

    record = serialize_forecast_document(
        document
    )

    record["revision"] = revision
    record["source"] = source

    if revision == 0:
        record["revision_type"] = (
            "original_forecast"
        )
        record["original_forecast_preserved"] = True

    else:
        record["revision_type"] = (
            "regenerated_forecast"
        )

    for field in (
        "revision_created_at",
    ):
        value = record.get(field)

        if hasattr(value, "isoformat"):
            record[field] = value.isoformat()

    return record


def load_finalized_origin_documents(origin):
    """
    Retrieve the original finalized forecast
    records for a reporting week.

    Does not include regenerated revisions.

    This helper is retained for compatibility.
    """

    collection = get_forecast_collection()

    documents = list(
        collection.find(
            {
                "forecast_origin": origin,
                "finalized": True,
            },
            {"_id": 0},
        )
    )

    records = {}

    for document in documents:
        grade = document.get("tea_type")

        if grade not in TEA_GRADES:
            raise ValueError(
                "Unexpected tea grade in forecast history."
            )

        if grade in records:
            raise ValueError(
                "Duplicate original finalized forecast "
                "records were found for this origin."
            )

        records[grade] = document

    return records


def get_previous_current_forecast(
    grade,
    current_origin,
):
    """
    Retrieve the current saved version of the
    forecast from exactly seven days earlier.

    Prefer the latest regenerated revision.
    Otherwise use the original forecast.
    """

    if grade not in TEA_GRADES:
        raise ValueError(
            "Unsupported tea grade."
        )

    origin_date = parse_week_date(
        current_origin,
        "current_origin",
    )

    previous_origin = (
        origin_date - timedelta(days=7)
    ).isoformat()

    previous_set = get_current_forecasts(
        previous_origin
    )

    if previous_set is None:
        return None

    forecast = previous_set[
        "forecasts"
    ].get(grade)

    if forecast is None:
        return None

    return serialize_history_record(
        forecast,
        revision=previous_set["revision"],
        source=previous_set["source"],
    )


def build_forecast_comparisons(origin):
    """
    Compare the latest saved forecasts for the
    current reporting week with the previous week.

    Comparison errors never change stored data.
    """

    comparisons = {}
    warnings = []

    current_set = get_current_forecasts(
        origin
    )

    if current_set is None:
        return {
            grade: []
            for grade in TEA_GRADES
        }, list(TEA_GRADES)

    for grade in TEA_GRADES:
        try:
            current = current_set[
                "forecasts"
            ].get(grade)

            previous = get_previous_current_forecast(
                grade,
                origin,
            )

            comparisons[grade] = compare_forecasts(
                previous,
                current,
            )

        except (ValueError, PyMongoError) as exc:
            logger.warning(
                "Forecast comparison unavailable "
                "for %s at %s: %s",
                grade,
                origin,
                exc,
            )

            comparisons[grade] = []
            warnings.append(grade)

    return comparisons, warnings


def get_latest_available_origin():
    """
    Find the latest reporting-week origin across
    original forecasts and regenerated revisions.

    Performs database reads only.
    """

    original = get_forecast_collection().find_one(
        {"finalized": True},
        {
            "_id": 0,
            "forecast_origin": 1,
        },
        sort=[
            ("forecast_origin", DESCENDING),
        ],
    )

    revised = get_revision_collection().find_one(
        {"finalized": True},
        {
            "_id": 0,
            "forecast_origin": 1,
        },
        sort=[
            ("forecast_origin", DESCENDING),
        ],
    )

    origins = [
        item["forecast_origin"]
        for item in (original, revised)
        if item is not None
    ]

    if not origins:
        return None

    return max(origins)


def build_complete_origin_history(origin):
    """
    Retrieve the original forecast and every
    regenerated revision for one reporting week.

    Revision 0 is the original forecast.

    Revisions 1+ are retrieved from the separate
    revision collection.

    No forecast generation occurs here.
    """

    original = get_complete_original_set(
        origin
    )

    revisions = list_forecast_revisions(
        origin
    )

    if not original and revisions:
        raise ValueError(
            "Forecast revisions exist without "
            "a complete original forecast."
        )

    records = []

    for grade in TEA_GRADES:
        document = original.get(grade)

        if document is None:
            continue

        records.append(
            serialize_history_record(
                document,
                revision=0,
                source="original_forecast",
            )
        )

    for document in revisions:
        records.append(
            serialize_history_record(
                document,
                revision=document["revision"],
                source="forecast_revision",
            )
        )

    records.sort(
        key=lambda item: (
            -item["revision"],
            TEA_GRADES.index(
                item["tea_type"]
            ),
        )
    )

    return records


@router.post("/finalize", status_code=201)
def finalize_eight_input_forecast(
    batch: EightInputWeeklyBatch,
):
    """
    Preserve the existing public finalization
    workflow for already saved factory observations.

    This endpoint does not create regenerated
    revisions or overwrite original forecasts.
    """

    try:
        origin = validate_operational_origin(
            batch
        )

        result = generate_and_finalize_saved_week(
            batch
        )

        if result.get("forecast_origin") != origin:
            raise ValueError(
                "Finalized forecast origin mismatch."
            )

        if result.get("finalized") is not True:
            raise ValueError(
                "Forecast finalization could "
                "not be confirmed."
            )

        comparisons, warnings = (
            build_forecast_comparisons(
                origin
            )
        )

        return {
            "status": "success",
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "collection": COLLECTION_NAME,
            "forecast_origin": origin,
            "forecast_status": result.get(
                "forecast_status"
            ),
            "forecast_generated": result.get(
                "forecast_generated",
                False,
            ),
            "saved_count": result.get(
                "saved_count",
                0,
            ),
            "saved": result.get(
                "forecast_generated",
                False,
            ),
            "forecasts": result.get(
                "forecasts",
                {},
            ),
            "comparisons": comparisons,
            "comparison_warnings": warnings,
            "finalized": True,
            "message": result.get(
                "message",
                "Four-week demand forecast is ready.",
            ),
        }

    except HTTPException:
        raise

    except DuplicateKeyError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Forecast records already exist "
                "for the selected reporting week."
            ),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Required production forecasting "
                "artifacts are unavailable."
            ),
        ) from exc

    except ValueError as exc:
        message = str(exc)

        if (
            "already exist" in message.lower()
            and "forecast" in message.lower()
        ):
            raise HTTPException(
                status_code=409,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=422,
            detail=message,
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "MongoDB forecast finalization failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast finalization could not be "
                "confirmed. Check forecast history "
                "before retrying."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Production forecast finalization failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast finalization failed. "
                "Check forecast history before retrying."
            ),
        ) from exc


@router.get("/history")
def get_eight_input_forecast_history(
    grade: str | None = Query(
        default=None,
        description="Optional tea-grade filter.",
    ),
    origin: date | None = Query(
        default=None,
        description="Optional reporting-week origin.",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
):
    """
    Retrieve combined original and regenerated
    forecast history, newest first.

    Includes:
        Original forecasts (Revision 0)
        Regenerated forecasts (Revision 1+)
        Historical backfilled forecasts
        Operational forecasts

    The limit applies to returned forecast records,
    not reporting weeks.

    Read-only operation.
    """

    try:
        if grade is not None and grade not in TEA_GRADES:
            raise ValueError(
                "Unsupported tea grade."
            )

        if origin is not None:
            origins = [
                validate_history_origin(
                    origin
                )
            ]

        else:
            original_origins = (
                get_forecast_collection().distinct(
                    "forecast_origin",
                    {"finalized": True},
                )
            )

            revision_origins = (
                get_revision_collection().distinct(
                    "forecast_origin",
                    {"finalized": True},
                )
            )

            origins = sorted(
                set(original_origins)
                | set(revision_origins),
                reverse=True,
            )

        forecasts = []

        for current_origin in origins:
            history = build_complete_origin_history(
                current_origin
            )

            if grade is not None:
                history = [
                    record
                    for record in history
                    if record.get("tea_type") == grade
                ]

            forecasts.extend(history)

            if len(forecasts) >= limit:
                break

        forecasts = forecasts[:limit]

        historical_count = sum(
            1
            for forecast in forecasts
            if forecast.get(
                "historical_backfill",
                False,
            )
        )

        operational_count = sum(
            1
            for forecast in forecasts
            if forecast.get(
                "operational_forecast",
                False,
            )
        )

        original_count = sum(
            1
            for forecast in forecasts
            if forecast["revision"] == 0
        )

        revision_count = (
            len(forecasts) - original_count
        )

        return {
            "status": "success",
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "original_collection": COLLECTION_NAME,
            "revision_collection": REVISION_COLLECTION_NAME,
            "count": len(forecasts),
            "original_forecast_count": original_count,
            "regenerated_forecast_count": revision_count,
            "historical_backfill_count": historical_count,
            "operational_forecast_count": operational_count,
            "forecasts": forecasts,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Forecast history retrieval failed."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc


@router.get("/history/latest")
def get_latest_eight_input_forecasts(
    origin: date | None = Query(
        default=None,
        description=(
            "Reporting-week origin. "
            "Defaults to the newest available week."
        ),
    ),
):
    """
    Retrieve the newest saved forecast set.

    Preference:
        Latest regenerated revision.
        Otherwise the original finalized forecast.

    Returns all five grades together.

    Designed for the Demand Forecast page.
    """

    try:
        if origin is None:
            selected_origin = (
                get_latest_available_origin()
            )

        else:
            selected_origin = (
                validate_history_origin(
                    origin
                )
            )

        if selected_origin is None:
            return {
                "status": "success",
                "available": False,
                "forecast_origin": None,
                "revision": None,
                "source": None,
                "forecasts": {},
            }

        current = get_current_forecasts(
            selected_origin
        )

        if current is None:
            return {
                "status": "success",
                "available": False,
                "forecast_origin": selected_origin,
                "revision": None,
                "source": None,
                "forecasts": {},
            }

        forecasts = {
            grade: serialize_history_record(
                current["forecasts"][grade],
                revision=current["revision"],
                source=current["source"],
            )
            for grade in TEA_GRADES
        }

        return {
            "status": "success",
            "available": True,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "forecast_origin": selected_origin,
            "revision": current["revision"],
            "source": current["source"],
            "grade_count": len(forecasts),
            "forecasts": forecasts,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Latest forecast retrieval failed."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc


@router.get("/history/revisions")
def get_eight_input_forecast_revisions(
    origin: date = Query(
        ...,
        description=(
            "Reporting-week origin for which all "
            "forecast revisions are required."
        ),
    ),
):
    """
    Retrieve the complete saved forecast history
    for one reporting week.

    Includes Revision 0 and every later revision.

    Designed for the Forecast History page.
    """

    try:
        selected_origin = (
            validate_history_origin(
                origin
            )
        )

        records = build_complete_origin_history(
            selected_origin
        )

        revision_numbers = sorted(
            {
                record["revision"]
                for record in records
            }
        )

        revision_groups = []

        for revision in revision_numbers:
            selected = [
                record
                for record in records
                if record["revision"] == revision
            ]

            revision_groups.append({
                "revision": revision,
                "source": (
                    "original_forecast"
                    if revision == 0
                    else "forecast_revision"
                ),
                "grade_count": len(selected),
                "forecasts": {
                    grade: next(
                        (
                            record
                            for record in selected
                            if record["tea_type"] == grade
                        ),
                        None,
                    )
                    for grade in TEA_GRADES
                },
            })

        return {
            "status": "success",
            "available": bool(records),
            "forecast_origin": selected_origin,
            "total_revisions": len(
                revision_groups
            ),
            "latest_revision": (
                max(revision_numbers)
                if revision_numbers
                else None
            ),
            "revisions": revision_groups,
            "forecasts": records,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Forecast revision retrieval failed."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc


@router.get("/history/previous/{grade}")
def get_previous_eight_input_forecast(
    grade: str,
    before_origin: date = Query(
        ...,
        description=(
            "Find the forecast from exactly "
            "seven days before this reporting week."
        ),
    ),
):
    """
    Retrieve the current saved version of the
    previous reporting week's forecast.

    Uses the latest revision when available.

    Does not substitute nonconsecutive weeks.
    """

    if grade not in TEA_GRADES:
        raise HTTPException(
            status_code=422,
            detail="Unsupported tea grade.",
        )

    try:
        previous = get_previous_current_forecast(
            grade,
            before_origin.isoformat(),
        )

        return {
            "status": "success",
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "tea_type": grade,
            "before_origin": (
                before_origin.isoformat()
            ),
            "available": previous is not None,
            "forecast": previous,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Previous forecast retrieval failed."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc
