
"""
Factory observation API for nine-input Tea Demand Forecasting.

Existing endpoints:
    GET   /factory/eight-input/status
    GET   /factory/eight-input/observations
    POST  /factory/eight-input/observations
    POST  /factory/eight-input/observations/batch
    POST  /factory/eight-input/observations/forecast/retry
    PATCH /factory/eight-input/observations

New endpoint:
    POST  /factory/eight-input/observations/forecast/generate

Generate Forecasts workflow:
    1. Validate five tea-grade observations.
    2. Validate latest completed reporting week.
    3. Generate predictions using submitted inputs in memory.
    4. Atomically save observations and forecasts.
    5. Return the newest forecast revision.

Original historical records remain protected.

Previously finalized forecasts are preserved.
Regenerated forecasts use separate revision storage.

The legacy correction endpoint remains disabled.

No database writes occur on module import.
"""

import logging
import threading

from datetime import date, timedelta

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from pymongo.errors import PyMongoError

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_observation_storage_service import (
    COLLECTION_NAME,
    DATA_SOURCE,
    EIGHT_INPUT_FIELDS,
    MODEL_NAME,
    TEA_GRADES,
    current_factory_date,
    get_factory_history_bounds,
    get_observation_collection,
    latest_completed_week_start,
    list_eight_input_observations,
    save_eight_input_observation,
    validate_completed_week,
    validate_latest_completed_week,
)

from app.services.eight_input_batch_storage_service import (
    save_eight_input_observation_batch,
    validate_weekly_batch,
)

from app.services.eight_input_auto_forecast_service import (
    generate_and_finalize_saved_week,
)

from app.services.eight_input_forecast_service import (
    generate_eight_input_forecast_preview,
)

from app.services.eight_input_generate_forecast_service import (
    commit_generated_forecast,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/factory/eight-input",
    tags=["Factory Tea Demand Observations"],
)

MODEL_VERSION = "production_v1"
FORECAST_HORIZON_WEEKS = 4

# Prevent duplicate generation within this Python process.
# MongoDB also validates revisions during the commit.
_generation_guard = threading.Lock()
_active_generation_weeks = set()


class FactoryObservationCorrection(BaseModel):
    """
    Preserve the legacy correction request structure.

    This operation remains disabled until
    verified operator authorization is available.
    """

    batch: EightInputWeeklyBatch

    reason: str = Field(
        min_length=3,
        max_length=500,
    )


class ForecastRetryRequest(BaseModel):
    """Identify an already saved reporting week."""

    week_start_date: date


def corrections_enabled():
    """
    Legacy corrections remain disabled without
    verified operator authorization.
    """

    return False


def require_corrections_enabled():
    """Reject legacy unauthenticated corrections."""

    raise HTTPException(
        status_code=403,
        detail=(
            "Factory observation corrections are "
            "unavailable until authenticated "
            "operator authorization is configured."
        ),
    )


def build_observation(batch):
    """
    Convert a single-record weekly batch into
    the existing observation storage format.
    """

    if len(batch.records) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Submit exactly one tea-grade "
                "observation for this operation."
            ),
        )

    return {
        "week_start_date": (
            batch.week_start_date.isoformat()
        ),
        **batch.records[0].model_dump(),
    }


def load_saved_forecast_batch(week_start_date):
    """
    Build a validated forecast request from
    already saved MongoDB observations.

    This function performs database reads only.
    """

    if not isinstance(week_start_date, date):
        raise ValueError(
            "A valid reporting date is required."
        )

    if week_start_date.weekday() != 0:
        raise ValueError(
            "The reporting week must begin Monday."
        )

    validate_completed_week(week_start_date)

    origin = week_start_date.isoformat()
    bounds = get_factory_history_bounds()

    if origin <= bounds["latest_week"]:
        raise ValueError(
            "Forecast retry is available only for "
            "observations saved after the protected "
            "historical dataset."
        )

    collection = get_observation_collection()

    documents = list(
        collection.find(
            {
                "week_start_date": origin,
            },
            {
                "_id": 0,
            },
        )
    )

    if len(documents) != len(TEA_GRADES):
        raise ValueError(
            "Five saved tea-grade observations "
            "are required before forecasting."
        )

    records_by_grade = {}

    for document in documents:
        grade = document.get("tea_type")

        if grade not in TEA_GRADES:
            raise ValueError(
                "Stored observations contain "
                "an unsupported tea grade."
            )

        if grade in records_by_grade:
            raise ValueError(
                "Duplicate tea-grade observations "
                "were found for this reporting week."
            )

        if document.get("source") != DATA_SOURCE:
            raise ValueError(
                "The selected week does not contain "
                "valid factory-entered observations."
            )

        if document.get("record_type") != (
            "factory_observation"
        ):
            raise ValueError(
                "A stored record has an invalid "
                "observation type."
            )

        record = {
            "tea_type": grade,
        }

        for field in EIGHT_INPUT_FIELDS:
            if field not in document:
                raise ValueError(
                    f"{grade}: missing {field}. "
                    "Saved observations may require "
                    "nine-input migration."
                )

            record[field] = document[field]

        records_by_grade[grade] = record

    if set(records_by_grade) != set(TEA_GRADES):
        raise ValueError(
            "The saved reporting week is incomplete."
        )

    try:
        return EightInputWeeklyBatch.model_validate(
            {
                "week_start_date": origin,
                "records": [
                    records_by_grade[grade]
                    for grade in TEA_GRADES
                ],
            }
        )

    except ValidationError as exc:
        raise ValueError(
            "Saved observations contain invalid "
            "or inconsistent nine-input values."
        ) from exc


def format_forecast_response(result):
    """
    Preserve the existing frontend-facing
    forecast response format.
    """

    return {
        "status": "success",
        "forecast": {
            "status": result.get(
                "forecast_status",
                "success",
            ),
            "forecast_generated": result.get(
                "forecast_generated",
                False,
            ),
            "finalized": result.get(
                "finalized",
                False,
            ),
            "forecast_origin": result.get(
                "forecast_origin"
            ),
            "forecast_count": result.get(
                "forecast_count",
                0,
            ),
            "forecasts": result.get(
                "forecasts",
                {},
            ),
            "message": result.get(
                "message",
                "Forecast is ready.",
            ),
        },
    }


def acquire_generation_slot(origin):
    """
    Reject overlapping generation requests for the
    same reporting week within one backend process.

    This is not a distributed lock. MongoDB remains
    responsible for transaction conflict detection.
    """

    with _generation_guard:
        if origin in _active_generation_weeks:
            return False

        _active_generation_weeks.add(origin)
        return True


def release_generation_slot(origin):
    """Release a completed generation request."""

    with _generation_guard:
        _active_generation_weeks.discard(origin)


def format_generated_forecast_response(
    committed,
    prediction,
):
    """
    Return the newly committed forecast and its
    revision in a frontend-friendly format.
    """

    origin = committed["forecast_origin"]
    revision = committed["revision"]

    return {
        "status": "success",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "week_start_date": origin,
        "weekly_data": {
            "status": (
                "saved"
                if committed["operation"] == "created"
                else "updated"
            ),
            "saved": True,
            "saved_count": committed[
                "observation_count"
            ],
            "saved_grades": list(TEA_GRADES),
            "atomic": True,
        },
        "forecast": {
            "status": "newly_finalized",
            "forecast_generated": True,
            "finalized": True,
            "forecast_origin": origin,
            "forecast_count": committed[
                "forecast_count"
            ],
            "revision": revision,
            "forecasts": prediction["forecasts"],
            "message": (
                "The four-week demand forecast "
                "was generated successfully."
            ),
        },
        "revision": revision,
        "operation": committed["operation"],
        "input_fingerprint": committed[
            "input_fingerprint"
        ],
        "message": (
            "Weekly observations and four-week "
            "forecasts saved successfully."
        ),
    }


@router.get("/status")
def get_eight_input_status():
    """Read factory observation status."""

    try:
        bounds = get_factory_history_bounds()

        collection = get_observation_collection()

        count = collection.count_documents({})

        latest = collection.find_one(
            {},
            {
                "_id": 0,
                "week_start_date": 1,
            },
            sort=[
                ("week_start_date", -1),
            ],
        )

        current_week = (
            latest_completed_week_start()
        )

        return {
            "status": "success",
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "collection": COLLECTION_NAME,
            "source_dataset": bounds,
            "mongodb": {
                "observation_count": count,
                "latest_saved_week": (
                    latest["week_start_date"]
                    if latest
                    else None
                ),
            },
            "supported_grades": list(
                TEA_GRADES
            ),
            "required_historical_inputs": list(
                EIGHT_INPUT_FIELDS
            ),
            "historical_input_count": len(
                EIGHT_INPUT_FIELDS
            ),
            "latest_completed_week": (
                current_week.isoformat()
            ),
            "latest_completed_week_end": (
                current_week
                + timedelta(days=6)
            ).isoformat(),
            "factory_current_date": (
                current_factory_date().isoformat()
            ),
            "normal_entry_policy": (
                "latest_completed_week_only"
            ),
            "corrections_enabled": False,
            "authentication_configured": False,
            "batch_saving_supported": True,
            "automatic_forecasting_supported": True,
            "forecast_retry_supported": True,
            "forecast_revisions_supported": True,
            "generate_forecasts_supported": True,
            "historical_forecast_initialization": (
                "separate_backend_operation"
            ),
            "write_performed": False,
        }

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Factory observation status unavailable."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc


@router.get("/observations")
def get_eight_input_observations(
    grade: str | None = Query(
        default=None
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
):
    """Retrieve previously saved observations."""

    try:
        observations = (
            list_eight_input_observations(
                grade=grade,
                limit=limit,
            )
        )

        return {
            "status": "success",
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "collection": COLLECTION_NAME,
            "count": len(observations),
            "observations": observations,
            "write_performed": False,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Factory observations unavailable."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB is currently unavailable.",
        ) from exc


@router.post(
    "/observations",
    status_code=201,
)
def create_eight_input_observation(
    batch: EightInputWeeklyBatch,
):
    """
    Preserve the existing single-grade creation API.

    Complete weeks should normally be saved using
    the batch or Generate Forecasts endpoint.
    """

    observation = build_observation(
        batch
    )

    try:
        result = save_eight_input_observation(
            observation
        )

        return {
            **result,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "collection": COLLECTION_NAME,
            "forecast_generated": False,
        }

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Single factory observation save failed."
        )

        raise HTTPException(
            status_code=503,
            detail="MongoDB operation failed.",
        ) from exc


@router.post(
    "/observations/batch",
    status_code=201,
)
def create_eight_input_observation_batch(
    batch: EightInputWeeklyBatch,
):
    """
    Preserve the existing atomic batch-save API.

    This legacy endpoint still attempts automatic
    forecasting after saving observations.
    """

    try:
        saved = (
            save_eight_input_observation_batch(
                batch
            )
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Atomic factory weekly save failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Weekly data saving could not be "
                "confirmed. Check saved records "
                "before attempting another submission."
            ),
        ) from exc

    response = {
        "status": "success",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "collection": COLLECTION_NAME,
        "week_start_date": saved[
            "week_start_date"
        ],
        "weekly_data": {
            "status": "saved",
            "saved": True,
            "saved_count": saved[
                "saved_count"
            ],
            "saved_grades": saved[
                "saved_grades"
            ],
            "atomic": True,
        },
        "forecast": {
            "status": "pending",
            "forecast_generated": False,
            "finalized": False,
            "forecasts": None,
        },
    }

    try:
        forecast_result = (
            generate_and_finalize_saved_week(
                batch
            )
        )

        response["forecast"] = (
            format_forecast_response(
                forecast_result
            )["forecast"]
        )

        response["message"] = (
            "Weekly data saved and the four-week "
            "demand forecast is ready."
        )

    except Exception:
        logger.exception(
            "Weekly observations saved but "
            "forecast finalization could not "
            "be confirmed for %s.",
            saved["week_start_date"],
        )

        response["status"] = (
            "saved_forecast_unconfirmed"
        )

        response["forecast"] = {
            "status": "unconfirmed",
            "forecast_generated": False,
            "finalized": False,
            "forecast_origin": saved[
                "week_start_date"
            ],
            "forecasts": None,
            "message": (
                "Weekly observations were saved. "
                "Forecast generation or saving "
                "could not be confirmed."
            ),
        }

        response["message"] = (
            "Weekly data saved; forecast requires "
            "checking or retrying."
        )

    return response


@router.post(
    "/observations/forecast/generate"
)
def generate_factory_demand_forecasts(
    batch: EightInputWeeklyBatch,
):
    """
    Generate and save the current four-week forecasts.

    The forecasting model receives the submitted
    latest-week values through an in-memory preview.

    Observation and forecast writes are committed
    together using a MongoDB transaction.

    A successful response contains the new revision
    and all five grades' four-week predictions.
    """

    origin = batch.week_start_date.isoformat()

    if not acquire_generation_slot(origin):
        raise HTTPException(
            status_code=409,
            detail=(
                "Forecast generation is already in "
                "progress for this reporting week. "
                "Please wait for it to finish."
            ),
        )

    try:
        # Validate the reporting week and all five
        # grades before invoking the TFT model.
        validate_latest_completed_week(
            batch.week_start_date
        )

        validate_weekly_batch(batch)

        # The model uses submitted values in memory.
        # This call does not write to MongoDB.
        prediction = (
            generate_eight_input_forecast_preview(
                batch
            )
        )

        # This service atomically commits the input
        # batch and the associated forecast revision.
        committed = commit_generated_forecast(
            batch,
            prediction,
        )

        return format_generated_forecast_response(
            committed,
            prediction,
        )

    except HTTPException:
        raise

    except (
        ValueError,
        ValidationError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        logger.exception(
            "Production forecasting artifact missing."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The production forecasting model "
                "is currently unavailable."
            ),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Forecast generation MongoDB error "
            "for %s.",
            origin,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The forecast could not be saved. "
                "Check the current forecast history "
                "before trying again."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Forecast generation failed for %s.",
            origin,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast generation could not be "
                "completed. The request was not "
                "confirmed as saved."
            ),
        ) from exc

    finally:
        release_generation_slot(origin)


@router.post(
    "/observations/forecast/retry"
)
def retry_eight_input_forecast(
    request: ForecastRetryRequest,
):
    """
    Retry an existing saved week through the
    legacy forecast finalization service.

    Observations are read from MongoDB.
    They are not changed by this endpoint.
    """

    try:
        batch = load_saved_forecast_batch(
            request.week_start_date
        )

        result = (
            generate_and_finalize_saved_week(
                batch
            )
        )

        return {
            **format_forecast_response(
                result
            ),
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "week_start_date": (
                request.week_start_date.isoformat()
            ),
            "weekly_data_modified": False,
        }

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except PyMongoError as exc:
        logger.exception(
            "Forecast retry MongoDB error."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast retry could not be "
                "confirmed. Check forecast history."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Forecast retry failed for %s.",
            request.week_start_date,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The forecast could not be completed. "
                "Saved weekly observations were not "
                "changed."
            ),
        ) from exc


@router.patch("/observations")
def correct_eight_input_observation(
    correction: FactoryObservationCorrection,
):
    """
    Reserved endpoint for controlled corrections.

    No database writes occur through this endpoint
    until verified operator authorization exists.
    """

    require_corrections_enabled()
