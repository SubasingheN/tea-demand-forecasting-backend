
"""
Versioned forecast storage for Tea Demand Forecasting.

Keeps every regenerated four-week demand forecast.

Existing original forecasts remain in:
    eight_input_final_forecasts

Regenerated forecasts are stored in:
    eight_input_forecast_revisions

Rules:
- Exactly five tea grades per revision.
- Four forecast weeks per tea grade.
- Same reporting-week origin for every grade.
- Production nine-input TFT model only.
- Every revision retains an input snapshot.
- Revision numbers increase for each reporting week.
- All five forecast documents are saved atomically.
- Existing finalized forecasts are never overwritten.
- The original historical dataset is never modified.

This module does not generate forecasts or edit observations.
No database writes occur during module import.
"""

import hashlib
import json

from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_database

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_batch_storage_service import (
    validate_weekly_batch,
)

from app.services.eight_input_forecast_storage_service import (
    COLLECTION_NAME as ORIGINAL_COLLECTION_NAME,
    MODEL_NAME,
    MODEL_VERSION,
    TEA_GRADES,
    normalize_forecast,
    serialize_forecast_document,
)


REVISION_COLLECTION_NAME = (
    "eight_input_forecast_revisions"
)

REVISION_INDEX_NAME = (
    "unique_eight_input_forecast_revision"
)

FORECAST_HORIZON = 4


def get_revision_collection():
    """
    Return the forecast revision collection.
    """

    return get_database()[REVISION_COLLECTION_NAME]


def initialize_forecast_revision_storage():
    """
    Ensure that a grade cannot have two forecast
    documents with the same origin and revision.
    """

    collection = get_revision_collection()

    return collection.create_index(
        [
            ("forecast_origin", ASCENDING),
            ("tea_type", ASCENDING),
            ("revision", ASCENDING),
        ],
        unique=True,
        name=REVISION_INDEX_NAME,
    )


def build_input_snapshot(
    batch: EightInputWeeklyBatch,
):
    """
    Build a validated, reproducible snapshot
    of the nine inputs used for forecasting.
    """

    observations = validate_weekly_batch(batch)

    snapshot = {
        "week_start_date": (
            batch.week_start_date.isoformat()
        ),
        "records": [
            {
                "tea_type": observation["tea_type"],
                "demand_kg": observation["demand_kg"],
                "production_kg": observation["production_kg"],
                "inventory_kg": observation["inventory_kg"],
                "usd_lkr_monday_rate": (
                    observation["usd_lkr_monday_rate"]
                ),
                "usd_lkr_sunday_rate": (
                    observation["usd_lkr_sunday_rate"]
                ),
                "festival_flag": (
                    observation["festival_flag"]
                ),
                "auction_price_high_lkr": (
                    observation["auction_price_high_lkr"]
                ),
                "auction_price_medium_lkr": (
                    observation["auction_price_medium_lkr"]
                ),
                "auction_price_low_lkr": (
                    observation["auction_price_low_lkr"]
                ),
            }
            for observation in observations
        ],
    }

    return snapshot


def calculate_input_fingerprint(snapshot):
    """
    Calculate a stable SHA-256 fingerprint for
    one complete weekly input snapshot.
    """

    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def validate_revision_result(result, snapshot):
    """
    Confirm that the forecast result matches the
    reporting week and five tea grades.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "A complete forecasting result is required."
        )

    if result.get("model") != MODEL_NAME:
        raise ValueError(
            "Forecast model does not match production."
        )

    if result.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Forecast model version does not match production."
        )

    origin = snapshot["week_start_date"]

    if result.get("forecast_origin") != origin:
        raise ValueError(
            "Forecast reporting week does not match "
            "the submitted weekly observations."
        )

    selected = result.get("selected_grades")

    if (
        not isinstance(selected, list)
        or len(selected) != len(TEA_GRADES)
        or len(set(selected)) != len(TEA_GRADES)
        or set(selected) != set(TEA_GRADES)
    ):
        raise ValueError(
            "A forecast revision must contain "
            "all five tea grades."
        )

    forecasts = result.get("forecasts")

    if not isinstance(forecasts, dict):
        raise ValueError(
            "Forecast results must be grouped "
            "by tea grade."
        )

    if set(forecasts) != set(TEA_GRADES):
        raise ValueError(
            "Forecast results must contain "
            "exactly the five supported tea grades."
        )

    if result.get("forecast_horizon_weeks") != (
        FORECAST_HORIZON
    ):
        raise ValueError(
            "Forecast revision must cover four weeks."
        )

    if result.get("forecast_type") != (
        "latest_completed_week_forecast"
    ):
        raise ValueError(
            "Only an operational forecast may be "
            "stored as a factory forecast revision."
        )


def get_complete_original_set(
    origin,
    *,
    session=None,
):
    """
    Check whether five original finalized
    forecast records exist for this origin.

    An incomplete set is not safe to revise.
    """

    collection = get_database()[
        ORIGINAL_COLLECTION_NAME
    ]

    documents = list(
        collection.find(
            {"forecast_origin": origin},
            {"_id": 0},
            session=session,
        )
    )

    if not documents:
        return {}

    by_grade = {}

    for document in documents:
        grade = document.get("tea_type")

        if grade not in TEA_GRADES:
            raise ValueError(
                "Unexpected tea grade in original "
                "forecast records."
            )

        if grade in by_grade:
            raise ValueError(
                "Duplicate original forecast records "
                "were found."
            )

        if document.get("finalized") is not True:
            raise ValueError(
                "An original forecast has not "
                "been finalized."
            )

        by_grade[grade] = document

    if set(by_grade) != set(TEA_GRADES):
        raise ValueError(
            "Original forecasts are incomplete. "
            "Review the reporting week's history "
            "before generating revisions."
        )

    return by_grade


def get_existing_revision_number(
    origin,
    *,
    session=None,
):
    """
    Find the latest complete revision number.

    Verify that every existing revision contains
    exactly one record for all five tea grades.
    """

    collection = get_revision_collection()

    documents = list(
        collection.find(
            {"forecast_origin": origin},
            {
                "_id": 0,
                "tea_type": 1,
                "revision": 1,
                "finalized": 1,
            },
            session=session,
        )
    )

    if not documents:
        return 0

    revisions = {}

    for document in documents:
        revision = document.get("revision")
        grade = document.get("tea_type")

        if (
            type(revision) is not int
            or revision < 1
        ):
            raise ValueError(
                "Invalid forecast revision number."
            )

        if grade not in TEA_GRADES:
            raise ValueError(
                "Unexpected tea grade in revisions."
            )

        if document.get("finalized") is not True:
            raise ValueError(
                "An existing forecast revision "
                "is not finalized."
            )

        revisions.setdefault(revision, set())

        if grade in revisions[revision]:
            raise ValueError(
                "Duplicate grade in forecast revisions."
            )

        revisions[revision].add(grade)

    for revision, grades in revisions.items():
        if grades != set(TEA_GRADES):
            raise ValueError(
                f"Forecast revision {revision} "
                "is incomplete."
            )

    latest = max(revisions)

    expected = set(range(1, latest + 1))

    if set(revisions) != expected:
        raise ValueError(
            "Forecast revision history has "
            "missing revision numbers."
        )

    return latest


def save_forecast_revision_atomic(
    result,
    batch: EightInputWeeklyBatch,
):
    """
    Save a new forecast revision for all five grades.

    The original forecast must already exist.
    The next revision number is assigned inside
    the MongoDB transaction.

    The original collection is never modified.

    This function must be called only after
    forecast generation against verified inputs.
    """

    snapshot = build_input_snapshot(batch)

    validate_revision_result(
        result,
        snapshot,
    )

    origin = snapshot["week_start_date"]

    fingerprint = calculate_input_fingerprint(
        snapshot
    )

    # Normalize all five forecast records before
    # attempting any database writes.
    normalized = [
        normalize_forecast(result, grade)
        for grade in TEA_GRADES
    ]

    initialize_forecast_revision_storage()

    database = get_database()
    collection = database[
        REVISION_COLLECTION_NAME
    ]

    now = datetime.now(timezone.utc)

    def insert_transaction(session):
        original = get_complete_original_set(
            origin,
            session=session,
        )

        if not original:
            raise ValueError(
                "The original finalized forecast "
                "must exist before creating revisions."
            )

        latest = get_existing_revision_number(
            origin,
            session=session,
        )

        next_revision = latest + 1

        documents = []

        for record in normalized:
            documents.append(
                {
                    **record,
                    "revision": next_revision,
                    "revision_type": (
                        "regenerated_forecast"
                    ),
                    "input_fingerprint": (
                        fingerprint
                    ),
                    "input_snapshot": snapshot,
                    "revision_created_at": now,
                    "original_forecast_preserved": True,
                }
            )

        inserted = collection.insert_many(
            documents,
            ordered=True,
            session=session,
        )

        return {
            "revision": next_revision,
            "saved_count": len(
                inserted.inserted_ids
            ),
        }

    try:
        with database.client.start_session() as session:
            saved = session.with_transaction(
                insert_transaction
            )

    except DuplicateKeyError as exc:
        raise ValueError(
            "Another forecast revision was saved "
            "for this reporting week. "
            "Refresh and retry with current data."
        ) from exc

    return {
        "status": "success",
        "forecast_origin": origin,
        "revision": saved["revision"],
        "saved": True,
        "finalized": True,
        "saved_count": saved["saved_count"],
        "saved_grades": list(TEA_GRADES),
        "input_fingerprint": fingerprint,
        "collection": REVISION_COLLECTION_NAME,
        "atomic": True,
        "message": (
            "All five forecast revisions "
            "were saved successfully."
        ),
    }


def get_latest_forecast_revision(
    origin,
):
    """
    Retrieve the newest complete forecast revision.

    Returns None when no revisions exist.
    """

    collection = get_revision_collection()

    latest = get_existing_revision_number(
        origin
    )

    if latest == 0:
        return None

    documents = list(
        collection.find(
            {
                "forecast_origin": origin,
                "revision": latest,
                "finalized": True,
            },
            {"_id": 0},
        )
    )

    by_grade = {
        document["tea_type"]: document
        for document in documents
    }

    if (
        len(documents) != len(TEA_GRADES)
        or set(by_grade) != set(TEA_GRADES)
    ):
        raise ValueError(
            "Latest forecast revision is incomplete."
        )

    return {
        "forecast_origin": origin,
        "revision": latest,
        "forecasts": {
            grade: serialize_forecast_document(
                by_grade[grade]
            )
            for grade in TEA_GRADES
        },
    }


def list_forecast_revisions(
    origin,
):
    """
    Retrieve every saved regenerated forecast
    revision for one reporting week.

    The original revision-zero forecasts are
    still stored in the original collection.
    """

    collection = get_revision_collection()

    get_existing_revision_number(origin)

    documents = list(
        collection.find(
            {"forecast_origin": origin},
            {"_id": 0},
        ).sort(
            [
                ("revision", DESCENDING),
                ("tea_type", ASCENDING),
            ]
        )
    )

    return [
        serialize_forecast_document(document)
        for document in documents
    ]


def get_current_forecasts(
    origin,
):
    """
    Retrieve the current forecast set.

    Prefer the newest regenerated revision.
    Otherwise return the original finalized set.
    """

    latest = get_latest_forecast_revision(
        origin
    )

    if latest is not None:
        return {
            "forecast_origin": origin,
            "revision": latest["revision"],
            "source": "forecast_revision",
            "forecasts": latest["forecasts"],
        }

    original = get_complete_original_set(
        origin
    )

    if not original:
        return None

    return {
        "forecast_origin": origin,
        "revision": 0,
        "source": "original_forecast",
        "forecasts": {
            grade: serialize_forecast_document(
                original[grade]
            )
            for grade in TEA_GRADES
        },
    }
