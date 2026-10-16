
"""
Six-input factory observation API.

Endpoints:
    GET  /factory/six-input/status
    GET  /factory/six-input/observations
    POST /factory/six-input/observations

Purpose:
    Store genuine, completed-week factory
    observations safely in MongoDB.

Safety rules:
    - Protect original factory CSV observations.
    - Accept one to five selected tea grades.
    - Reject unfinished reporting weeks.
    - Reject duplicate grade/week observations.
    - Require consecutive weekly grade history.
    - Enforce shared USD/LKR and festival values.
    - Use transactions for atomic batch inserts.
    - Never overwrite saved factory observations.

The existing TFT model and original research
artifacts remain unchanged.
"""

import logging

from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from pymongo.errors import (
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
)

from app.db.mongodb import get_database

from app.schemas.six_input_weekly import (
    SixInputWeeklyBatch,
    TeaGrade,
)

from app.services.six_input_observation_storage_service import (
    COLLECTION_NAME,
    get_factory_history_bounds,
    list_six_input_observations,
    normalize_observation,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/factory/six-input",
    tags=["Six-Input Factory Observations"],
)

WEEK_METADATA_COLLECTION = (
    "six_input_factory_week_metadata"
)

UNIQUE_INDEX_NAME = (
    "unique_six_input_week_grade"
)


def get_storage_collections():
    """
    Retrieve MongoDB database and relevant
    collections without creating records.
    """

    database = get_database()

    return (
        database,
        database[COLLECTION_NAME],
        database[WEEK_METADATA_COLLECTION],
    )


def verify_storage_index(collection):
    """
    Require the database-level unique index.

    Refuse writes if duplicate protection has
    not been initialized correctly.
    """

    indexes = list(collection.list_indexes())

    for index in indexes:
        if index.get("name") != UNIQUE_INDEX_NAME:
            continue

        expected_keys = [
            ("week_start_date", 1),
            ("tea_type", 1),
        ]

        if (
            index.get("unique") is True
            and list(index["key"].items())
            == expected_keys
        ):
            return True

    raise RuntimeError(
        "The six-input unique grade/week index "
        "is missing or incorrectly configured."
    )


def validate_batch_for_storage(
    batch: SixInputWeeklyBatch,
):
    """
    Normalize the complete batch before writing.

    Returns:
        week_start_date
        previous_week
        normalized documents

    Does not write to MongoDB.
    """

    week_start = batch.week_start_date

    previous_week = (
        week_start - timedelta(weeks=1)
    )

    documents = []

    for record in batch.records:
        raw = {
            "week_start_date": week_start,
            **record.model_dump(),
        }

        normalized = normalize_observation(raw)

        documents.append(normalized)

    if len(documents) != len(
        {
            document["tea_type"]
            for document in documents
        }
    ):
        raise ValueError(
            "Duplicate tea grades are not allowed."
        )

    if not documents:
        raise ValueError(
            "At least one tea grade is required."
        )

    return {
        "week_start_date": week_start.isoformat(),
        "previous_week": previous_week.isoformat(),
        "documents": documents,
        "selected_grades": batch.selected_grades,
        "usd_lkr_rate": documents[0][
            "usd_lkr_rate"
        ],
        "festival_flag": documents[0][
            "festival_flag"
        ],
    }


def verify_protected_history(
    submission,
    history_bounds,
):
    """
    Ensure new observations are strictly newer
    than the original protected CSV dataset.
    """

    latest_csv_week = history_bounds[
        "latest_week"
    ]

    week_start = submission[
        "week_start_date"
    ]

    if week_start <= latest_csv_week:
        raise ValueError(
            "The selected reporting week belongs "
            "to the protected factory CSV. "
            "Historical observations cannot "
            "be replaced."
        )

    return latest_csv_week


def verify_previous_grade_weeks(
    collection,
    submission,
    csv_latest_week,
    session,
):
    """
    Each submitted grade needs its preceding
    reporting week.

    The final CSV week acts as the predecessor
    of the first newly stored week.

    MongoDB provides predecessor records for
    all subsequent reporting weeks.
    """

    previous_week = submission[
        "previous_week"
    ]

    if previous_week == csv_latest_week:
        return

    for document in submission["documents"]:
        grade = document["tea_type"]

        previous_record = collection.find_one(
            {
                "week_start_date": previous_week,
                "tea_type": grade,
            },
            {"_id": 1},
            session=session,
        )

        if previous_record is None:
            raise ValueError(
                f"{grade} has no recorded observation "
                f"for the preceding week "
                f"{previous_week}. "
                "Weekly grade history must "
                "remain continuous."
            )


def verify_no_duplicate_observations(
    collection,
    submission,
    session,
):
    """
    Reject previously saved observations.

    The unique MongoDB index provides another
    layer of duplicate protection.
    """

    week_start = submission[
        "week_start_date"
    ]

    grades = submission[
        "selected_grades"
    ]

    existing = list(
        collection.find(
            {
                "week_start_date": week_start,
                "tea_type": {
                    "$in": grades,
                },
            },
            {
                "_id": 0,
                "tea_type": 1,
            },
            session=session,
        )
    )

    if existing:
        existing_grades = sorted(
            {
                record["tea_type"]
                for record in existing
            }
        )

        raise ValueError(
            "Observations already exist for "
            f"{week_start}: "
            + ", ".join(existing_grades)
            + ". Existing observations cannot "
            "be overwritten."
        )


def lock_and_verify_week_metadata(
    metadata_collection,
    submission,
    session,
):
    """
    Coordinate concurrent submissions to
    the same reporting week.

    The transaction updates one shared metadata
    document, producing a write conflict if
    competing transactions modify that document.

    The metadata also stores the week's common
    exchange rate and festival indicator.
    """

    week_start = submission[
        "week_start_date"
    ]

    exchange_rate = float(
        submission["usd_lkr_rate"]
    )

    festival_flag = int(
        submission["festival_flag"]
    )

    metadata_collection.update_one(
        {
            "_id": week_start,
        },
        {
            "$setOnInsert": {
                "usd_lkr_rate": exchange_rate,
                "festival_flag": festival_flag,
                "created_at": datetime.now(
                    timezone.utc
                ),
            },
            "$inc": {
                "revision": 1,
            },
        },
        upsert=True,
        session=session,
    )

    metadata = metadata_collection.find_one(
        {
            "_id": week_start,
        },
        session=session,
    )

    if metadata is None:
        raise RuntimeError(
            "Weekly metadata was not available "
            "inside the transaction."
        )

    recorded_rate = float(
        metadata["usd_lkr_rate"]
    )

    recorded_festival = int(
        metadata["festival_flag"]
    )

    if abs(
        recorded_rate - exchange_rate
    ) > 0.001:
        raise ValueError(
            "The submitted USD/LKR exchange rate "
            "conflicts with the rate already "
            "recorded for this week."
        )

    if recorded_festival != festival_flag:
        raise ValueError(
            "The submitted festival flag conflicts "
            "with the classification already "
            "recorded for this week."
        )


def verify_saved_week_values(
    collection,
    submission,
    session,
):
    """
    Cross-check previously stored grades.

    This also detects inconsistencies in older
    documents that may exist independently of
    the weekly metadata collection.
    """

    week_start = submission[
        "week_start_date"
    ]

    expected_rate = float(
        submission["usd_lkr_rate"]
    )

    expected_festival = int(
        submission["festival_flag"]
    )

    existing_records = collection.find(
        {
            "week_start_date": week_start,
        },
        {
            "_id": 0,
            "tea_type": 1,
            "usd_lkr_rate": 1,
            "festival_flag": 1,
        },
        session=session,
    )

    for record in existing_records:
        if abs(
            float(record["usd_lkr_rate"])
            - expected_rate
        ) > 0.001:
            raise ValueError(
                "Previously saved grade "
                f"{record['tea_type']} has a "
                "conflicting weekly exchange rate."
            )

        if int(
            record["festival_flag"]
        ) != expected_festival:
            raise ValueError(
                "Previously saved grade "
                f"{record['tea_type']} has a "
                "conflicting festival flag."
            )


@router.get("/status")
def get_six_input_factory_status():
    """
    Report the protected CSV range and the
    current MongoDB observation status.
    """

    try:
        database, collection, _ = (
            get_storage_collections()
        )

        bounds = get_factory_history_bounds()

        latest_saved = collection.find_one(
            {},
            {
                "_id": 0,
                "week_start_date": 1,
            },
            sort=[
                ("week_start_date", -1),
            ],
        )

        saved_count = (
            collection.count_documents({})
        )

        return {
            "status": "success",
            "database": database.name,
            "csv_first_week": bounds[
                "first_week"
            ],
            "csv_latest_week": bounds[
                "latest_week"
            ],
            "csv_records": bounds[
                "total_rows"
            ],
            "saved_observation_count": (
                saved_count
            ),
            "latest_saved_week": (
                latest_saved["week_start_date"]
                if latest_saved is not None
                else None
            ),
            "model_retrained": False,
        }

    except Exception as error:
        logger.exception(
            "Failed to retrieve factory status."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to retrieve factory "
                "observation status."
            ),
        ) from error


@router.get("/observations")
def get_six_input_factory_observations(
    grade: TeaGrade | None = None,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
):
    """
    Retrieve saved observations without
    modifying database records.
    """

    try:
        records = (
            list_six_input_observations(
                grade=grade,
                limit=limit,
            )
        )

        return {
            "status": "success",
            "count": len(records),
            "records": records,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except PyMongoError as error:
        logger.exception(
            "Failed to list observations."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Factory observation storage "
                "is unavailable."
            ),
        ) from error


@router.post(
    "/observations",
    status_code=201,
)
def save_six_input_factory_observations(
    batch: SixInputWeeklyBatch,
):
    """
    Atomically save observations for
    one to five selected tea grades.

    All validation and insert operations
    occur inside a MongoDB transaction
    where database state must be consistent.

    A failure aborts the entire transaction.

    Requires MongoDB transaction support
    and the configured unique index.
    """

    try:
        submission = (
            validate_batch_for_storage(
                batch
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    try:
        database, collection, metadata_collection = (
            get_storage_collections()
        )

        verify_storage_index(
            collection
        )

        bounds = get_factory_history_bounds()

        csv_latest = verify_protected_history(
            submission,
            bounds,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except (RuntimeError, PyMongoError) as error:
        logger.exception(
            "Factory observation storage "
            "preparation failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Factory storage is not ready "
                "for safe observation saving."
            ),
        ) from error

    documents = [
        dict(document)
        for document in submission["documents"]
    ]

    timestamp = datetime.now(
        timezone.utc
    )

    for document in documents:
        document["created_at"] = timestamp

        document["source"] = (
            "factory_weekly_manual_entry"
        )

    try:
        with database.client.start_session() as session:
            with session.start_transaction():

                # Serialize writes for the same week
                # and check shared metadata.
                lock_and_verify_week_metadata(
                    metadata_collection,
                    submission,
                    session,
                )

                # Detect existing grade records.
                verify_no_duplicate_observations(
                    collection,
                    submission,
                    session,
                )

                # Require the previous reporting week.
                verify_previous_grade_weeks(
                    collection,
                    submission,
                    csv_latest,
                    session,
                )

                # Check other grades already saved
                # for the same reporting week.
                verify_saved_week_values(
                    collection,
                    submission,
                    session,
                )

                # The unique grade/week index is
                # the final duplicate safeguard.
                collection.insert_many(
                    documents,
                    ordered=True,
                    session=session,
                )

    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except DuplicateKeyError as error:
        raise HTTPException(
            status_code=409,
            detail=(
                "One or more selected tea grades "
                "already have a saved observation "
                "for this reporting week."
            ),
        ) from error

    except OperationFailure as error:
        logger.exception(
            "MongoDB observation transaction failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The observation transaction could "
                "not be committed. Check stored "
                "records before retrying."
            ),
        ) from error

    except PyMongoError as error:
        logger.exception(
            "MongoDB factory observation saving failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Factory observation storage is "
                "temporarily unavailable. Check "
                "saved records before retrying."
            ),
        ) from error

    except RuntimeError as error:
        logger.exception(
            "Factory observation transaction "
            "validation failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Factory observation saving "
                "could not be completed safely."
            ),
        ) from error

    return {
        "status": "success",
        "saved": True,
        "week_start_date": submission[
            "week_start_date"
        ],
        "selected_grades": submission[
            "selected_grades"
        ],
        "saved_count": len(documents),
        "storage": COLLECTION_NAME,
        "message": (
            "Completed-week factory observations "
            "saved successfully to MongoDB."
        ),
    }
