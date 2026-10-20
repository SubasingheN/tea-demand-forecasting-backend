
"""
Atomic weekly batch storage for Tea Demand Forecasting.

Uses the nine-input production forecasting schema.

Supports:
- First-time saving of all five factory tea-grade observations.
- Atomic updates to an existing complete reporting week.
- Editing only the latest completed reporting week.
- Protection of the original historical dataset.
- Validation of all nine historical inputs.
- Identical shared weekly values across grades.
- Continuous weekly observation history.
- Preservation of previous input values and revisions.
- MongoDB transaction rollback when a write fails.

Existing insert-only behavior remains unchanged.

Forecast generation and forecast revision storage are
handled by separate services.

No database writes occur on module import.
"""

from datetime import date, datetime, timedelta, timezone

from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_database

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_observation_storage_service import (
    COLLECTION_NAME,
    DATA_SOURCE,
    EIGHT_INPUT_FIELDS,
    MODEL_NAME,
    SHARED_WEEKLY_FIELDS,
    TEA_GRADES,
    get_factory_history_bounds,
    initialize_observation_storage,
    normalize_observation,
    normalize_numeric,
    shared_values_match,
    validate_latest_completed_week,
)


def validate_weekly_batch(
    batch: EightInputWeeklyBatch,
):
    """
    Validate all five grade records before database writes.

    Return normalized observations in standard grade order.
    """

    if not isinstance(batch, EightInputWeeklyBatch):
        raise ValueError(
            "A valid nine-input weekly batch is required."
        )

    submitted_grades = [
        record.tea_type
        for record in batch.records
    ]

    expected_grades = set(TEA_GRADES)

    if set(submitted_grades) != expected_grades:
        missing = sorted(
            expected_grades - set(submitted_grades)
        )

        extra = sorted(
            set(submitted_grades) - expected_grades
        )

        details = []

        if missing:
            details.append(
                "Missing: " + ", ".join(missing)
            )

        if extra:
            details.append(
                "Unsupported: " + ", ".join(extra)
            )

        raise ValueError(
            "All five tea grades must be provided. "
            + "; ".join(details)
        )

    if len(submitted_grades) != len(set(submitted_grades)):
        raise ValueError(
            "Duplicate tea grades are not allowed."
        )

    week_start = validate_latest_completed_week(
        batch.week_start_date
    ).isoformat()

    normalized_by_grade = {}

    for record in batch.records:
        observation = normalize_observation(
            {
                "week_start_date": week_start,
                **record.model_dump(),
            }
        )

        normalized_by_grade[
            observation["tea_type"]
        ] = observation

    normalized = [
        normalized_by_grade[grade]
        for grade in TEA_GRADES
    ]

    reference = normalized[0]

    for observation in normalized[1:]:
        for field in SHARED_WEEKLY_FIELDS:
            if not shared_values_match(
                reference[field],
                observation[field],
                field,
            ):
                raise ValueError(
                    f"{field} must have the same "
                    "value across all five tea grades."
                )

    for observation in normalized:
        observation[
            "usd_lkr_exchange_rate_range"
        ] = (
            f"{observation['usd_lkr_monday_rate']:.4f}"
            "-"
            f"{observation['usd_lkr_sunday_rate']:.4f}"
        )

    return normalized


def validate_protected_boundary(week_start):
    """
    Prevent modifications to the historical CSV period.

    This function does not modify the original dataset.
    """

    bounds = get_factory_history_bounds()
    latest_csv_week = bounds["latest_week"]

    if week_start <= latest_csv_week:
        raise ValueError(
            "The selected reporting week belongs "
            "to the protected historical dataset."
        )

    return latest_csv_week


def check_weekly_batch_for_storage(
    observations,
    collection,
    session,
):
    """
    Validate a new weekly batch before insertion.

    Existing records are rejected by this function.
    """

    if len(observations) != len(TEA_GRADES):
        raise ValueError(
            "Exactly five observations are required."
        )

    week_start = observations[0]["week_start_date"]

    if any(
        item["week_start_date"] != week_start
        for item in observations
    ):
        raise ValueError(
            "All observations must belong "
            "to the same reporting week."
        )

    validate_latest_completed_week(week_start)

    latest_csv_week = validate_protected_boundary(
        week_start
    )

    existing = collection.find_one(
        {
            "week_start_date": week_start,
        },
        {
            "_id": 1,
            "tea_type": 1,
        },
        session=session,
    )

    if existing is not None:
        raise ValueError(
            "Factory observations already exist "
            f"for {week_start}. "
            "Use the weekly batch update workflow "
            "to edit existing observations."
        )

    previous_week = (
        date.fromisoformat(week_start)
        - timedelta(weeks=1)
    ).isoformat()

    if previous_week < latest_csv_week:
        raise ValueError(
            "The selected week does not correctly "
            "continue the factory observation history."
        )

    if previous_week != latest_csv_week:
        previous_records = list(
            collection.find(
                {
                    "week_start_date": previous_week,
                },
                {
                    "_id": 0,
                    "tea_type": 1,
                },
                session=session,
            )
        )

        preceding_grades = {
            record["tea_type"]
            for record in previous_records
        }

        missing_grades = [
            grade
            for grade in TEA_GRADES
            if grade not in preceding_grades
        ]

        if missing_grades:
            raise ValueError(
                "The previous reporting week "
                f"({previous_week}) is incomplete. "
                "Missing: "
                + ", ".join(missing_grades)
            )

    reference = observations[0]

    for observation in observations[1:]:
        for field in SHARED_WEEKLY_FIELDS:
            if not shared_values_match(
                reference[field],
                observation[field],
                field,
            ):
                raise ValueError(
                    f"Conflicting {field} values "
                    "within the submitted batch."
                )

    return {
        "week_start_date": week_start,
        "previous_week": previous_week,
        "historical_dataset_latest_week": (
            latest_csv_week
        ),
    }


def save_eight_input_observation_batch(
    batch: EightInputWeeklyBatch,
):
    """
    Atomically insert a new completed reporting week.

    Existing observations are never overwritten here.
    """

    observations = validate_weekly_batch(batch)

    initialize_observation_storage()

    database = get_database()
    collection = database[COLLECTION_NAME]

    now = datetime.now(timezone.utc)

    documents = []

    for observation in observations:
        documents.append(
            {
                **observation,
                "created_at": now,
                "source": DATA_SOURCE,
                "record_type": "factory_observation",
                "model_context": MODEL_NAME,
                "data_status": "submitted_completed_week",
                "revision": 0,
                "batch_revision": 0,
            }
        )

    def insert_transaction(session):
        check_weekly_batch_for_storage(
            observations,
            collection,
            session,
        )

        result = collection.insert_many(
            documents,
            ordered=True,
            session=session,
        )

        return [
            str(value)
            for value in result.inserted_ids
        ]

    try:
        with database.client.start_session() as session:
            inserted_ids = session.with_transaction(
                insert_transaction
            )

    except DuplicateKeyError as exc:
        raise ValueError(
            "One or more observations already "
            "exist for the selected reporting week. "
            "The batch was not fully saved."
        ) from exc

    return {
        "status": "saved",
        "week_start_date": (
            observations[0]["week_start_date"]
        ),
        "saved": True,
        "saved_count": len(inserted_ids),
        "saved_grades": list(TEA_GRADES),
        "observation_ids": inserted_ids,
        "atomic": True,
        "forecast_generated": False,
        "batch_revision": 0,
        "message": (
            "All five tea-grade observations "
            "were saved successfully."
        ),
    }


def check_weekly_batch_for_update(
    observations,
    collection,
    session,
):
    """
    Verify an existing complete week can be edited.

    Updates are restricted to the latest completed week,
    beyond the historical dataset boundary.
    """

    if len(observations) != len(TEA_GRADES):
        raise ValueError(
            "Exactly five observations are required."
        )

    week_start = observations[0]["week_start_date"]

    if any(
        item["week_start_date"] != week_start
        for item in observations
    ):
        raise ValueError(
            "All submitted observations must belong "
            "to the same reporting week."
        )

    validate_latest_completed_week(week_start)

    validate_protected_boundary(week_start)

    documents = list(
        collection.find(
            {
                "week_start_date": week_start,
            },
            session=session,
        )
    )

    if len(documents) != len(TEA_GRADES):
        raise ValueError(
            "A complete saved reporting week "
            "with all five tea grades is required "
            "before editing."
        )

    existing_by_grade = {}

    for document in documents:
        grade = document.get("tea_type")

        if grade not in TEA_GRADES:
            raise ValueError(
                "Stored observations contain "
                "an unsupported tea grade."
            )

        if grade in existing_by_grade:
            raise ValueError(
                "Duplicate stored tea-grade observations "
                "were found for this reporting week."
            )

        if document.get("source") != DATA_SOURCE:
            raise ValueError(
                "Only factory-entered observations "
                "can be edited."
            )

        if document.get("record_type") != (
            "factory_observation"
        ):
            raise ValueError(
                "The selected records cannot be edited."
            )

        missing_fields = [
            field
            for field in EIGHT_INPUT_FIELDS
            if field not in document
        ]

        if missing_fields:
            raise ValueError(
                f"{grade}: stored observations use an "
                "incomplete input format. "
                "Migration is required before editing. "
                "Missing: "
                + ", ".join(missing_fields)
            )

        existing_by_grade[grade] = document

    if set(existing_by_grade) != set(TEA_GRADES):
        raise ValueError(
            "Stored observations do not contain "
            "all five required tea grades."
        )

    reference = observations[0]

    for observation in observations[1:]:
        for field in SHARED_WEEKLY_FIELDS:
            if not shared_values_match(
                reference[field],
                observation[field],
                field,
            ):
                raise ValueError(
                    f"{field} differs between tea grades."
                )

    return existing_by_grade


def find_changed_fields(existing, updated):
    """
    Compare stored and submitted nine-input values.

    Return the previous and new value of each change.
    """

    changed = {}

    for field in EIGHT_INPUT_FIELDS:
        previous = normalize_numeric(
            existing[field],
            field,
        )

        new_value = updated[field]

        if previous != new_value:
            changed[field] = {
                "previous": previous,
                "new": new_value,
            }

    old_range = existing.get(
        "usd_lkr_exchange_rate_range"
    )

    new_range = updated[
        "usd_lkr_exchange_rate_range"
    ]

    if old_range != new_range:
        changed[
            "usd_lkr_exchange_rate_range"
        ] = {
            "previous": old_range,
            "new": new_range,
        }

    return changed


def update_eight_input_observation_batch(
    batch: EightInputWeeklyBatch,
    *,
    reason="Updated from factory weekly data entry",
):
    """
    Atomically update all five existing grade records.

    The complete previous values are preserved in
    the correction history of each grade.

    Only the latest completed reporting week can
    be edited.

    This function does not modify saved forecasts.
    It must be coordinated with forecast revision
    generation by the API workflow.

    IMPORTANT:
    Do not expose this function through an unrestricted
    public correction endpoint. The API layer must
    enforce the permitted latest-week workflow.
    """

    observations = validate_weekly_batch(batch)

    if not isinstance(reason, str):
        raise ValueError(
            "An update reason is required."
        )

    reason = reason.strip()

    if not 3 <= len(reason) <= 500:
        raise ValueError(
            "The update reason must contain "
            "between 3 and 500 characters."
        )

    initialize_observation_storage()

    database = get_database()
    collection = database[COLLECTION_NAME]

    week_start = observations[0]["week_start_date"]

    normalized_by_grade = {
        observation["tea_type"]: observation
        for observation in observations
    }

    def update_transaction(session):
        existing_by_grade = (
            check_weekly_batch_for_update(
                observations,
                collection,
                session,
            )
        )

        changes_by_grade = {
            grade: find_changed_fields(
                existing_by_grade[grade],
                normalized_by_grade[grade],
            )
            for grade in TEA_GRADES
        }

        changed_grades = [
            grade
            for grade in TEA_GRADES
            if changes_by_grade[grade]
        ]

        current_batch_revisions = {
            int(
                existing_by_grade[grade].get(
                    "batch_revision",
                    existing_by_grade[grade].get(
                        "revision",
                        0,
                    ),
                )
            )
            for grade in TEA_GRADES
        }

        if len(current_batch_revisions) != 1:
            raise ValueError(
                "Saved observations have inconsistent "
                "batch revision numbers. "
                "Review the records before editing."
            )

        previous_batch_revision = next(
            iter(current_batch_revisions)
        )

        if not changed_grades:
            return {
                "status": "unchanged",
                "updated_count": 0,
                "updated_grades": [],
                "batch_revision": previous_batch_revision,
                "previous_batch_revision": (
                    previous_batch_revision
                ),
                "write_performed": False,
            }

        next_batch_revision = (
            previous_batch_revision + 1
        )

        now = datetime.now(timezone.utc)

        for grade in TEA_GRADES:
            existing = existing_by_grade[grade]
            updated = normalized_by_grade[grade]

            old_revision = int(
                existing.get("revision", 0)
            )

            # Check the exact record revision to detect
            # conflicting concurrent updates.
            revision_filter = {
                "_id": existing["_id"],
            }

            if "revision" in existing:
                revision_filter["revision"] = (
                    old_revision
                )
            else:
                revision_filter["revision"] = {
                    "$exists": False,
                }

            if "batch_revision" in existing:
                revision_filter["batch_revision"] = (
                    previous_batch_revision
                )
            else:
                revision_filter["batch_revision"] = {
                    "$exists": False,
                }

            previous_values = {
                field: normalize_numeric(
                    existing[field],
                    field,
                )
                for field in EIGHT_INPUT_FIELDS
            }

            update_values = {
                field: updated[field]
                for field in EIGHT_INPUT_FIELDS
            }

            update_values[
                "usd_lkr_exchange_rate_range"
            ] = updated[
                "usd_lkr_exchange_rate_range"
            ]

            update_values["updated_at"] = now
            update_values["data_status"] = (
                "updated_completed_week"
            )
            update_values["batch_revision"] = (
                next_batch_revision
            )

            result = collection.update_one(
                revision_filter,
                {
                    "$set": update_values,
                    "$inc": {
                        "revision": 1,
                    },
                    "$push": {
                        "correction_history": {
                            "revision": old_revision + 1,
                            "batch_revision": (
                                next_batch_revision
                            ),
                            "updated_at": now,
                            "reason": reason,
                            "previous_values": (
                                previous_values
                            ),
                            "new_values": {
                                field: updated[field]
                                for field in EIGHT_INPUT_FIELDS
                            },
                            "changed_fields": (
                                changes_by_grade[grade]
                            ),
                        },
                    },
                },
                session=session,
            )

            if result.matched_count != 1:
                raise ValueError(
                    "The saved weekly observations "
                    "changed during this update. "
                    "Refresh the page and try again."
                )

        return {
            "status": "updated",
            "updated_count": len(changed_grades),
            "updated_grades": changed_grades,
            "batch_revision": next_batch_revision,
            "previous_batch_revision": (
                previous_batch_revision
            ),
            "write_performed": True,
        }

    with database.client.start_session() as session:
        transaction_result = session.with_transaction(
            update_transaction
        )

    return {
        **transaction_result,
        "week_start_date": week_start,
        "saved": True,
        "saved_count": len(TEA_GRADES),
        "saved_grades": list(TEA_GRADES),
        "atomic": True,
        "forecast_generated": False,
        "message": (
            "Weekly observations updated successfully. "
            "Forecast revision generation is required."
            if transaction_result["write_performed"]
            else
            "No input changes were necessary."
        ),
    }
