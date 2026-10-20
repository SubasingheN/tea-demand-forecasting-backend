
"""
Coordinated weekly observation and forecast persistence.

This service commits the latest completed week's observations
and the associated five-grade forecast as one MongoDB transaction.

First generation:
    - Insert five observations.
    - Save five original finalized forecasts.
    - Use forecast revision 0.

Subsequent generations:
    - Update five observations when values changed.
    - Preserve previous observation values.
    - Insert five forecast revision documents.
    - Preserve all earlier finalized forecasts.

Predictions must already have been generated with the
submitted observations using the production TFT preview
function before calling this service.

No model inference occurs inside the transaction.
No database writes occur during module import.
"""

from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_database

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_batch_storage_service import (
    check_weekly_batch_for_storage,
    check_weekly_batch_for_update,
    find_changed_fields,
    validate_weekly_batch,
)

from app.services.eight_input_observation_storage_service import (
    COLLECTION_NAME as OBSERVATION_COLLECTION,
    DATA_SOURCE,
    EIGHT_INPUT_FIELDS,
    MODEL_NAME as OBSERVATION_MODEL,
    TEA_GRADES,
    initialize_observation_storage,
    normalize_numeric,
)

from app.services.eight_input_forecast_storage_service import (
    COLLECTION_NAME as ORIGINAL_FORECAST_COLLECTION,
    MODEL_NAME,
    MODEL_VERSION,
    initialize_forecast_storage,
    normalize_forecast,
)

from app.services.eight_input_forecast_revision_storage_service import (
    REVISION_COLLECTION_NAME,
    build_input_snapshot,
    calculate_input_fingerprint,
    get_complete_original_set,
    get_existing_revision_number,
    initialize_forecast_revision_storage,
    validate_revision_result,
)


def prepare_generation(
    batch: EightInputWeeklyBatch,
    forecast_result,
):
    """
    Validate all inputs and predictions before
    attempting any MongoDB modification.
    """

    observations = validate_weekly_batch(batch)

    snapshot = build_input_snapshot(batch)

    validate_revision_result(
        forecast_result,
        snapshot,
    )

    normalized_forecasts = [
        normalize_forecast(
            forecast_result,
            grade,
        )
        for grade in TEA_GRADES
    ]

    fingerprint = calculate_input_fingerprint(
        snapshot
    )

    return {
        "observations": observations,
        "forecasts": normalized_forecasts,
        "snapshot": snapshot,
        "fingerprint": fingerprint,
        "origin": snapshot["week_start_date"],
    }


def make_initial_observation(
    observation,
    now,
):
    """Build a new factory observation document."""

    return {
        **observation,
        "created_at": now,
        "source": DATA_SOURCE,
        "record_type": "factory_observation",
        "model_context": OBSERVATION_MODEL,
        "data_status": "submitted_completed_week",
        "revision": 0,
        "batch_revision": 0,
    }


def make_original_forecast(
    document,
    fingerprint,
    snapshot,
):
    """Build the original finalized forecast document."""

    return {
        **document,
        "revision": 0,
        "input_fingerprint": fingerprint,
        "input_snapshot": snapshot,
    }


def make_revision_forecast(
    document,
    revision,
    fingerprint,
    snapshot,
    now,
):
    """Build one regenerated forecast document."""

    return {
        **document,
        "revision": revision,
        "revision_type": "regenerated_forecast",
        "input_fingerprint": fingerprint,
        "input_snapshot": snapshot,
        "revision_created_at": now,
        "original_forecast_preserved": True,
    }


def update_existing_observations(
    observations,
    existing_by_grade,
    collection,
    session,
    now,
    revision,
):
    """
    Update the complete saved week inside an
    existing transaction.

    Preserve the previous inputs in correction_history.
    Every grade advances to the same batch revision.
    """

    normalized_by_grade = {
        record["tea_type"]: record
        for record in observations
    }

    changed_grades = []

    previous_revisions = {
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

    if len(previous_revisions) != 1:
        raise ValueError(
            "The saved reporting week has inconsistent "
            "batch revisions. Review its records."
        )

    previous_batch_revision = next(
        iter(previous_revisions)
    )

    if revision != previous_batch_revision + 1:
        raise ValueError(
            "The observation revision does not match "
            "the expected new forecast revision."
        )

    for grade in TEA_GRADES:
        existing = existing_by_grade[grade]
        updated = normalized_by_grade[grade]

        changed_fields = find_changed_fields(
            existing,
            updated,
        )

        if changed_fields:
            changed_grades.append(grade)

        old_revision = int(
            existing.get("revision", 0)
        )

        previous_values = {
            field: normalize_numeric(
                existing[field],
                field,
            )
            for field in EIGHT_INPUT_FIELDS
        }

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

        updates = {
            field: updated[field]
            for field in EIGHT_INPUT_FIELDS
        }

        updates["usd_lkr_exchange_rate_range"] = (
            updated["usd_lkr_exchange_rate_range"]
        )

        updates["updated_at"] = now
        updates["data_status"] = (
            "updated_completed_week"
        )
        updates["batch_revision"] = revision

        result = collection.update_one(
            revision_filter,
            {
                "$set": updates,
                "$inc": {
                    "revision": 1,
                },
                "$push": {
                    "correction_history": {
                        "revision": old_revision + 1,
                        "batch_revision": revision,
                        "updated_at": now,
                        "reason": (
                            "Forecast regenerated from "
                            "Weekly Data Entry"
                        ),
                        "previous_values": previous_values,
                        "new_values": {
                            field: updated[field]
                            for field in EIGHT_INPUT_FIELDS
                        },
                        "changed_fields": changed_fields,
                    },
                },
            },
            session=session,
        )

        if result.matched_count != 1:
            raise ValueError(
                "Weekly observations changed while "
                "the forecast was being generated. "
                "Refresh and try again."
            )

    return changed_grades


def commit_generated_forecast(
    batch: EightInputWeeklyBatch,
    forecast_result,
):
    """
    Atomically save or update weekly observations
    together with the associated finalized forecasts.

    Requires a MongoDB deployment supporting
    multi-document transactions.

    Forecast generation must happen before this call.
    """

    prepared = prepare_generation(
        batch,
        forecast_result,
    )

    observations = prepared["observations"]
    normalized_forecasts = prepared["forecasts"]
    snapshot = prepared["snapshot"]
    fingerprint = prepared["fingerprint"]
    origin = prepared["origin"]

    initialize_observation_storage()
    initialize_forecast_storage()
    initialize_forecast_revision_storage()

    database = get_database()

    observation_collection = database[
        OBSERVATION_COLLECTION
    ]

    original_collection = database[
        ORIGINAL_FORECAST_COLLECTION
    ]

    revision_collection = database[
        REVISION_COLLECTION_NAME
    ]

    def write_transaction(session):
        now = datetime.now(timezone.utc)

        saved_observations = list(
            observation_collection.find(
                {"week_start_date": origin},
                session=session,
            )
        )

        original = get_complete_original_set(
            origin,
            session=session,
        )

        latest_revision = get_existing_revision_number(
            origin,
            session=session,
        )

        if not saved_observations:
            # Initial generation must not overwrite
            # pre-existing or partially saved forecasts.
            if original or latest_revision != 0:
                raise ValueError(
                    "Forecast records already exist "
                    "without a complete saved observation "
                    "week. Manual reconciliation is required."
                )

            check_weekly_batch_for_storage(
                observations,
                observation_collection,
                session,
            )

            documents = [
                make_initial_observation(
                    observation,
                    now,
                )
                for observation in observations
            ]

            observation_collection.insert_many(
                documents,
                ordered=True,
                session=session,
            )

            originals = [
                make_original_forecast(
                    document,
                    fingerprint,
                    snapshot,
                )
                for document in normalized_forecasts
            ]

            original_collection.insert_many(
                originals,
                ordered=True,
                session=session,
            )

            return {
                "revision": 0,
                "operation": "created",
                "updated_grades": [],
                "observations_written": 5,
            }

        if not original:
            raise ValueError(
                "Saved observations exist but their "
                "original finalized forecasts are missing. "
                "Complete the existing forecast retry "
                "workflow before regenerating."
            )

        existing_by_grade = (
            check_weekly_batch_for_update(
                observations,
                observation_collection,
                session,
            )
        )

        # No earlier revision is ever modified.
        next_revision = latest_revision + 1

        if next_revision == 1:
            expected_previous = 0
        else:
            expected_previous = latest_revision - 1

        # First generated forecast has revision zero.
        # Subsequent generations use revision 1, 2, ...
        if next_revision <= 0:
            raise ValueError(
                "Invalid next forecast revision."
            )

        # Ensure observation revisions are aligned with
        # forecasts before committing another revision.
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

        if current_batch_revisions != {
            latest_revision
        }:
            raise ValueError(
                "Observation and forecast revisions "
                "are not synchronized. Review the saved "
                "week before regenerating."
            )

        changed_grades = update_existing_observations(
            observations,
            existing_by_grade,
            observation_collection,
            session,
            now,
            next_revision,
        )

        revisions = [
            make_revision_forecast(
                document,
                next_revision,
                fingerprint,
                snapshot,
                now,
            )
            for document in normalized_forecasts
        ]

        revision_collection.insert_many(
            revisions,
            ordered=True,
            session=session,
        )

        return {
            "revision": next_revision,
            "operation": "regenerated",
            "updated_grades": changed_grades,
            "observations_written": 5,
        }

    try:
        with database.client.start_session() as session:
            committed = session.with_transaction(
                write_transaction
            )

    except DuplicateKeyError as exc:
        raise ValueError(
            "A forecast generation request for this "
            "reporting week was saved concurrently. "
            "Refresh the page and try again."
        ) from exc

    return {
        "status": "success",
        "saved": True,
        "finalized": True,
        "forecast_origin": origin,
        "revision": committed["revision"],
        "operation": committed["operation"],
        "saved_grades": list(TEA_GRADES),
        "observation_count": (
            committed["observations_written"]
        ),
        "forecast_count": 20,
        "updated_grades": committed["updated_grades"],
        "input_fingerprint": fingerprint,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "forecasts": forecast_result["forecasts"],
        "message": (
            "The four-week demand forecasts "
            "were generated and saved successfully."
        ),
    }
