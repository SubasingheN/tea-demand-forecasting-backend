
"""
Persistent factory observation storage for the nine-input TFT.

Supports:
- Protected historical factory records
- Latest-completed-week observation entry
- Nine numeric historical model inputs
- Shared weekly market information
- Duplicate protection
- Continuous weekly history
- Controlled corrections
- MongoDB correction audit history

The original historical CSV is never modified.

Normal creation cannot overwrite existing records.
Corrections preserve previous values and require
explicit API-side authorization.

The module performs no database writes on import.
"""

from datetime import date, datetime, timedelta, timezone
import math

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_database

from app.services.eight_input_history_service import (
    load_original_factory_history,
)


COLLECTION_NAME = (
    "eight_input_factory_weekly_observations"
)

TEA_GRADES = (
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
)

EXCHANGE_RATE_FIELDS = (
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
)

AUCTION_FIELDS = (
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
)

EIGHT_INPUT_FIELDS = (
    "demand_kg",
    "production_kg",
    "inventory_kg",
    *EXCHANGE_RATE_FIELDS,
    "festival_flag",
    *AUCTION_FIELDS,
)

SHARED_WEEKLY_FIELDS = (
    *EXCHANGE_RATE_FIELDS,
    "festival_flag",
    *AUCTION_FIELDS,
)

REQUIRED_FIELDS = (
    "week_start_date",
    "tea_type",
    *EIGHT_INPUT_FIELDS,
)

MODEL_NAME = "nine_input_tft"

DATA_SOURCE = "factory_weekly_manual_entry"

FACTORY_TIMEZONE = timezone(
    timedelta(hours=5, minutes=30)
)


def get_observation_collection():
    """Return the existing factory observation collection."""

    return get_database()[COLLECTION_NAME]


def current_factory_date():
    """Get today's date in the factory's local timezone."""

    return datetime.now(
        FACTORY_TIMEZONE
    ).date()


def latest_completed_week_start():
    """
    Get the Monday of the latest fully completed
    Monday-to-Sunday reporting week.
    """

    today = current_factory_date()

    current_monday = (
        today - timedelta(days=today.weekday())
    )

    return current_monday - timedelta(weeks=1)


def normalize_week_date(value):
    """Validate an ISO Monday reporting date."""

    if isinstance(value, datetime):
        value = value.date()

    elif isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "week_start_date must use YYYY-MM-DD."
            ) from exc

        if parsed.isoformat() != value:
            raise ValueError(
                "week_start_date must use YYYY-MM-DD."
            )

        value = parsed

    if not isinstance(value, date):
        raise ValueError(
            "A valid week_start_date is required."
        )

    if value.weekday() != 0:
        raise ValueError(
            "The reporting week must start on Monday."
        )

    return value


def validate_completed_week(week_start):
    """Accept only fully completed reporting weeks."""

    week_start = normalize_week_date(week_start)

    week_end = (
        week_start + timedelta(days=6)
    )

    if week_end >= current_factory_date():
        raise ValueError(
            "The reporting week has not finished. "
            "Submit only completed weekly observations."
        )

    return week_start


def validate_latest_completed_week(week_start):
    """
    Restrict normal new factory observations to
    the latest completed reporting week.
    """

    week_start = validate_completed_week(
        week_start
    )

    expected = latest_completed_week_start()

    if week_start != expected:
        raise ValueError(
            "Normal Weekly Data Entry accepts only "
            "the latest completed reporting week: "
            f"{expected.isoformat()}."
        )

    return week_start


def normalize_numeric(value, field):
    """Validate and normalize a factory numeric input."""

    if (
        value is None
        or isinstance(value, bool)
        or (
            isinstance(value, str)
            and not value.strip()
        )
    ):
        raise ValueError(
            f"{field} must be a valid number."
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be numeric."
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{field} must be finite."
        )

    positive_fields = {
        "demand_kg",
        *EXCHANGE_RATE_FIELDS,
        *AUCTION_FIELDS,
    }

    if field in positive_fields:
        if number <= 0:
            raise ValueError(
                f"{field} must be greater than zero."
            )
    elif number < 0:
        raise ValueError(
            f"{field} cannot be negative."
        )

    limits = {
        "demand_kg": 100000,
        "production_kg": 100000,
        "inventory_kg": 100000,
        "usd_lkr_monday_rate": 10000,
        "usd_lkr_sunday_rate": 10000,
        "auction_price_high_lkr": 100000,
        "auction_price_medium_lkr": 100000,
        "auction_price_low_lkr": 100000,
    }

    if (
        field in limits
        and number > limits[field]
    ):
        raise ValueError(
            f"{field} exceeds its allowed maximum."
        )

    if field == "festival_flag":
        if number not in (0.0, 1.0):
            raise ValueError(
                "festival_flag must be 0 or 1."
            )

        return int(number)

    return number


def normalize_observation(payload):
    """
    Validate an observation without writing
    or changing any database records.
    """

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(
            mode="python"
        )

    if not isinstance(payload, dict):
        raise ValueError(
            "Observation must be an object."
        )

    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in payload
    ]

    if missing:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing)
        )

    week_start = validate_completed_week(
        payload["week_start_date"]
    )

    grade = payload["tea_type"]

    if grade not in TEA_GRADES:
        raise ValueError(
            f"Unsupported tea grade: {grade}"
        )

    normalized = {
        "week_start_date": week_start.isoformat(),
        "tea_type": grade,
    }

    for field in EIGHT_INPUT_FIELDS:
        normalized[field] = normalize_numeric(
            payload[field],
            field,
        )

    # Reference/display field derived from
    # observed numeric rate endpoints.
    normalized["usd_lkr_exchange_rate_range"] = (
        f"{normalized['usd_lkr_monday_rate']:.4f}"
        "-"
        f"{normalized['usd_lkr_sunday_rate']:.4f}"
    )

    return normalized


def get_factory_history_bounds():
    """Read the protected historical dataset bounds."""

    history = load_original_factory_history()

    if history.empty:
        raise ValueError(
            "The original historical dataset is empty."
        )

    dates = history["week_start_date"]

    return {
        "first_week": dates.min().date().isoformat(),
        "latest_week": dates.max().date().isoformat(),
        "total_rows": len(history),
    }


def validate_editable_week(week_start):
    """
    Permit corrections only to completed
    weeks after the protected historical CSV.

    Correcting an existing observation does
    not open normal entry for historical weeks.
    """

    normalized = validate_completed_week(
        week_start
    ).isoformat()

    bounds = get_factory_history_bounds()

    if normalized <= bounds["latest_week"]:
        raise ValueError(
            "This reporting week belongs to "
            "protected historical factory data. "
            "It cannot be changed."
        )

    return normalized


def shared_values_match(left, right, field):
    """Compare shared market values consistently."""

    left_number = normalize_numeric(
        left,
        field,
    )

    right_number = normalize_numeric(
        right,
        field,
    )

    if field == "festival_flag":
        return left_number == right_number

    return math.isclose(
        left_number,
        right_number,
        rel_tol=0,
        abs_tol=0.001,
    )


def check_observation_for_storage(
    payload,
    collection=None,
):
    """
    Validate new observations before storage.

    Performs database reads only.
    Restricts normal entry to the latest
    fully completed reporting week.
    """

    observation = normalize_observation(
        payload
    )

    week_start = observation[
        "week_start_date"
    ]

    validate_latest_completed_week(
        week_start
    )

    if collection is None:
        collection = get_observation_collection()

    grade = observation["tea_type"]

    latest_csv_week = get_factory_history_bounds()[
        "latest_week"
    ]

    if week_start <= latest_csv_week:
        raise ValueError(
            "This reporting week belongs to the "
            "protected historical dataset. "
            "Original observations cannot be replaced."
        )

    existing = collection.find_one(
        {
            "week_start_date": week_start,
            "tea_type": grade,
        },
        {"_id": 1},
    )

    if existing is not None:
        raise ValueError(
            f"{grade} already has an observation "
            f"for {week_start}. Existing observations "
            "cannot be overwritten through normal entry."
        )

    previous_week = (
        date.fromisoformat(week_start)
        - timedelta(weeks=1)
    ).isoformat()

    if previous_week != latest_csv_week:
        predecessor = collection.find_one(
            {
                "week_start_date": previous_week,
                "tea_type": grade,
            },
            {"_id": 1},
        )

        if predecessor is None:
            raise ValueError(
                f"{grade} is missing its preceding "
                f"weekly observation: {previous_week}. "
                "Continuous weekly history is required."
            )

    peers = collection.find(
        {
            "week_start_date": week_start,
            "tea_type": {"$ne": grade},
        },
        {
            "_id": 0,
            **{
                field: 1
                for field in SHARED_WEEKLY_FIELDS
            },
        },
    )

    for peer in peers:
        for field in SHARED_WEEKLY_FIELDS:
            if field not in peer:
                raise ValueError(
                    "Stored factory observation uses "
                    "an outdated schema or lacks "
                    f"{field}. Migration is required."
                )

            if not shared_values_match(
                observation[field],
                peer[field],
                field,
            ):
                raise ValueError(
                    f"{field} conflicts with another "
                    f"tea grade for {week_start}."
                )

    return {
        "status": "valid",
        "observation": observation,
        "dataset_latest_week": latest_csv_week,
        "previous_week": previous_week,
        "write_performed": False,
    }


def initialize_observation_storage():
    """Ensure grade/week uniqueness in MongoDB."""

    collection = get_observation_collection()

    return collection.create_index(
        [
            ("week_start_date", ASCENDING),
            ("tea_type", ASCENDING),
        ],
        unique=True,
        name="unique_eight_input_week_grade",
    )


def save_eight_input_observation(payload):
    """
    Save one new weekly factory observation.

    Existing records are never overwritten
    through this operation.
    """

    initialize_observation_storage()

    collection = get_observation_collection()

    checked = check_observation_for_storage(
        payload,
        collection=collection,
    )

    observation = dict(
        checked["observation"]
    )

    observation["created_at"] = (
        datetime.now(timezone.utc)
    )

    observation["source"] = DATA_SOURCE
    observation["record_type"] = "factory_observation"
    observation["model_context"] = MODEL_NAME
    observation["data_status"] = (
        "submitted_completed_week"
    )
    observation["revision"] = 0

    try:
        result = collection.insert_one(
            observation
        )
    except DuplicateKeyError as exc:
        raise ValueError(
            "An observation already exists for "
            "this tea grade and reporting week."
        ) from exc

    return {
        "status": "saved",
        "observation_id": str(
            result.inserted_id
        ),
        "week_start_date": observation[
            "week_start_date"
        ],
        "tea_type": observation["tea_type"],
        "saved": True,
    }


def update_eight_input_observation(
    payload,
    *,
    authorized=False,
    operator_id=None,
    reason=None,
):
    """
    Correct a previously saved factory observation.

    The API must authorize this operation before
    invoking the service.

    Shared market values are updated across all
    saved grades within the same reporting week.

    Updates and correction-history entries use
    one MongoDB transaction.

    Existing finalized forecasts are not
    automatically replaced by corrections.
    """

    if not authorized:
        raise PermissionError(
            "This correction requires authorization."
        )

    if (
        not isinstance(operator_id, str)
        or not operator_id.strip()
    ):
        raise ValueError(
            "A verified operator identifier is required."
        )

    if (
        not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError(
            "A correction reason is required."
        )

    operator_id = operator_id.strip()
    reason = reason.strip()

    if len(reason) > 500:
        raise ValueError(
            "The correction reason is too long."
        )

    observation = normalize_observation(
        payload
    )

    week_start = validate_editable_week(
        observation["week_start_date"]
    )

    grade = observation["tea_type"]

    initialize_observation_storage()

    database = get_database()
    collection = database[COLLECTION_NAME]

    now = datetime.now(timezone.utc)

    with database.client.start_session() as session:
        with session.start_transaction():

            existing = collection.find_one(
                {
                    "week_start_date": week_start,
                    "tea_type": grade,
                },
                session=session,
            )

            if existing is None:
                raise ValueError(
                    "No saved factory observation "
                    "exists for this grade and week."
                )

            if existing.get("source") != DATA_SOURCE:
                raise ValueError(
                    "Only factory-entered records "
                    "can be corrected."
                )

            if (
                existing.get("record_type")
                != "factory_observation"
            ):
                raise ValueError(
                    "This record cannot be corrected."
                )

            peers = list(
                collection.find(
                    {
                        "week_start_date": week_start,
                    },
                    session=session,
                )
            )

            if not peers:
                raise ValueError(
                    "No saved records exist "
                    "for the selected week."
                )

            for peer in peers:
                if (
                    peer.get("source") != DATA_SOURCE
                    or peer.get("record_type")
                    != "factory_observation"
                ):
                    raise ValueError(
                        "This reporting week contains "
                        "records that cannot be corrected."
                    )

                missing_fields = [
                    field
                    for field in EIGHT_INPUT_FIELDS
                    if field not in peer
                ]

                if missing_fields:
                    raise ValueError(
                        "Stored observations use an outdated "
                        "input format. Migration is required "
                        f"before corrections: {missing_fields}"
                    )

            shared_changed = any(
                not shared_values_match(
                    observation[field],
                    existing[field],
                    field,
                )
                for field in SHARED_WEEKLY_FIELDS
            )

            updated_grades = []

            for peer in peers:
                is_selected = (
                    peer["tea_type"] == grade
                )

                updates = {}

                if is_selected:
                    for field in EIGHT_INPUT_FIELDS:
                        updates[field] = observation[field]

                if shared_changed:
                    for field in SHARED_WEEKLY_FIELDS:
                        updates[field] = observation[field]

                # Keep the human-readable exchange
                # range synchronized with numeric values.
                effective_monday = updates.get(
                    "usd_lkr_monday_rate",
                    peer["usd_lkr_monday_rate"],
                )

                effective_sunday = updates.get(
                    "usd_lkr_sunday_rate",
                    peer["usd_lkr_sunday_rate"],
                )

                effective_range = (
                    f"{float(effective_monday):.4f}"
                    "-"
                    f"{float(effective_sunday):.4f}"
                )

                changed_fields = {}

                for field, new_value in updates.items():
                    previous_value = normalize_numeric(
                        peer[field],
                        field,
                    )

                    if previous_value != new_value:
                        changed_fields[field] = {
                            "previous": previous_value,
                            "new": new_value,
                        }

                previous_range = peer.get(
                    "usd_lkr_exchange_rate_range"
                )

                if previous_range != effective_range:
                    changed_fields[
                        "usd_lkr_exchange_rate_range"
                    ] = {
                        "previous": previous_range,
                        "new": effective_range,
                    }

                if not changed_fields:
                    continue

                old_revision = int(
                    peer.get("revision", 0)
                )

                revision_filter = {
                    "_id": peer["_id"],
                }

                if "revision" in peer:
                    revision_filter["revision"] = (
                        old_revision
                    )
                else:
                    revision_filter["revision"] = {
                        "$exists": False
                    }

                update_result = collection.update_one(
                    revision_filter,
                    {
                        "$set": {
                            **{
                                field: values["new"]
                                for field, values
                                in changed_fields.items()
                            },
                            "updated_at": now,
                            "data_status": (
                                "corrected_completed_week"
                            ),
                        },
                        "$inc": {
                            "revision": 1,
                        },
                        "$push": {
                            "correction_history": {
                                "revision": old_revision + 1,
                                "corrected_at": now,
                                "operator_id": operator_id,
                                "reason": reason,
                                "changed_fields": changed_fields,
                            },
                        },
                    },
                    session=session,
                )

                if update_result.matched_count != 1:
                    raise ValueError(
                        "The record changed during "
                        "correction. Reload the week "
                        "and try again."
                    )

                updated_grades.append(
                    peer["tea_type"]
                )

    return {
        "status": (
            "updated"
            if updated_grades
            else "unchanged"
        ),
        "week_start_date": week_start,
        "tea_type": grade,
        "updated_grades": updated_grades,
        "updated_count": len(updated_grades),
        "shared_values_changed": shared_changed,
        "write_performed": bool(updated_grades),
        "message": (
            "Factory records updated successfully."
            if updated_grades
            else "No changes were necessary."
        ),
    }


def list_eight_input_observations(
    grade=None,
    limit=100,
):
    """
    Retrieve saved factory observations
    without exposing correction audit details.
    """

    if (
        grade is not None
        and grade not in TEA_GRADES
    ):
        raise ValueError(
            "Invalid tea grade."
        )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 500
    ):
        raise ValueError(
            "limit must be between 1 and 500."
        )

    query = {}

    if grade is not None:
        query["tea_type"] = grade

    collection = get_observation_collection()

    cursor = (
        collection.find(
            query,
            {"_id": 0},
        )
        .sort([
            ("week_start_date", DESCENDING),
            ("tea_type", ASCENDING),
        ])
        .limit(limit)
    )

    output = []

    for document in cursor:
        item = dict(document)

        for timestamp_field in (
            "created_at",
            "updated_at",
        ):
            if isinstance(
                item.get(timestamp_field),
                datetime,
            ):
                item[timestamp_field] = (
                    item[timestamp_field].isoformat()
                )

        # Audit records remain protected in MongoDB.
        item.pop("correction_history", None)

        output.append(item)

    return output
