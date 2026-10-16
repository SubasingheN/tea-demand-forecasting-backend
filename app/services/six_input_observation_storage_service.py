
"""
Persistent storage for six-input TFT factory observations.

Protects:
- The original factory dataset
- The frozen TFT checkpoint
- Previously recorded weekly observations
- Existing MongoDB research collections

New observations are stored in a separate
MongoDB collection, one document per grade/week.

The original CSV history boundary is always
calculated from the protected CSV alone.
"""

from datetime import date, datetime, timedelta, timezone

import math

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_database

from app.services.six_input_history_service import (
    load_original_factory_history,
)


COLLECTION_NAME = (
    "six_input_factory_weekly_observations"
)

TEA_GRADES = (
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
)

SIX_INPUT_FIELDS = (
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_rate",
    "festival_flag",
    "colombo_auction_price_lkr",
)

REQUIRED_FIELDS = (
    "week_start_date",
    "tea_type",
    *SIX_INPUT_FIELDS,
)


def get_observation_collection():
    """Return the dedicated MongoDB collection."""

    return get_database()[COLLECTION_NAME]


def normalize_week_date(value):
    """Validate a Monday reporting-week date."""

    if isinstance(value, datetime):
        value = value.date()

    elif isinstance(value, str):
        try:
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "week_start_date must use YYYY-MM-DD."
            ) from exc

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
    """
    A reporting week becomes eligible for storage
    only after its Sunday has finished.
    """

    week_start = normalize_week_date(week_start)

    end_date = week_start + timedelta(days=6)

    if end_date >= date.today():
        raise ValueError(
            "The reporting week has not finished. "
            "Submit only finalized factory observations."
        )


def normalize_numeric(value, field):
    """Reject missing, nonnumeric or infinite inputs."""

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

    if field in (
        "demand_kg",
        "usd_lkr_rate",
        "colombo_auction_price_lkr",
    ):
        if number <= 0:
            raise ValueError(
                f"{field} must be greater than zero."
            )

    elif number < 0:
        raise ValueError(
            f"{field} cannot be negative."
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
    Validate one grade's weekly observation.

    Returns an independent, normalized dictionary.
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

    week_start = normalize_week_date(
        payload["week_start_date"]
    )

    validate_completed_week(week_start)

    grade = payload["tea_type"]

    if grade not in TEA_GRADES:
        raise ValueError(
            f"Unsupported tea grade: {grade}"
        )

    normalized = {
        "week_start_date": week_start.isoformat(),
        "tea_type": grade,
    }

    for field in SIX_INPUT_FIELDS:
        normalized[field] = normalize_numeric(
            payload[field],
            field,
        )

    return normalized


def get_factory_history_bounds():
    """
    Inspect only the protected factory CSV.

    IMPORTANT:
    Do not use load_six_input_history() here.

    That function combines CSV and MongoDB data.
    Using combined history would cause the
    protected dataset boundary to move whenever
    a new MongoDB observation is saved.

    This function must always return the actual
    original CSV boundaries and row count.
    """

    history = load_original_factory_history()

    if history.empty:
        raise ValueError(
            "The protected factory CSV is empty."
        )

    dates = history["week_start_date"]

    return {
        "first_week": (
            dates.min().date().isoformat()
        ),
        "latest_week": (
            dates.max().date().isoformat()
        ),
        "total_rows": len(history),
    }


def check_observation_for_storage(
    payload,
    collection=None,
):
    """
    Validate one new completed-week record.

    Safety checks:
    - Must follow the original protected CSV
    - Must not overwrite an existing observation
    - Must follow a recorded previous week
    - Shared weekly FX/festival values must agree

    This function performs no database writes.
    """

    observation = normalize_observation(payload)

    collection = (
        collection
        if collection is not None
        else get_observation_collection()
    )

    bounds = get_factory_history_bounds()

    week_start = observation["week_start_date"]

    grade = observation["tea_type"]

    latest_csv_week = bounds["latest_week"]

    if week_start <= latest_csv_week:
        raise ValueError(
            "This reporting week already belongs "
            "to the protected factory CSV dataset. "
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
            f"{grade} already has a stored "
            f"observation for {week_start}. "
            "Existing records cannot be overwritten."
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
                f"{grade} is missing its previous "
                f"weekly observation: {previous_week}. "
                "Weekly history must remain continuous."
            )

    peers = collection.find(
        {
            "week_start_date": week_start,
            "tea_type": {"$ne": grade},
        },
        {
            "_id": 0,
            "usd_lkr_rate": 1,
            "festival_flag": 1,
        },
    )

    for peer in peers:
        if not math.isclose(
            observation["usd_lkr_rate"],
            float(peer["usd_lkr_rate"]),
            rel_tol=0,
            abs_tol=0.001,
        ):
            raise ValueError(
                "USD/LKR exchange rate conflicts "
                "with another stored grade for "
                "the same week."
            )

        if (
            observation["festival_flag"]
            != int(peer["festival_flag"])
        ):
            raise ValueError(
                "Festival flag conflicts with "
                "another stored grade for "
                "the same week."
            )

    return {
        "status": "valid",
        "observation": observation,
        "dataset_latest_week": latest_csv_week,
        "previous_week": previous_week,
        "write_performed": False,
    }


def initialize_observation_storage():
    """
    Create the unique grade/week index.

    Call this explicitly during controlled setup.

    This function changes database index metadata.
    """

    collection = get_observation_collection()

    return collection.create_index(
        [
            ("week_start_date", ASCENDING),
            ("tea_type", ASCENDING),
        ],
        unique=True,
        name="unique_six_input_week_grade",
    )


def save_six_input_observation(payload):
    """
    Store one validated observation.

    This function performs a MongoDB insert.

    It never updates or replaces an existing
    observation.

    Note:
    The multi-grade API has its own transactional
    save process. This helper is retained for
    compatibility with existing callers.
    """

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

    observation["source"] = (
        "factory_weekly_manual_entry"
    )

    try:
        result = collection.insert_one(
            observation
        )
    except DuplicateKeyError as exc:
        raise ValueError(
            "An observation for this grade and "
            "week already exists."
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


def list_six_input_observations(
    grade=None,
    limit=100,
):
    """
    Read the most recently stored factory
    observations without modifying records.

    Records are selected newest-first so a
    limit does not exclude the latest weeks.

    The returned list is arranged in
    chronological order for easy display.
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
        .sort(
            [
                ("week_start_date", DESCENDING),
                ("tea_type", ASCENDING),
            ]
        )
        .limit(limit)
    )

    output = []

    for document in cursor:
        item = dict(document)

        if isinstance(
            item.get("created_at"),
            datetime,
        ):
            item["created_at"] = (
                item["created_at"].isoformat()
            )

        output.append(item)

    output.sort(
        key=lambda item: (
            item["week_start_date"],
            item["tea_type"],
        )
    )

    return output
