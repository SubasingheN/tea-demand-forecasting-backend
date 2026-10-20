
"""
Permanent MongoDB storage for production nine-input
TFT tea demand forecasts.

Preserves the existing MongoDB collection:
    eight_input_final_forecasts

Supports:
- One finalized forecast per origin and tea grade
- Atomic multi-grade forecast saving
- Strict date and quantile validation
- Model/version identification
- Historical backfilled forecast identification
- Unique-index duplicate protection
- Forecast history retrieval
- Consecutive-week forecast comparisons

Protected:
- Original historical factory dataset
- Production TFT checkpoint
- Factory observation records
- Previously finalized forecasts
- Existing six-input forecast collections

Historical backfilled forecasts are not represented
as predictions genuinely issued in the past.

No database writes occur during module import.
"""

from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)

import math

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.db.mongodb import get_database


COLLECTION_NAME = "eight_input_final_forecasts"

MODEL_NAME = "nine_input_tft"
MODEL_VERSION = "production_v1"

TEA_GRADES = (
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
)

FORECAST_HORIZON = 4
HISTORICAL_INPUT_WEEKS = 52

FORECAST_FIELDS = (
    "lower_demand_kg",
    "median_demand_kg",
    "upper_demand_kg",
)

HISTORICAL_FORECAST_TYPES = {
    "historical_replay",
    "historical_backfill",
}

OPERATIONAL_FORECAST_TYPE = (
    "latest_completed_week_forecast"
)


def get_forecast_collection():
    """Return the existing final forecast collection."""

    return get_database()[COLLECTION_NAME]


def parse_week_date(value, field_name):
    """Validate a Monday date in ISO format."""

    if isinstance(value, datetime):
        raise ValueError(
            f"{field_name} must be a date "
            "without a time component."
        )

    if isinstance(value, date):
        parsed = value

    elif isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must use YYYY-MM-DD."
            ) from exc

        if parsed.isoformat() != value:
            raise ValueError(
                f"{field_name} must use YYYY-MM-DD."
            )

    else:
        raise ValueError(
            f"{field_name} must be a valid date."
        )

    if parsed.weekday() != 0:
        raise ValueError(
            f"{field_name} must be a Monday."
        )

    return parsed


def normalize_number(value, field):
    """Validate a finite nonnegative demand value."""

    if (
        value is None
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"{field} must be numeric."
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

    if number < 0:
        raise ValueError(
            f"{field} cannot be negative."
        )

    return number


def normalize_forecast_type(result):
    """
    Identify whether the forecast was generated
    operationally or reconstructed historically.

    A historical forecast created using the
    full-data production model is a backfill,
    not a genuine previously issued prediction.
    """

    forecast_type = result.get(
        "forecast_type"
    )

    if forecast_type in HISTORICAL_FORECAST_TYPES:
        return {
            "forecast_type": "historical_backfill",
            "historical_backfill": True,
            "operational_forecast": False,
            "provenance_label": "Historical backfill",
            "issued_at_origin": False,
        }

    if forecast_type == OPERATIONAL_FORECAST_TYPE:
        return {
            "forecast_type": OPERATIONAL_FORECAST_TYPE,
            "historical_backfill": False,
            "operational_forecast": True,
            "provenance_label": "Operational forecast",
            "issued_at_origin": False,
        }

    raise ValueError(
        "Unsupported forecast_type. Expected "
        "latest_completed_week_forecast, "
        "historical_replay or historical_backfill."
    )


def normalize_forecast(result, grade):
    """
    Validate and normalize one grade's
    four-week forecast before storage.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Forecast result must be a dictionary."
        )

    if result.get("model") != MODEL_NAME:
        raise ValueError(
            "Only the production nine-input "
            "TFT model is supported for new saves."
        )

    if result.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Production model version mismatch."
        )

    if grade not in TEA_GRADES:
        raise ValueError(
            "Unsupported tea grade."
        )

    if result.get(
        "historical_input_weeks"
    ) != HISTORICAL_INPUT_WEEKS:
        raise ValueError(
            "Forecast historical input length mismatch."
        )

    if result.get(
        "forecast_horizon_weeks"
    ) != FORECAST_HORIZON:
        raise ValueError(
            "Forecast horizon must be four weeks."
        )

    origin_date = parse_week_date(
        result.get("forecast_origin"),
        "forecast_origin",
    )

    expected_start = (
        origin_date + timedelta(weeks=1)
    )

    expected_end = (
        origin_date
        + timedelta(weeks=FORECAST_HORIZON)
    )

    if parse_week_date(
        result.get("forecast_start_date"),
        "forecast_start_date",
    ) != expected_start:
        raise ValueError(
            "Forecast start date does not match "
            "the reporting-week origin."
        )

    if parse_week_date(
        result.get("forecast_end_date"),
        "forecast_end_date",
    ) != expected_end:
        raise ValueError(
            "Forecast end date does not match "
            "the four-week horizon."
        )

    forecasts_by_grade = result.get(
        "forecasts"
    )

    if not isinstance(forecasts_by_grade, dict):
        raise ValueError(
            "Forecast results must be grouped "
            "by tea grade."
        )

    forecasts = forecasts_by_grade.get(
        grade
    )

    if not isinstance(forecasts, list):
        raise ValueError(
            f"Forecast data missing for {grade}."
        )

    if len(forecasts) != FORECAST_HORIZON:
        raise ValueError(
            "Exactly four forecast weeks "
            "are required."
        )

    normalized_weeks = []

    for expected_horizon, item in enumerate(
        forecasts,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "Each weekly forecast must "
                "be a dictionary."
            )

        horizon = item.get(
            "horizon"
        )

        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon != expected_horizon
        ):
            raise ValueError(
                "Invalid forecast horizon ordering."
            )

        expected_date = (
            origin_date
            + timedelta(weeks=expected_horizon)
        )

        actual_date = parse_week_date(
            item.get("week"),
            "forecast_target_week",
        )

        if actual_date != expected_date:
            raise ValueError(
                "Forecast target weeks must be "
                "consecutive Mondays after "
                "the reporting week."
            )

        lower = normalize_number(
            item.get("lower_demand_kg"),
            "lower_demand_kg",
        )

        median = normalize_number(
            item.get("median_demand_kg"),
            "median_demand_kg",
        )

        upper = normalize_number(
            item.get("upper_demand_kg"),
            "upper_demand_kg",
        )

        if not lower <= median <= upper:
            raise ValueError(
                "Forecast demand estimates "
                "must be correctly ordered."
            )

        normalized_weeks.append({
            "week": actual_date.isoformat(),
            "horizon": expected_horizon,
            "lower_demand_kg": lower,
            "median_demand_kg": median,
            "upper_demand_kg": upper,
        })

    provenance = normalize_forecast_type(
        result
    )

    now = datetime.now(
        timezone.utc
    )

    return {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "forecast_origin": (
            origin_date.isoformat()
        ),
        "forecast_start_date": (
            expected_start.isoformat()
        ),
        "forecast_end_date": (
            expected_end.isoformat()
        ),
        "tea_type": grade,
        **provenance,
        "historical_input_weeks": (
            HISTORICAL_INPUT_WEEKS
        ),
        "forecast_horizon_weeks": (
            FORECAST_HORIZON
        ),
        "unit": "kg",
        "quantiles": {
            "lower": 0.1,
            "median": 0.5,
            "upper": 0.9,
        },
        "forecasts": normalized_weeks,
        "created_at": now,
        "finalized_at": now,
        "finalized": True,
        "saved": True,
        "input_features": result.get(
            "input_features",
            [],
        ),
        "source_dataset_latest_week": (
            result.get("dataset_latest_week")
        ),
        "message": (
            "Backfilled using a production model "
            "trained on the complete approved "
            "historical dataset. This is not an "
            "independently issued past forecast."
            if provenance["historical_backfill"]
            else
            "Generated from a completed factory "
            "reporting week."
        ),
    }


def initialize_forecast_storage():
    """
    Create or reuse the existing unique index.

    The collection name and index name remain
    unchanged for backward compatibility.
    """

    collection = get_forecast_collection()

    return collection.create_index(
        [
            ("forecast_origin", ASCENDING),
            ("tea_type", ASCENDING),
        ],
        unique=True,
        name="unique_eight_input_forecast_origin_grade",
    )


def save_final_forecasts_atomic(
    result,
    grades,
):
    """
    Save selected tea-grade forecasts atomically.

    Either all requested grades are committed
    or none are committed.

    Existing forecasts are never overwritten,
    including forecasts from older models.
    """

    grades = list(grades)

    if not grades:
        raise ValueError(
            "At least one tea grade is required."
        )

    if (
        len(grades) != len(set(grades))
        or any(
            grade not in TEA_GRADES
            for grade in grades
        )
    ):
        raise ValueError(
            "Invalid or duplicate tea grades."
        )

    documents = [
        normalize_forecast(
            result,
            grade,
        )
        for grade in grades
    ]

    database = get_database()
    collection = database[COLLECTION_NAME]

    initialize_forecast_storage()

    with database.client.start_session() as session:

        def write_transaction(active_session):
            """
            Verify the origin is entirely available
            before inserting any new forecasts.
            """

            origin = documents[0][
                "forecast_origin"
            ]

            existing = collection.find_one(
                {
                    "forecast_origin": origin,
                },
                {
                    "_id": 1,
                    "tea_type": 1,
                    "model": 1,
                },
                session=active_session,
            )

            if existing is not None:
                raise ValueError(
                    "Forecast records already exist "
                    f"for reporting week {origin}. "
                    "Existing forecasts will not "
                    "be overwritten or mixed "
                    "with another model version."
                )

            result_insert = collection.insert_many(
                documents,
                ordered=True,
                session=active_session,
            )

            return [
                {
                    "tea_type": document[
                        "tea_type"
                    ],
                    "forecast_id": str(
                        inserted_id
                    ),
                }
                for document, inserted_id in zip(
                    documents,
                    result_insert.inserted_ids,
                )
            ]

        try:
            saved = session.with_transaction(
                write_transaction
            )

        except DuplicateKeyError as exc:
            raise ValueError(
                "Forecast records already exist "
                "for one or more selected grades. "
                "The existing records have "
                "not been overwritten."
            ) from exc

    return saved


def save_final_forecast(result, grade):
    """
    Compatibility function for saving
    one tea-grade forecast.
    """

    saved = save_final_forecasts_atomic(
        result,
        [grade],
    )

    return {
        "status": "saved",
        "forecast_id": saved[0][
            "forecast_id"
        ],
        "forecast_origin": result[
            "forecast_origin"
        ],
        "tea_type": grade,
        "saved": True,
    }


def serialize_forecast_document(document):
    """
    Make a stored forecast document
    suitable for JSON API responses.
    """

    record = dict(document)

    record.pop("_id", None)

    for field in (
        "created_at",
        "finalized_at",
        "updated_at",
    ):
        if isinstance(
            record.get(field),
            datetime,
        ):
            record[field] = (
                record[field].isoformat()
            )

    return record


def list_final_forecasts(
    grade=None,
    limit=100,
):
    """
    Retrieve finalized forecasts, newest first.

    Includes both operational and historical
    backfilled forecasts.

    Older records retain their original model
    identifiers and metadata.
    """

    if (
        grade is not None
        and grade not in TEA_GRADES
    ):
        raise ValueError(
            "Unsupported tea grade."
        )

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 500
    ):
        raise ValueError(
            "limit must be between 1 and 500."
        )

    query = {
        "finalized": True,
    }

    if grade is not None:
        query["tea_type"] = grade

    cursor = (
        get_forecast_collection()
        .find(
            query,
            {"_id": 0},
        )
        .sort([
            (
                "forecast_origin",
                DESCENDING,
            ),
            (
                "tea_type",
                ASCENDING,
            ),
        ])
        .limit(limit)
    )

    return [
        serialize_forecast_document(item)
        for item in cursor
    ]


def get_previous_final_forecast(
    grade,
    current_origin,
):
    """
    Retrieve a finalized forecast from exactly
    one reporting week before current_origin.

    Returns None when no forecast exists.
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

    document = (
        get_forecast_collection()
        .find_one(
            {
                "tea_type": grade,
                "finalized": True,
                "forecast_origin": (
                    previous_origin
                ),
            },
            {"_id": 0},
        )
    )

    if document is None:
        return None

    return serialize_forecast_document(
        document
    )


def compare_forecasts(
    previous,
    current,
):
    """
    Compare P50 forecasts for common target weeks.

    Only compare:
    - The same tea grade
    - Forecast origins seven days apart
    - The same model and model version
    - The same forecast provenance category

    This prevents misleading comparisons
    between older model versions or between
    historical backfills and live forecasts.
    """

    if previous is None or current is None:
        return []

    if (
        previous.get("finalized") is not True
        or current.get("finalized") is not True
    ):
        return []

    if (
        previous.get("tea_type")
        != current.get("tea_type")
    ):
        return []

    if (
        previous.get("model")
        != current.get("model")
    ):
        return []

    if (
        previous.get("model_version")
        != current.get("model_version")
    ):
        return []

    if (
        previous.get("historical_backfill")
        != current.get("historical_backfill")
    ):
        return []

    previous_origin = parse_week_date(
        previous.get("forecast_origin"),
        "previous_forecast_origin",
    )

    current_origin = parse_week_date(
        current.get("forecast_origin"),
        "current_forecast_origin",
    )

    if (
        current_origin - previous_origin
        != timedelta(days=7)
    ):
        return []

    previous_weeks = {
        item["week"]: item
        for item in previous.get(
            "forecasts",
            [],
        )
    }

    rows = []

    for item in current.get(
        "forecasts",
        [],
    ):
        old = previous_weeks.get(
            item["week"]
        )

        if old is None:
            continue

        previous_value = normalize_number(
            old.get("median_demand_kg"),
            "previous_median_demand_kg",
        )

        current_value = normalize_number(
            item.get("median_demand_kg"),
            "current_median_demand_kg",
        )

        difference = (
            current_value - previous_value
        )

        percentage = (
            difference / previous_value * 100
            if previous_value != 0
            else None
        )

        rows.append({
            "tea_type": current[
                "tea_type"
            ],
            "target_week": item[
                "week"
            ],
            "previous_forecast_origin": (
                previous["forecast_origin"]
            ),
            "current_forecast_origin": (
                current["forecast_origin"]
            ),
            "previous_demand_kg": (
                previous_value
            ),
            "current_demand_kg": (
                current_value
            ),
            "difference_kg": (
                difference
            ),
            "change_percent": (
                percentage
            ),
        })

    return rows
