from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING

from app.db.mongodb import get_database


COLLECTION_NAME = "forecast_runs"


def get_forecast_collection():
    database = get_database()

    return database[
        COLLECTION_NAME
    ]


def create_forecast_indexes():
    collection = (
        get_forecast_collection()
    )

    collection.create_index(
        [
            (
                "forecast_id",
                ASCENDING,
            )
        ],
        unique=True,
        name="unique_forecast_id",
    )

    collection.create_index(
        [
            (
                "generated_at",
                DESCENDING,
            )
        ],
        name="generated_at_desc",
    )


def generate_forecast_id() -> str:
    """
    Generate the next friendly
    forecast ID.

    Example:
    forecast_1
    forecast_2
    forecast_3
    """

    collection = (
        get_forecast_collection()
    )

    existing_records = collection.find(
        {},
        {
            "forecast_id": 1,
        },
    )

    existing_numbers = []

    for record in existing_records:
        forecast_id = record.get(
            "forecast_id",
            ""
        )

        if forecast_id.startswith(
            "forecast_"
        ):
            try:
                number = int(
                    forecast_id.split(
                        "_"
                    )[-1]
                )

                existing_numbers.append(
                    number
                )

            except ValueError:
                continue

    if not existing_numbers:
        return "forecast_1"

    next_number = (
        max(existing_numbers) + 1
    )

    return (
        f"forecast_{next_number}"
    )


def save_forecast_to_database(
    forecasts: list,
    input_week_start_date=None,
    source: str = "research",
    forecast_id=None,
    generated_at=None,
) -> dict:
    """
    Save one forecast run to
    MongoDB Atlas.
    """

    create_forecast_indexes()

    collection = (
        get_forecast_collection()
    )

    if forecast_id is None:
        forecast_id = (
            generate_forecast_id()
        )

    if generated_at is None:
        generated_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    forecast_record = {
        "forecast_id":
            forecast_id,

        "generated_at":
            str(generated_at),

        "source":
            source,

        "input_week_start_date":
            (
                str(
                    input_week_start_date
                )
                if input_week_start_date
                else None
            ),

        "forecast_horizon_weeks":
            4,

        "tea_type_count":
            len(forecasts),

        "forecasts":
            forecasts,
    }

    collection.update_one(
        {
            "forecast_id":
                forecast_id,
        },
        {
            "$set":
                forecast_record,
        },
        upsert=True,
    )

    return forecast_record


def get_all_forecasts() -> list:
    """
    Return forecast history without
    exposing MongoDB ObjectIds.
    """

    create_forecast_indexes()

    collection = (
        get_forecast_collection()
    )

    documents = list(
        collection.find(
            {},
            {
                "_id": 0,
            },
        ).sort(
            "generated_at",
            ASCENDING,
        )
    )

    return documents


def get_forecast_by_id(
    forecast_id: str,
):
    """
    Find one forecast by its
    friendly forecast ID.
    """

    collection = (
        get_forecast_collection()
    )

    document = collection.find_one(
        {
            "forecast_id":
                forecast_id,
        },
        {
            "_id": 0,
        },
    )

    return document


def get_latest_operational_forecast():
    """
    Return the most recently saved
    operational forecast.
    """

    create_forecast_indexes()

    collection = (
        get_forecast_collection()
    )

    document = collection.find_one(
        {
            "source":
                "operational",
        },
        {
            "_id": 0,
        },
        sort=[
            (
                "generated_at",
                DESCENDING,
            )
        ],
    )

    return document


def count_forecasts() -> int:
    collection = (
        get_forecast_collection()
    )

    return collection.count_documents(
        {}
    )