from datetime import datetime, timezone

from pymongo import (
    ASCENDING,
    DESCENDING,
)

from app.db.mongodb import (
    get_database,
)

from app.schemas.weekly_input import (
    WeeklyBatchInput,
)


COLLECTION_NAME = "weekly_inputs"


def get_weekly_inputs_collection():
    database = get_database()

    return database[
        COLLECTION_NAME
    ]


def create_weekly_input_indexes():
    collection = (
        get_weekly_inputs_collection()
    )

    collection.create_index(
        [
            (
                "week_start_date",
                ASCENDING,
            )
        ],
        unique=True,
        name="unique_week_start_date",
    )

    collection.create_index(
        [
            (
                "updated_at",
                DESCENDING,
            )
        ],
        name="updated_at_desc",
    )


def upsert_weekly_input(
    batch_input: WeeklyBatchInput,
):
    collection = (
        get_weekly_inputs_collection()
    )

    create_weekly_input_indexes()


    now = datetime.now(
        timezone.utc
    )


    week_start_date = (
        batch_input
        .week_start_date
        .isoformat()
    )


    records = [
        record.model_dump()
        for record
        in batch_input.records
    ]


    result = collection.update_one(
        {
            "week_start_date":
                week_start_date,
        },
        {
            "$set": {
                "week_start_date":
                    week_start_date,

                "records":
                    records,

                "record_count":
                    len(records),

                "updated_at":
                    now,
            },

            "$setOnInsert": {
                "created_at":
                    now,
            },
        },
        upsert=True,
    )


    document = collection.find_one(
        {
            "week_start_date":
                week_start_date,
        }
    )


    if document is None:
        raise ValueError(
            "Weekly input could not be "
            "retrieved after saving."
        )


    return {
        "id":
            str(document["_id"]),

        "week_start_date":
            document[
                "week_start_date"
            ],

        "record_count":
            document[
                "record_count"
            ],

        "created":
            result.upserted_id
            is not None,

        "updated":
            result.matched_count
            > 0,
    }


def get_latest_weekly_input():
    """
    Return the most recently saved
    operational weekly input.
    """

    create_weekly_input_indexes()

    collection = (
        get_weekly_inputs_collection()
    )


    document = collection.find_one(
        {},
        {
            "_id": 0,
            "created_at": 0,
            "updated_at": 0,
        },
        sort=[
            (
                "week_start_date",
                DESCENDING,
            )
        ],
    )


    return document


def get_weekly_input_by_date(
    week_start_date: str,
):
    """
    Return one operational weekly
    input for a specific Monday.

    The week_start_date is stored
    in MongoDB as YYYY-MM-DD.
    """

    create_weekly_input_indexes()

    collection = (
        get_weekly_inputs_collection()
    )


    document = collection.find_one(
        {
            "week_start_date":
                str(
                    week_start_date
                ),
        },
        {
            "_id": 0,
            "created_at": 0,
            "updated_at": 0,
        },
    )


    return document