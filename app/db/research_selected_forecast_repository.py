
from datetime import datetime, timezone
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING

from app.db.mongodb import get_database


COLLECTION_NAME = "research_selected_forecast_runs"

VALID_INPUT_SOURCES = {
    "actual_factory_data",
    "test_data",
}

VALID_GRADES = {
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
}


def get_collection():
    database = get_database()
    return database[COLLECTION_NAME]


def create_indexes():
    collection = get_collection()

    collection.create_index(
        [("run_id", ASCENDING)],
        unique=True,
        name="unique_research_run_id",
    )

    collection.create_index(
        [
            ("forecast_origin", ASCENDING),
            ("generated_at", DESCENDING),
        ],
        name="research_origin_history",
    )

    collection.create_index(
        [
            ("selected_grades", ASCENDING),
            ("generated_at", DESCENDING),
        ],
        name="research_grade_history",
    )

    return collection


def save_selected_forecast_run(
    forecast_result: dict,
    input_source: str,
    input_records: list[dict],
):
    """
    Insert a new selective research TFT forecast.

    Every run is stored independently.
    Existing forecast runs are never overwritten.

    Test runs are explicitly distinguished
    from verified factory-data runs.
    """
    if input_source not in VALID_INPUT_SOURCES:
        raise ValueError(
            "input_source must be "
            "'actual_factory_data' or 'test_data'."
        )

    if forecast_result.get("status") != "success":
        raise ValueError(
            "Only successful forecasts can be saved."
        )

    if forecast_result.get("model") != "no_price_tft":
        raise ValueError(
            "Unexpected research forecasting model."
        )

    grades = forecast_result.get("selected_grades")

    if (
        not isinstance(grades, list)
        or not 1 <= len(grades) <= 5
        or len(grades) != len(set(grades))
        or not set(grades).issubset(VALID_GRADES)
    ):
        raise ValueError(
            "Invalid selected tea grades."
        )

    forecasts = forecast_result.get("forecasts")

    if not isinstance(forecasts, dict):
        raise ValueError(
            "Forecast data is missing."
        )

    if set(forecasts) != set(grades):
        raise ValueError(
            "Forecast grades do not match selection."
        )

    if any(
        not isinstance(forecasts[grade], list)
        or len(forecasts[grade]) != 4
        for grade in grades
    ):
        raise ValueError(
            "Each grade must have four forecasts."
        )

    if forecast_result.get("forecast_count") != (
        len(grades) * 4
    ):
        raise ValueError(
            "Invalid total forecast count."
        )

    if (
        not isinstance(input_records, list)
        or len(input_records) != len(grades)
        or {
            record.get("tea_type")
            for record in input_records
        } != set(grades)
    ):
        raise ValueError(
            "Input records do not match selected grades."
        )

    collection = create_indexes()

    run_id = f"research_{uuid4().hex}"

    document = {
        "run_id": run_id,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_source": input_source,
        "model": "no_price_tft",
        "forecast_type": (
            "selected_operational_forecast"
        ),
        "forecast_origin": forecast_result[
            "forecast_origin"
        ],
        "forecast_start_date": forecast_result[
            "forecast_start_date"
        ],
        "forecast_end_date": forecast_result[
            "forecast_end_date"
        ],
        "historical_input_weeks": 52,
        "forecast_horizon_weeks": 4,
        "selected_grades": grades,
        "tea_grade_count": len(grades),
        "forecast_count": len(grades) * 4,
        "unit": "kg",
        "quantiles": {
            "lower": 0.1,
            "median": 0.5,
            "upper": 0.9,
        },
        "input_records": input_records,
        "forecasts": forecasts,
    }

    collection.insert_one(document)

    return {
        key: value
        for key, value in document.items()
        if key != "_id"
    }


def get_selected_forecast_run(run_id: str):
    return get_collection().find_one(
        {"run_id": run_id},
        {"_id": 0},
    )


def get_selected_forecast_history(
    input_source=None,
    limit=50,
):
    if input_source is not None:
        if input_source not in VALID_INPUT_SOURCES:
            raise ValueError(
                "Invalid input_source filter."
            )

    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError(
            "limit must be between 1 and 100."
        )

    query = {}

    if input_source is not None:
        query["input_source"] = input_source

    return list(
        get_collection()
        .find(query, {"_id": 0})
        .sort("generated_at", DESCENDING)
        .limit(limit)
    )
