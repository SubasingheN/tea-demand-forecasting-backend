
"""
Automatic four-week forecasting after factory data saving.

Production model:
    nine_input_tft / production_v1

Responsibilities:
- Require five tea grades for every saved reporting week.
- Verify that all observations are already in MongoDB.
- Verify submitted observations match saved observations.
- Reuse complete finalized forecasts.
- Reject incomplete or inconsistent forecast sets.
- Generate predictions with the production TFT.
- Persist all five grade forecasts atomically.
- Never overwrite finalized forecasts.
- Never alter saved observations or training records.

Normal Weekly Data Entry remains restricted to
the latest completed reporting week.

Historical forecast initialization uses a separate
backend procedure.

No model training or database writes on import.
"""

from datetime import date

from pymongo.errors import DuplicateKeyError

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_observation_storage_service import (
    TEA_GRADES,
    get_observation_collection,
    normalize_observation,
)

from app.services.eight_input_forecast_service import (
    MODEL_NAME,
    MODEL_VERSION,
    generate_eight_input_forecast,
)

from app.services.eight_input_forecast_storage_service import (
    get_forecast_collection,
    initialize_forecast_storage,
    save_final_forecasts_atomic,
)


FORECAST_HORIZON = 4


def validate_complete_batch(
    batch: EightInputWeeklyBatch,
):
    """Require one valid record for each tea grade."""

    if not isinstance(batch, EightInputWeeklyBatch):
        raise ValueError(
            "A valid weekly factory batch is required."
        )

    submitted = [
        record.tea_type
        for record in batch.records
    ]

    if (
        len(submitted) != len(TEA_GRADES)
        or set(submitted) != set(TEA_GRADES)
        or len(submitted) != len(set(submitted))
    ):
        raise ValueError(
            "Automatic forecasting requires exactly "
            "one record for each of the five tea grades."
        )

    return batch.week_start_date.isoformat()


def verify_saved_week(
    batch: EightInputWeeklyBatch,
):
    """
    Verify that a complete reporting week
    exists in MongoDB.

    Request values must match the saved values.
    No records are modified.
    """

    origin = validate_complete_batch(batch)

    collection = get_observation_collection()

    documents = list(
        collection.find(
            {"week_start_date": origin},
            {"_id": 0},
        )
    )

    saved_by_grade = {}

    for document in documents:
        grade = document.get("tea_type")

        if grade not in TEA_GRADES:
            raise ValueError(
                "An unsupported tea grade was found "
                "in saved factory observations."
            )

        if grade in saved_by_grade:
            raise ValueError(
                "Duplicate saved tea-grade observations "
                "were found for this reporting week."
            )

        saved_by_grade[grade] = document

    if set(saved_by_grade) != set(TEA_GRADES):
        missing = [
            grade
            for grade in TEA_GRADES
            if grade not in saved_by_grade
        ]

        raise ValueError(
            "The reporting week is incomplete. Missing: "
            + ", ".join(missing)
        )

    for record in batch.records:
        expected = normalize_observation(
            {
                "week_start_date": origin,
                **record.model_dump(),
            }
        )

        stored = normalize_observation(
            saved_by_grade[record.tea_type]
        )

        for field, value in expected.items():
            recorded = stored[field]

            if isinstance(value, (int, float)):
                if abs(
                    float(value) - float(recorded)
                ) > 0.001:
                    raise ValueError(
                        f"{record.tea_type}: {field} "
                        "does not match saved factory data."
                    )
            elif value != recorded:
                raise ValueError(
                    "Submitted weekly observations "
                    "do not match saved factory data."
                )

    return origin


def get_existing_origin_forecasts(origin):
    """
    Retrieve all existing forecast documents
    for one reporting-week origin.

    Includes incomplete and non-finalized records,
    so they cannot be silently overwritten.
    """

    collection = get_forecast_collection()

    documents = list(
        collection.find(
            {"forecast_origin": origin},
            {"_id": 0},
        )
    )

    existing = {}

    for document in documents:
        grade = document.get("tea_type")

        if grade not in TEA_GRADES:
            raise ValueError(
                "Unexpected tea grade in forecast history."
            )

        if grade in existing:
            raise ValueError(
                "Duplicate forecast documents were found "
                "for this reporting week."
            )

        existing[grade] = document

    return existing


def build_existing_response(origin, existing):
    """Return an intact previously finalized forecast set."""

    if set(existing) != set(TEA_GRADES):
        raise ValueError(
            "Forecast records already exist for this "
            "reporting week, but the set is incomplete. "
            "Review forecast history before retrying."
        )

    for grade, document in existing.items():
        if document.get("finalized") is not True:
            raise ValueError(
                f"{grade}: forecast exists but has "
                "not been finalized."
            )

        if document.get("model") != MODEL_NAME:
            raise ValueError(
                f"{grade}: the saved forecast belongs "
                "to a different model version and "
                "must not be silently replaced."
            )

        saved_version = document.get(
            "model_version",
            MODEL_VERSION,
        )

        if saved_version != MODEL_VERSION:
            raise ValueError(
                f"{grade}: saved forecast model "
                "version does not match production."
            )

        weekly_forecasts = document.get(
            "forecasts"
        )

        if (
            not isinstance(weekly_forecasts, list)
            or len(weekly_forecasts) != FORECAST_HORIZON
        ):
            raise ValueError(
                f"{grade}: stored four-week forecast "
                "is incomplete."
            )

        expected_origin = date.fromisoformat(
            origin
        )

        for horizon, forecast in enumerate(
            weekly_forecasts,
            start=1,
        ):
            if not isinstance(forecast, dict):
                raise ValueError(
                    f"{grade}: invalid forecast entry."
                )

            expected_week = (
                expected_origin
                + __import__("datetime").timedelta(
                    weeks=horizon
                )
            ).isoformat()

            if forecast.get("week") != expected_week:
                raise ValueError(
                    f"{grade}: forecast week {horizon} "
                    "does not match its saved origin."
                )

            if forecast.get("horizon") != horizon:
                raise ValueError(
                    f"{grade}: invalid forecast horizon."
                )

            lower = forecast.get("lower_demand_kg")
            median = forecast.get("median_demand_kg")
            upper = forecast.get("upper_demand_kg")

            try:
                lower = float(lower)
                median = float(median)
                upper = float(upper)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{grade}: saved forecasts contain "
                    "invalid demand estimates."
                ) from exc

            if not 0 <= lower <= median <= upper:
                raise ValueError(
                    f"{grade}: saved forecast quantiles "
                    "are not correctly ordered."
                )

    return {
        "status": "success",
        "forecast_status": "already_finalized",
        "forecast_origin": origin,
        "forecast_generated": False,
        "finalized": True,
        "tea_grade_count": len(TEA_GRADES),
        "forecast_count": (
            len(TEA_GRADES) * FORECAST_HORIZON
        ),
        "forecasts": {
            grade: existing[grade]["forecasts"]
            for grade in TEA_GRADES
        },
        "message": (
            "The four-week demand forecast was "
            "already finalized for this week."
        ),
    }


def generate_and_finalize_saved_week(
    batch: EightInputWeeklyBatch,
):
    """
    Generate and finalize the four-week forecast
    for a complete saved factory reporting week.

    Safe to retry when a complete production
    forecast already exists.

    Existing finalized forecasts are immutable.
    """

    origin = verify_saved_week(batch)

    initialize_forecast_storage()

    existing = get_existing_origin_forecasts(
        origin
    )

    if existing:
        return build_existing_response(
            origin,
            existing,
        )

    result = generate_eight_input_forecast(
        batch
    )

    if result.get("model") != MODEL_NAME:
        raise ValueError(
            "The forecasting service returned "
            "an unexpected model identity."
        )

    if result.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "The forecasting service returned "
            "an unexpected production model version."
        )

    if result.get("forecast_origin") != origin:
        raise ValueError(
            "Forecast origin does not match "
            "the saved reporting week."
        )

    selected_grades = result.get(
        "selected_grades"
    )

    if (
        not isinstance(selected_grades, list)
        or len(selected_grades) != len(TEA_GRADES)
        or set(selected_grades) != set(TEA_GRADES)
    ):
        raise ValueError(
            "The production model did not return "
            "all five tea-grade forecasts."
        )

    if result.get(
        "forecast_horizon_weeks"
    ) != FORECAST_HORIZON:
        raise ValueError(
            "The production model must return "
            "four consecutive forecast weeks."
        )

    try:
        saved = save_final_forecasts_atomic(
            result,
            TEA_GRADES,
        )

    except (DuplicateKeyError, ValueError):
        # Another request may have finalized the
        # same forecast while this one was running.
        concurrent = get_existing_origin_forecasts(
            origin
        )

        if concurrent:
            return build_existing_response(
                origin,
                concurrent,
            )

        raise

    return {
        "status": "success",
        "forecast_status": "newly_finalized",
        "forecast_origin": origin,
        "forecast_start_date": result.get(
            "forecast_start_date"
        ),
        "forecast_end_date": result.get(
            "forecast_end_date"
        ),
        "forecast_generated": True,
        "finalized": True,
        "tea_grade_count": len(TEA_GRADES),
        "forecast_count": (
            len(TEA_GRADES) * FORECAST_HORIZON
        ),
        "saved_count": len(saved),
        "forecasts": result["forecasts"],
        "message": (
            "The next four weeks of tea demand "
            "were forecast and saved successfully."
        ),
    }
