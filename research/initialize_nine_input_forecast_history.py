
"""
Initialize five historical nine-input TFT forecast origins.

Historical origins:
    2026-09-07
    2026-09-14
    2026-09-21
    2026-09-28
    2026-10-05

Each origin:
    52 historical input weeks
    5 tea grades
    4 future forecast weeks
    P10 / P50 / P90 demand estimates

Modes:
    python initialize_nine_input_forecast_history.py
        Preview only. No database writes.

    python initialize_nine_input_forecast_history.py --commit
        Save validated forecasts to MongoDB.

Safeguards:
    - Never modifies historical observations.
    - Never changes Weekly Data Entry restrictions.
    - Never overwrites finalized forecasts.
    - Rejects partial or conflicting forecast origins.
    - Uses the frozen production nine-input model.
    - Labels all initialized forecasts as backfilled.
    - Does not claim forecasts were issued historically.

IMPORTANT:
    This is retrospective model backfilling, not
    independent chronological backtesting.
"""

import argparse
import math
import sys

from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)

from app.services.eight_input_history_service import (
    load_original_factory_history,
)

from app.services.eight_input_forecast_service import (
    MODEL_NAME,
    MODEL_VERSION,
    generate_eight_input_forecast,
)

from app.services.eight_input_forecast_storage_service import (
    get_forecast_collection,
    save_final_forecasts_atomic,
    initialize_forecast_storage,
    normalize_forecast,
)


GRADES = (
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
)

ORIGINS = (
    "2026-09-07",
    "2026-09-14",
    "2026-09-21",
    "2026-09-28",
    "2026-10-05",
)

INPUT_FIELDS = (
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
    "festival_flag",
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
)

FORECAST_FIELDS = (
    "lower_demand_kg",
    "median_demand_kg",
    "upper_demand_kg",
)


def build_historical_batch(history, origin):
    """
    Reconstruct a forecasting request from
    authentic historical dataset observations.

    No historical observation is inserted
    into MongoDB.
    """

    week = date.fromisoformat(origin)

    records = []

    for grade in GRADES:
        matching = history.loc[
            (
                history["grade"] == grade
            )
            &
            (
                history["week_start_date"].dt.date
                == week
            )
        ]

        if len(matching) != 1:
            raise ValueError(
                f"{origin}: expected exactly one "
                f"historical record for {grade}."
            )

        row = matching.iloc[0]

        record = {
            "tea_type": grade,
        }

        for field in INPUT_FIELDS:
            value = float(row[field])

            if not math.isfinite(value):
                raise ValueError(
                    f"{origin} {grade}: invalid {field}."
                )

            if field == "festival_flag":
                value = int(value)

            record[field] = value

        records.append(record)

    return EightInputWeeklyBatch.model_validate({
        "week_start_date": origin,
        "records": records,
    })


def validate_result(result, origin):
    """
    Confirm model identity, five tea grades,
    four forecast dates and valid quantiles.

    Reuse the same document validator used
    for permanent MongoDB storage.
    """

    if result.get("model") != MODEL_NAME:
        raise ValueError(
            f"{origin}: unexpected model identity."
        )

    if result.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"{origin}: unexpected model version."
        )

    if result.get("forecast_origin") != origin:
        raise ValueError(
            f"{origin}: forecast origin mismatch."
        )

    if result.get("forecast_horizon_weeks") != 4:
        raise ValueError(
            f"{origin}: expected four forecast weeks."
        )

    if result.get("historical_input_weeks") != 52:
        raise ValueError(
            f"{origin}: expected 52 history weeks."
        )

    if set(result.get("selected_grades", [])) != (
        set(GRADES)
    ):
        raise ValueError(
            f"{origin}: forecasts are missing tea grades."
        )

    if result.get("forecast_count") != 20:
        raise ValueError(
            f"{origin}: expected 20 weekly predictions."
        )

    # Explicitly mark the retrospective nature.
    result["forecast_type"] = "historical_backfill"

    normalized = [
        normalize_forecast(result, grade)
        for grade in GRADES
    ]

    if len(normalized) != 5:
        raise ValueError(
            f"{origin}: incomplete normalized results."
        )

    for document in normalized:
        if not document.get("historical_backfill"):
            raise ValueError(
                f"{origin}: missing historical backfill label."
            )

        if document.get("issued_at_origin") is not False:
            raise ValueError(
                f"{origin}: incorrect historical provenance."
            )

        weekly = document["forecasts"]

        if len(weekly) != 4:
            raise ValueError(
                f"{origin}: incomplete four-week result."
            )

        for position, forecast in enumerate(
            weekly,
            start=1,
        ):
            expected_week = (
                date.fromisoformat(origin)
                + timedelta(weeks=position)
            ).isoformat()

            if forecast["week"] != expected_week:
                raise ValueError(
                    f"{origin}: incorrect forecast date."
                )

            lower, median, upper = [
                float(forecast[field])
                for field in FORECAST_FIELDS
            ]

            if not all(
                math.isfinite(value)
                for value in (lower, median, upper)
            ):
                raise ValueError(
                    f"{origin}: non-finite prediction."
                )

            if not 0 <= lower <= median <= upper:
                raise ValueError(
                    f"{origin}: invalid prediction interval."
                )

    return normalized


def inspect_existing_forecasts(origin):
    """
    Inspect existing records without modifying them.

    A complete existing origin is protected.
    A partial origin is considered a conflict.
    """

    collection = get_forecast_collection()

    existing = list(
        collection.find(
            {"forecast_origin": origin},
            {"_id": 0},
        )
    )

    if not existing:
        return "available"

    if len(existing) != len(GRADES):
        raise ValueError(
            f"{origin}: partial forecast history exists. "
            "Manual review is required."
        )

    grades = [
        document.get("tea_type")
        for document in existing
    ]

    if (
        len(set(grades)) != len(GRADES)
        or set(grades) != set(GRADES)
    ):
        raise ValueError(
            f"{origin}: existing grade records are invalid."
        )

    if not all(
        document.get("finalized") is True
        for document in existing
    ):
        raise ValueError(
            f"{origin}: unfinalized forecasts exist."
        )

    # Complete existing origins are deliberately
    # not replaced, regardless of model identity.
    return "already_exists"


def generate_backfills(history):
    """
    Generate and validate all five historical
    origins before permitting database writes.
    """

    results = []

    for origin in ORIGINS:
        print(f"\nGenerating origin: {origin}")

        batch = build_historical_batch(
            history,
            origin,
        )

        result = generate_eight_input_forecast(
            batch
        )

        validate_result(
            result,
            origin,
        )

        results.append(
            (origin, result)
        )

        print("  Tea grades: 5")
        print("  Forecast weeks per grade: 4")
        print("  Weekly predictions: 20")
        print("  Validation: PASS")

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Initialize five historical production "
            "TFT forecast origins."
        )
    )

    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Save forecasts to MongoDB. "
            "Without this flag, preview only."
        ),
    )

    arguments = parser.parse_args()

    mode = (
        "COMMIT"
        if arguments.commit
        else "PREVIEW ONLY"
    )

    print(
        "\n=== HISTORICAL FORECAST INITIALIZATION ==="
    )

    print("Mode:", mode)
    print("Model:", MODEL_NAME)
    print("Model version:", MODEL_VERSION)
    print("Historical origins:", len(ORIGINS))
    print("Tea grades:", len(GRADES))

    history = load_original_factory_history()

    print("Historical dataset records:", len(history))

    if len(history) != 1500:
        raise ValueError(
            "Expected the approved 1,500-row dataset."
        )

    latest_dataset_week = (
        history["week_start_date"].max().date()
    )

    if latest_dataset_week != date(2026, 10, 5):
        raise ValueError(
            "Dataset end date differs from "
            "the approved historical history."
        )

    # All five origins must first generate
    # successfully before attempting writes.
    results = generate_backfills(
        history
    )

    print("\n=== MONGODB CONFLICT CHECK ===")

    statuses = {}

    for origin, _ in results:
        status = inspect_existing_forecasts(
            origin
        )

        statuses[origin] = status

        print(f"{origin}: {status}")

    available = [
        origin
        for origin in ORIGINS
        if statuses[origin] == "available"
    ]

    already_existing = [
        origin
        for origin in ORIGINS
        if statuses[origin] == "already_exists"
    ]

    print("\n=== INITIALIZATION SUMMARY ===")
    print("Valid historical origins:", len(results))
    print("Forecast documents per origin:", 5)
    print("Weekly predictions per origin:", 20)
    print("Total weekly predictions:", 100)
    print("Available origins:", len(available))
    print("Protected existing origins:", len(already_existing))

    if not arguments.commit:
        print("\nDatabase writes: NONE")
        print("HISTORICAL FORECAST PREVIEW: PASS")
        print(
            "\nTo save later, run the same "
            "script with --commit."
        )
        return

    # Create/reuse the unique forecast index
    # only after explicit commit authorization.
    initialize_forecast_storage()

    saved_origins = 0
    saved_documents = 0

    for origin, result in results:
        if statuses[origin] == "already_exists":
            print(
                f"\nSkipping protected origin: {origin}"
            )
            continue

        # The atomic storage service rechecks
        # conflicts inside its transaction.
        saved = save_final_forecasts_atomic(
            result,
            GRADES,
        )

        if len(saved) != len(GRADES):
            raise RuntimeError(
                f"{origin}: incomplete save response."
            )

        saved_origins += 1
        saved_documents += len(saved)

        print(
            f"Saved {origin}: {len(saved)} forecasts"
        )

    print("\n=== DATABASE COMMIT RESULTS ===")
    print("New origins saved:", saved_origins)
    print("New forecast documents:", saved_documents)
    print(
        "Existing origins preserved:",
        len(already_existing),
    )
    print("Historical observations modified: NO")
    print("Weekly Data Entry modified: NO")
    print("\nHISTORICAL FORECAST INITIALIZATION: PASS")


if __name__ == "__main__":
    main()
