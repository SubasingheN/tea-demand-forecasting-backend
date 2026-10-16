
"""
Isolated end-to-end test for six-input TFT forecasting.

Tests:
1. Save synthetic BOP and BOPF observations
2. Retrieve MongoDB historical observations
3. Reject duplicate observations
4. Generate four-week P10/P50/P90 TFT forecasts
5. Clean up test-created MongoDB records

Run ONLY with:
    MONGODB_DATABASE=tea_demand_forecasting_test

No production dataset, checkpoint, or system clock is changed.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=ROOT / ".env",
    override=True,
)

from fastapi.testclient import TestClient

from app.db.mongodb import get_database
from app.main import app

import app.api.six_input_observations as observation_api
import app.api.six_input_history as history_api

import app.services.six_input_observation_storage_service as storage
import app.services.six_input_history_service as history


TEST_DATABASE = "tea_demand_forecasting_test"
TEST_WEEK = "2026-10-12"
SIMULATED_TODAY = date(2026, 10, 19)

TEST_GRADES = ("BOP", "BOPF")
TEST_MARKER = "synthetic_e2e_" + uuid4().hex

OBSERVATION_COLLECTION = storage.COLLECTION_NAME

METADATA_COLLECTION = (
    observation_api.WEEK_METADATA_COLLECTION
)


class SimulatedHistoryDate(date):
    """
    Used only by the historical-data services
    that check today's date.
    """

    @classmethod
    def today(cls):
        return SIMULATED_TODAY


def check_database():
    database = get_database()

    if database.name != TEST_DATABASE:
        raise RuntimeError(
            "STOP: Incorrect database. "
            f"Expected {TEST_DATABASE}, "
            f"received {database.name}."
        )

    return database


def make_test_batch():
    """
    Synthetic observations, NOT real factory data.
    """

    return {
        "week_start_date": TEST_WEEK,
        "records": [
            {
                "tea_type": "BOP",
                "demand_kg": 4100,
                "production_kg": 4250,
                "inventory_kg": 300,
                "usd_lkr_rate": 335,
                "festival_flag": 0,
                "colombo_auction_price_lkr": 1720,
            },
            {
                "tea_type": "BOPF",
                "demand_kg": 4700,
                "production_kg": 4800,
                "inventory_kg": 250,
                "usd_lkr_rate": 335,
                "festival_flag": 0,
                "colombo_auction_price_lkr": 2190,
            },
        ],
    }


def simulated_completed_week_validation(week_start):
    """
    Validate the reporting week using a simulated
    reference date without replacing Python's
    built-in date class.

    Other production input checks remain active.
    """

    normalized = storage.normalize_week_date(
        week_start
    )

    end_date = normalized + timedelta(days=6)

    if end_date >= SIMULATED_TODAY:
        raise ValueError(
            "The reporting week has not finished. "
            "Submit only finalized factory observations."
        )


def assert_status(response, expected, step):
    if response.status_code != expected:
        raise RuntimeError(
            f"{step} failed. "
            f"Expected HTTP {expected}, "
            f"received HTTP {response.status_code}. "
            f"Response: {response.text[:1500]}"
        )

    return response.json()


def main():
    database = check_database()

    observations = database[
        OBSERVATION_COLLECTION
    ]

    metadata = database[
        METADATA_COLLECTION
    ]

    # Refuse to use an occupied reporting week.
    existing_records = observations.count_documents(
        {"week_start_date": TEST_WEEK}
    )

    existing_metadata = metadata.count_documents(
        {"_id": TEST_WEEK}
    )

    if existing_records or existing_metadata:
        raise RuntimeError(
            "STOP: Existing test-week data found. "
            "No database writes performed."
        )

    original_validator = (
        observation_api.validate_batch_for_storage
    )

    def tagged_validator(batch):
        submission = original_validator(batch)

        # Attach markers before MongoDB insertion.
        for document in submission["documents"]:
            document["test_marker"] = TEST_MARKER
            document["test_data"] = True

        return submission

    batch = make_test_batch()

    print("=" * 60)
    print("SIX-INPUT MONGODB END-TO-END TEST")
    print("=" * 60)

    print("Database:", database.name)
    print("Test week:", TEST_WEEK)
    print("Simulated date:", SIMULATED_TODAY)
    print("Test marker:", TEST_MARKER)

    test_started = False
    test_passed = False
    cleanup_passed = False

    try:
        # Only the completed-week validation is
        # substituted in the storage service.
        #
        # The storage.date class is NOT replaced:
        # isinstance(value, date) keeps working.
        #
        # Historical services see a simulated
        # date for their date.today() checks.
        with (
            patch.object(
                storage,
                "validate_completed_week",
                simulated_completed_week_validation,
            ),
            patch.object(
                history,
                "date",
                SimulatedHistoryDate,
            ),
            patch.object(
                history_api,
                "date",
                SimulatedHistoryDate,
            ),
            patch.object(
                observation_api,
                "validate_batch_for_storage",
                tagged_validator,
            ),
        ):
            test_started = True

            with TestClient(
                app,
                raise_server_exceptions=False,
            ) as client:

                print(
                    "\nTEST 1: Save synthetic observations"
                )

                response = client.post(
                    "/factory/six-input/observations",
                    json=batch,
                )

                result = assert_status(
                    response,
                    201,
                    "Observation saving",
                )

                if (
                    result.get("status") != "success"
                    or result.get("saved") is not True
                    or result.get("saved_count") != 2
                ):
                    raise RuntimeError(
                        "Incorrect save confirmation."
                    )

                saved_records = list(
                    observations.find(
                        {
                            "week_start_date": TEST_WEEK,
                            "test_marker": TEST_MARKER,
                            "test_data": True,
                        }
                    )
                )

                if len(saved_records) != 2:
                    raise RuntimeError(
                        "Expected two tagged observations."
                    )

                if {
                    record["tea_type"]
                    for record in saved_records
                } != set(TEST_GRADES):
                    raise RuntimeError(
                        "Incorrect saved tea grades."
                    )

                print(
                    "PASS: Two observations saved."
                )

                print(
                    "\nTEST 2: Historical retrieval"
                )

                response = client.get(
                    "/factory/six-input/history/"
                    + TEST_WEEK
                )

                result = assert_status(
                    response,
                    200,
                    "Historical retrieval",
                )

                if (
                    set(result.get("selected_grades", []))
                    != set(TEST_GRADES)
                ):
                    raise RuntimeError(
                        "Historical grades mismatch."
                    )

                if result.get("record_count") != 2:
                    raise RuntimeError(
                        "Incorrect historical record count."
                    )

                if (
                    result.get("source")
                    != "mongodb_factory_observations"
                ):
                    raise RuntimeError(
                        "Incorrect historical source."
                    )

                print(
                    "PASS: MongoDB history retrieved."
                )

                print(
                    "\nTEST 3: Duplicate protection"
                )

                response = client.post(
                    "/factory/six-input/observations",
                    json=batch,
                )

                assert_status(
                    response,
                    409,
                    "Duplicate protection",
                )

                print(
                    "PASS: Duplicate observations rejected."
                )

                print(
                    "\nTEST 4: Four-week TFT forecasting"
                )

                response = client.post(
                    "/forecast/six-input/selected",
                    json=batch,
                )

                result = assert_status(
                    response,
                    200,
                    "TFT forecasting",
                )

                if result.get("status") != "success":
                    raise RuntimeError(
                        "Forecasting status was not success."
                    )

                if result.get("model") != "six_input_tft":
                    raise RuntimeError(
                        "Unexpected model."
                    )

                if result.get("forecast_count") != 8:
                    raise RuntimeError(
                        "Expected eight forecast rows."
                    )

                if (
                    result.get("historical_input_weeks")
                    != 52
                ):
                    raise RuntimeError(
                        "Expected a 52-week encoder."
                    )

                if (
                    result.get("forecast_horizon_weeks")
                    != 4
                ):
                    raise RuntimeError(
                        "Expected a four-week horizon."
                    )

                expected_dates = [
                    "2026-10-19",
                    "2026-10-26",
                    "2026-11-02",
                    "2026-11-09",
                ]

                forecasts = result.get(
                    "forecasts",
                    {},
                )

                for grade in TEST_GRADES:
                    rows = forecasts.get(
                        grade,
                        [],
                    )

                    if len(rows) != 4:
                        raise RuntimeError(
                            f"{grade}: Expected four rows."
                        )

                    actual_dates = [
                        row["week"]
                        for row in rows
                    ]

                    if actual_dates != expected_dates:
                        raise RuntimeError(
                            f"{grade}: Forecast dates mismatch."
                        )

                    medians = []

                    for row in rows:
                        p10 = row["lower_demand_kg"]
                        p50 = row["median_demand_kg"]
                        p90 = row["upper_demand_kg"]

                        if not (
                            0 <= p10 <= p50 <= p90
                        ):
                            raise RuntimeError(
                                f"{grade}: Invalid P10/P50/P90."
                            )

                        medians.append(p50)

                    print(
                        grade,
                        "P50:",
                        medians,
                    )

                print(
                    "PASS: Four-week TFT forecasts "
                    "generated for both grades."
                )

                test_passed = True

    finally:
        if test_started:
            print(
                "\nTEST 5: Cleanup synthetic records"
            )

            # Recheck database isolation.
            check_database()

            deletion = observations.delete_many(
                {
                    "week_start_date": TEST_WEEK,
                    "test_marker": TEST_MARKER,
                    "test_data": True,
                }
            )

            print(
                "Deleted tagged observations:",
                deletion.deleted_count,
            )

            remaining_week_records = (
                observations.count_documents(
                    {"week_start_date": TEST_WEEK}
                )
            )

            # The metadata did not exist before
            # this test. Remove it only if there
            # are no remaining observations.
            if remaining_week_records == 0:
                metadata.delete_one(
                    {"_id": TEST_WEEK}
                )

            remaining_tagged = (
                observations.count_documents(
                    {"test_marker": TEST_MARKER}
                )
            )

            remaining_metadata = (
                metadata.count_documents(
                    {"_id": TEST_WEEK}
                )
            )

            cleanup_passed = (
                remaining_week_records == 0
                and remaining_tagged == 0
                and remaining_metadata == 0
            )

            print(
                "Remaining week observations:",
                remaining_week_records,
            )

            print(
                "Remaining week metadata:",
                remaining_metadata,
            )

            if cleanup_passed:
                print(
                    "PASS: Synthetic database data cleaned up."
                )
            else:
                raise RuntimeError(
                    "CLEANUP INCOMPLETE. "
                    "Inspect the test database."
                )

    if test_passed and cleanup_passed:
        print("\n" + "=" * 60)
        print("PASS: ALL END-TO-END TESTS COMPLETED")
        print("=" * 60)


if __name__ == "__main__":
    main()
