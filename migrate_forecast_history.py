import json
import sys

from pathlib import Path


# Add the backend project root
# to Python's import path.
PROJECT_ROOT = Path(
    __file__
).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.db.forecast_repository import (
    count_forecasts,
    save_forecast_to_database,
)


FORECAST_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecast_history.json"
)


def migrate_forecast_history():
    if not FORECAST_HISTORY_FILE.exists():
        print(
            "forecast_history.json "
            "was not found."
        )
        return


    with open(
        FORECAST_HISTORY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        history = json.load(file)


    if not isinstance(
        history,
        list,
    ):
        print(
            "Invalid forecast history "
            "format."
        )
        return


    print(
        "JSON forecast records:",
        len(history),
    )

    print(
        "MongoDB records before:",
        count_forecasts(),
    )


    migrated = 0


    for record in history:
        forecast_id = record.get(
            "forecast_id"
        )

        forecasts = record.get(
            "forecasts",
            []
        )


        if not forecast_id:
            print(
                "Skipping record without "
                "forecast_id."
            )
            continue


        if not forecasts:
            print(
                f"Skipping {forecast_id}: "
                "no forecasts."
            )
            continue


        save_forecast_to_database(
            forecasts=forecasts,
            input_week_start_date=(
                record.get(
                    "input_week_start_date"
                )
            ),
            source=record.get(
                "source",
                "research",
            ),
            forecast_id=forecast_id,
            generated_at=record.get(
                "generated_at"
            ),
        )


        migrated += 1

        print(
            f"Migrated: {forecast_id}"
        )


    print()

    print(
        "Migration completed."
    )

    print(
        "Records processed:",
        migrated,
    )

    print(
        "MongoDB records after:",
        count_forecasts(),
    )


if __name__ == "__main__":
    migrate_forecast_history()