"""Freeze a dated auction forecast before future outcomes are published.

Usage: python research/freeze_auction_forecast.py

Stores only forecast weeks starting strictly after today's UTC date.
Does not retrain the model or overwrite an existing snapshot.
"""

import csv
import hashlib
import json
from datetime import (
    date,
    datetime,
    timezone,
)
from pathlib import Path


RESEARCH = Path(__file__).resolve().parent
OUTPUTS = RESEARCH / "outputs"

TARGETS = (
    "high_1000kg",
    "medium_1000kg",
    "low_1000kg",
)


def digest(path):
    checksum = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            checksum.update(block)

    return checksum.hexdigest()


def main():
    now = datetime.now(
        timezone.utc
    )

    forecast_file = (
        OUTPUTS
        / "latest_tft_forecast.json"
    )
    dataset_file = (
        OUTPUTS
        / "auction_calendar_weeks.csv"
    )
    checkpoint_file = (
        OUTPUTS
        / "checkpoints"
        / "auction-tft.ckpt"
    )

    sources = sorted(
        (
            RESEARCH / "sources"
        ).glob(
            "Weekly Tea Auction Quantities "
            "& Averages 2026*.pdf"
        )
    )

    if len(sources) != 1:
        raise ValueError(
            "Expected one original "
            "2026 source PDF; found "
            f"{len(sources)}"
        )

    for path in (
        forecast_file,
        dataset_file,
        checkpoint_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    forecast = json.loads(
        forecast_file.read_text(
            encoding="utf-8"
        )
    )

    if (
        forecast.get("target_kind")
        !=
        "observed_auction_sales_by_elevation"
    ):
        raise ValueError(
            "Unexpected forecast target"
        )

    origin = date.fromisoformat(
        forecast[
            "origin_week_start"
        ]
    )

    if origin > now.date():
        raise ValueError(
            "Forecast origin is "
            "in the future"
        )

    with dataset_file.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        weeks = list(
            csv.DictReader(handle)
        )

    if (
        not weeks
        or weeks[-1]["week_start"]
        != origin.isoformat()
    ):
        raise ValueError(
            "Forecast origin does "
            "not match the latest "
            "source week"
        )

    history = weeks[-8:]

    if (
        len(history) != 8
        or any(
            int(row["sale_count"])
            != 1
            for row in history
        )
    ):
        raise ValueError(
            "Latest eight weeks must "
            "each have one observed sale"
        )

    if (
        weeks[-1][
            "source_sale_dates"
        ]
        != forecast[
            "last_source_sale_date"
        ]
    ):
        raise ValueError(
            "Forecast last sale does "
            "not match source data"
        )

    rows = forecast.get(
        "forecasts", []
    )

    expected = {
        (target, horizon)
        for target in TARGETS
        for horizon in range(1, 5)
    }

    actual_keys = {
        (
            row.get("target"),
            row.get("horizon_week"),
        )
        for row in rows
    }

    if (
        len(rows) != 12
        or actual_keys != expected
    ):
        raise ValueError(
            "Forecast must contain "
            "four weeks for three "
            "elevations"
        )

    future = []

    for row in rows:
        target_week = date.fromisoformat(
            row["week_start"]
        )

        if target_week <= now.date():
            # This week has already begun.
            continue

        target = row["target"]

        past_four = [
            float(item[target])
            for item in history[-4:]
        ]

        future.append({
            "target": target,
            "target_week_start":
                target_week.isoformat(),
            "horizon_week":
                row["horizon_week"],
            "tft_p10_1000kg":
                row["p10_1000kg"],
            "tft_p50_1000kg":
                row["p50_1000kg"],
            "tft_p90_1000kg":
                row["p90_1000kg"],
            "last_baseline_1000kg":
                float(
                    history[-1][target]
                ),
            "mean4_baseline_1000kg":
                round(
                    sum(past_four)
                    / 4,
                    4,
                ),
        })

    if not future:
        raise ValueError(
            "No future forecast weeks "
            "remain; refresh auction "
            "source data"
        )

    snapshot = {
        "issued_at_utc":
            now.isoformat(),
        "forecast_origin_week_start":
            origin.isoformat(),
        "last_source_sale_date":
            forecast[
                "last_source_sale_date"
            ],
        "target_kind":
            "observed_auction_sales_by_elevation",
        "unit":
            "thousand kilograms",
        "caution": (
            "Origin data may be old. "
            "These predictions were "
            "saved before target weeks "
            "began; accuracy is unknown "
            "until independently "
            "published outcomes are "
            "available."
        ),
        "sha256": {
            "auction_calendar_weeks.csv":
                digest(dataset_file),
            "auction-tft.ckpt":
                digest(checkpoint_file),
            sources[0].name:
                digest(sources[0]),
        },
        "forecasts": sorted(
            future,
            key=lambda row: (
                row[
                    "target_week_start"
                ],
                row["target"],
            ),
        ),
    }

    folder = (
        OUTPUTS / "prospective"
    )
    folder.mkdir(
        exist_ok=True
    )

    output = folder / (
        "auction_forecast_"
        f"{now.strftime('%Y%m%dT%H%M%SZ')}"
        ".json"
    )

    # Exclusive creation prevents replacing
    # an existing snapshot.
    with output.open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            snapshot,
            handle,
            indent=2,
        )
        handle.write("\n")

    print(
        json.dumps(
            {
                "snapshot":
                    str(output),
                "issued_at_utc":
                    snapshot[
                        "issued_at_utc"
                    ],
                "future_rows":
                    len(future),
                "future_week_starts":
                    sorted({
                        row[
                            "target_week_start"
                        ]
                        for row in future
                    }),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()