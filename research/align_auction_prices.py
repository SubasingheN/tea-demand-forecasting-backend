"""Join past auction prices to the existing eight-week volume windows.

Usage: python research/align_auction_prices.py
No price from a forecast horizon is included in the input windows.
"""

import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs"

ELEVATIONS = {
    "high_1000kg": "high_lkr_per_kg",
    "medium_1000kg": "medium_lkr_per_kg",
    "low_1000kg": "low_lkr_per_kg",
}


def rows_from(filename):
    with (
        ROOT / filename
    ).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def week_start(day):
    return day - timedelta(days=day.weekday())


def main():
    windows = rows_from("tft_8week_windows.csv")
    prices = rows_from("auction_prices_2024_2026.csv")
    sales = rows_from("auction_sales_2024_2026.csv")

    price_by_week = {}
    sale_by_week = {}

    for row in prices:
        sale_date = date.fromisoformat(row["sale_date"])
        week = week_start(sale_date)
        if week in price_by_week:
            raise ValueError(
                f"More than one priced sale in week {week}"
            )
        price_by_week[week] = row

    for row in sales:
        week = week_start(
            date.fromisoformat(row["sale_date"])
        )
        if week in sale_by_week:
            raise ValueError(
                f"More than one sale in week {week}"
            )
        sale_by_week[week] = row

    if set(price_by_week) != set(sale_by_week):
        raise ValueError(
            "Price and quantity weeks do not match"
        )

    enriched = []
    counts = Counter()

    for window in windows:
        target = window["target"]
        if target not in ELEVATIONS:
            raise ValueError(
                f"Unexpected target {target}"
            )

        origin = date.fromisoformat(
            window["origin_week"]
        )
        if week_start(origin) != origin:
            raise ValueError(
                f"Origin is not a Monday: {origin}"
            )

        item = dict(window)

        for week_number in range(1, 9):
            week = origin + timedelta(
                weeks=week_number - 8
            )
            if week not in price_by_week:
                raise ValueError(
                    f"Missing historical price in week {week}"
                )

            priced = price_by_week[week]
            sale_date = date.fromisoformat(
                priced["sale_date"]
            )
            if sale_date >= origin + timedelta(weeks=1):
                raise ValueError(
                    f"Future price leaked into origin {origin}"
                )

            volume = float(
                sale_by_week[week][target]
            )
            history_volume = float(
                window[
                    f"history_week_{week_number}_1000kg"
                ]
            )
            if volume != history_volume:
                raise ValueError(
                    f"Quantity misaligned for "
                    f"{target} on {week}"
                )

            price = float(
                priced[ELEVATIONS[target]]
            )
            if price <= 0:
                raise ValueError(
                    f"Invalid price for {target} on {week}"
                )

            item[
                f"history_price_{week_number}_lkr_per_kg"
            ] = price

        for horizon in range(1, 5):
            expected = origin + timedelta(
                weeks=horizon
            )
            observed = date.fromisoformat(
                window[f"future_week_{horizon}_date"]
            )
            if observed != expected:
                raise ValueError(
                    f"Forecast horizon mismatch: "
                    f"{origin}, {observed}"
                )

        enriched.append(item)
        counts[(window["split"], target)] += 1

    if not enriched:
        raise ValueError("No eligible windows")

    path = ROOT / "tft_8week_price_windows.csv"
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=enriched[0].keys(),
        )
        writer.writeheader()
        writer.writerows(enriched)

    summary = {
        "window_rows": len(enriched),
        "counts": {
            f"{split}/{target}": count
            for (split, target), count
            in sorted(counts.items())
        },
        "history_weeks": 8,
        "forecast_weeks": 4,
        "price_fields": (
            "eight past observed elevation-specific "
            "auction prices, LKR per kg"
        ),
        "future_prices_in_input": False,
        "output": str(path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()