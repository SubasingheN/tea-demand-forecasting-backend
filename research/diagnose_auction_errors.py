"""Locate weak forecast dates on the matched 2026 auction test set.

Usage: python research/diagnose_auction_errors.py
Forecast origins overlap; summaries are descriptive.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs"

TARGETS = (
    "high_1000kg",
    "medium_1000kg",
    "low_1000kg",
)
METHODS = ("last", "mean4", "tft")


def read_rows(filename):
    with (
        ROOT / filename
    ).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return (
        row["origin_week"],
        row["target_week"],
        int(row["horizon_week"]),
        row["target"],
    )


def main():
    baseline = [
        row
        for row in read_rows(
            "baseline_matched_8week.csv"
        )
        if row["target"] in TARGETS
    ]
    predictions = [
        row
        for row in read_rows(
            "tft_predictions.csv"
        )
        if row["target"] in TARGETS
    ]

    actual = {}
    values = {}

    for row in baseline:
        identifier = key(row)
        observed = float(
            row["actual_1000kg"]
        )

        if (
            identifier in actual
            and actual[identifier] != observed
        ):
            raise ValueError(
                f"Conflicting observed sales "
                f"for {identifier}"
            )

        actual[identifier] = observed
        method_key = (
            identifier,
            row["method"],
        )

        if method_key in values:
            raise ValueError(
                f"Duplicate baseline row: "
                f"{method_key}"
            )

        values[method_key] = float(
            row["predicted_1000kg"]
        )

    seen = set()

    for row in predictions:
        identifier = key(row)

        if identifier in seen:
            raise ValueError(
                f"Duplicate TFT prediction: "
                f"{identifier}"
            )

        seen.add(identifier)

        if identifier not in actual:
            raise ValueError(
                f"TFT prediction outside "
                f"matched test: {identifier}"
            )

        values[
            (identifier, "tft")
        ] = float(
            row["predicted_1000kg"]
        )

    if seen != set(actual):
        raise ValueError(
            f"Missing "
            f"{len(set(actual) - seen)} "
            f"matched TFT predictions"
        )

    origins = defaultdict(list)

    for identifier, observed in actual.items():
        origin, target_week, horizon, target = (
            identifier
        )

        for method in METHODS:
            method_key = (
                identifier,
                method,
            )

            if method_key not in values:
                raise ValueError(
                    f"Missing {method} "
                    f"forecast for "
                    f"{identifier}"
                )

            origins[
                (target, origin, method)
            ].append(
                abs(
                    observed
                    - values[method_key]
                )
            )

    rows = []

    for (
        target,
        origin,
        method,
    ), errors in sorted(origins.items()):
        if len(errors) != 4:
            raise ValueError(
                f"Expected four forecast "
                f"weeks: {target}, "
                f"{origin}, {method}"
            )

        rows.append({
            "target": target,
            "origin_week": origin,
            "method": method,
            "mae_four_weeks_1000kg":
                round(
                    sum(errors) / 4,
                    2,
                ),
        })

    output = (
        ROOT
        / "auction_error_by_origin.csv"
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    by_target = {}

    for target in TARGETS:
        target_origins = sorted({
            row["origin_week"]
            for row in rows
            if row["target"] == target
        })

        midpoint = (
            len(target_origins) // 2
        )

        if len(target_origins) < 2:
            raise ValueError(
                f"Too few test origins "
                f"for {target}"
            )

        periods = {
            "earlier": set(
                target_origins[:midpoint]
            ),
            "later": set(
                target_origins[midpoint:]
            ),
        }

        mean_errors = {}

        for period, dates in periods.items():
            mean_errors[period] = {}

            for method in METHODS:
                matching = [
                    row[
                        "mae_four_weeks_1000kg"
                    ]
                    for row in rows
                    if (
                        row["target"]
                        == target
                        and row["method"]
                        == method
                        and row["origin_week"]
                        in dates
                    )
                ]

                mean_errors[period][method] = (
                    round(
                        sum(matching)
                        / len(matching),
                        2,
                    )
                )

        worst = sorted(
            (
                row
                for row in rows
                if (
                    row["target"]
                    == target
                    and row["method"]
                    == "tft"
                )
            ),
            key=lambda row:
                row[
                    "mae_four_weeks_1000kg"
                ],
            reverse=True,
        )[:3]

        by_target[target] = {
            "origins":
                len(target_origins),
            "mean_origin_mae_1000kg":
                mean_errors,
            "worst_three_tft_origins":
                worst,
        }

    summary = {
        "target_kind": (
            "observed_auction_sales_"
            "by_elevation"
        ),
        "unit": "thousand kg",
        "caution": (
            "Overlapping four-week forecast "
            "windows; errors are descriptive. "
            "Test outcomes were inspected "
            "during development."
        ),
        "by_elevation": by_target,
    }

    (
        ROOT
        / "auction_error_diagnostics.json"
    ).write_text(
        json.dumps(summary, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()