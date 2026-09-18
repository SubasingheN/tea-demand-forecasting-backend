"""Extract public auction sales and benchmark four-sale forecasts.

Usage: python research/auction_audit.py research/sources
The source PDFs remain unchanged. Results are written to research/outputs/.
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader


DATE_ROW = re.compile(
    r"^\s*(\d{2}-[A-Z]{3,4}-\d{2,4})\s+(.+)$"
)
NUMBER = re.compile(
    r"^\d[\d,]*(?:\.\d+)?$"
)

SOURCE_YEARS = (2024, 2025, 2026)

# Explicit corrections for printed source dates.
DATE_CORRECTIONS = {
    (2025, "14-MAT-2025"): "14-MAY-2025",
    (2026, "03-JUNE-2026"): "03-JUN-2026",
    (2026, "01-JUL-2027"): "01-JUL-2026",
}


def extract_year(path, year):
    reader = PdfReader(str(path))

    raw = "\n".join(
        page.extract_text(extraction_mode="layout")
        for page in reader.pages
    )

    rows = []
    issues = []

    for line_number, line in enumerate(
        raw.splitlines(), 1
    ):
        match = DATE_ROW.match(line)

        if not match:
            continue

        original_date, rest = match.groups()
        tokens = rest.split()

        # The first four numeric columns are
        # High, Medium, Low, and Total quantity.
        # One source PDF splits a price decimal
        # into two tokens, but those four
        # quantity columns remain readable.
        if (
            len(tokens) < 4
            or not all(
                NUMBER.fullmatch(value)
                for value in tokens[:4]
            )
        ):
            issues.append(
                f"{path.name}:{line_number}: "
                "cannot parse sale row"
            )
            continue

        corrected_date = (
            DATE_CORRECTIONS.get(
                (year, original_date),
                original_date,
            )
        )

        if corrected_date != original_date:
            issues.append(
                f"{path.name}:{line_number}: "
                "corrected printed date "
                f"{original_date} to "
                f"{corrected_date}"
            )

        try:
            sale_date = datetime.strptime(
                corrected_date,
                (
                    "%d-%b-%Y"
                    if len(corrected_date) == 11
                    else "%d-%b-%y"
                ),
            ).date()
        except ValueError:
            issues.append(
                f"{path.name}:{line_number}: "
                f"invalid date {original_date}"
            )
            continue

        if sale_date.year != year:
            issues.append(
                f"{path.name}:{line_number}: "
                f"unexpected year "
                f"{sale_date.year}"
            )
            continue

        high, medium, low, total = [
            int(value.replace(",", ""))
            for value in tokens[:4]
        ]

        # Components are rounded to
        # the nearest 1,000 kg in the PDF.
        difference = (
            high + medium + low - total
        )

        if abs(difference) > 2:
            issues.append(
                f"{path.name}:{line_number}: "
                "rounded component sum "
                "differs from reported total "
                f"by {difference} thousand kg"
            )

        rows.append({
            "sale_date":
                sale_date.isoformat(),
            "source_year": year,
            "high_1000kg": high,
            "medium_1000kg": medium,
            "low_1000kg": low,
            "total_1000kg": total,
            "source_file": path.name,
            "source_line": line_number,
            "printed_date": original_date,
        })

    return rows, issues


def benchmark(rows):
    """Walk forward using only earlier sale quantities."""
    result = {}

    columns = (
        "high_1000kg",
        "medium_1000kg",
        "low_1000kg",
        "total_1000kg",
    )

    for column in columns:
        errors = {
            method: [
                [] for _ in range(4)
            ]
            for method in (
                "last",
                "mean4",
            )
        }

        origins = 0

        for index in range(
            4,
            len(rows) - 3,
        ):
            if (
                rows[index]["source_year"]
                != 2026
            ):
                continue

            observed = [
                row[column]
                for row in rows[:index]
            ]

            guesses = {
                "last": observed[-1],
                "mean4": (
                    sum(observed[-4:]) / 4
                ),
            }

            for horizon in range(4):
                actual = (
                    rows[index + horizon]
                    [column]
                )

                for method, guess in (
                    guesses.items()
                ):
                    errors[method][
                        horizon
                    ].append(
                        abs(
                            actual - guess
                        )
                    )

            origins += 1

        result[column] = {
            "origins": origins,
            "mae_1000kg_by_horizon": {
                method: [
                    (
                        round(
                            sum(values)
                            / len(values),
                            2,
                        )
                        if values
                        else None
                    )
                    for values
                    in horizons
                ]
                for method, horizons
                in errors.items()
            },
        }

    return result


def main(folder):
    rows = []
    issues = []

    for year in SOURCE_YEARS:
        matches = sorted(
            folder.glob(
                "Weekly Tea Auction "
                "Quantities & Averages "
                f"{year}*.pdf"
            )
        )

        if len(matches) != 1:
            raise SystemExit(
                "Expected exactly one PDF "
                f"for {year}; found "
                f"{len(matches)}"
            )

        year_rows, year_issues = (
            extract_year(
                matches[0],
                year,
            )
        )

        rows.extend(year_rows)
        issues.extend(year_issues)

    rows.sort(
        key=lambda row:
            row["sale_date"]
    )

    dates = [
        row["sale_date"]
        for row in rows
    ]

    if len(dates) != len(
        set(dates)
    ):
        raise SystemExit(
            "Duplicate sale dates: "
            "review source PDFs"
        )

    output = (
        Path(__file__).parent
        / "outputs"
    )
    output.mkdir(exist_ok=True)

    with (
        output
        / "auction_sales_2024_2026.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "target_definition": (
            "Observed Colombo auction sale "
            "quantity by elevation; not "
            "factory orders or tea grades"
        ),
        "unit": (
            "thousand kilograms "
            "(as rounded in source PDFs)"
        ),
        "rows_by_year": {
            str(year): sum(
                row["source_year"]
                == year
                for row in rows
            )
            for year in SOURCE_YEARS
        },
        "first_sale": dates[0],
        "last_sale": dates[-1],
        "issues": issues,
        "baseline": benchmark(rows),
    }

    (
        output / "audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python "
            "research/auction_audit.py "
            "research/sources"
        )

    main(
        Path(sys.argv[1])
    )