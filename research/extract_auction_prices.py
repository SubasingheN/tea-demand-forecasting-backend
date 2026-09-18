"""Audit historical average auction prices by elevation from source PDFs.

Usage: python research/extract_auction_prices.py research/sources
Prices are in Sri Lankan rupees per kilogram and do not represent demand.
This script extracts and validates prices; it does not change the model.
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from auction_audit import DATE_CORRECTIONS, DATE_ROW, SOURCE_YEARS


ROOT = Path(__file__).resolve().parent / "outputs"
NUMERIC = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract(path, year):
    result = {}
    text = "\n".join(
        page.extract_text(extraction_mode="layout")
        for page in PdfReader(str(path)).pages
    )

    for line_number, line in enumerate(text.splitlines(), 1):
        match = DATE_ROW.match(line)
        if not match:
            continue

        printed_date, body = match.groups()
        corrected = DATE_CORRECTIONS.get(
            (year, printed_date), printed_date
        )

        try:
            parsed = datetime.strptime(
                corrected,
                "%d-%b-%Y" if len(corrected) == 11 else "%d-%b-%y",
            ).date()
        except ValueError as exc:
            raise ValueError(
                f"Invalid source date "
                f"{path.name}:{line_number}: {printed_date}"
            ) from exc

        if parsed.year != year:
            raise ValueError(
                f"Unexpected year at {path.name}:{line_number}"
            )

        # One 2024 row prints a price as "1161. 84".
        body = re.sub(
            r"\b(\d+)\.\s+(\d{2})\b",
            r"\1.\2",
            body,
        )
        values = NUMERIC.findall(body)

        # Four rounded sale quantities, then four average prices.
        if len(values) != 8:
            raise ValueError(
                f"Expected four quantities and four prices at "
                f"{path.name}:{line_number}; found {values}"
            )

        prices = [
            float(number.replace(",", ""))
            for number in values[4:]
        ]
        if any(
            not (100 <= price <= 5000)
            for price in prices
        ):
            raise ValueError(
                f"Implausible price at "
                f"{path.name}:{line_number}: {prices}"
            )

        key = parsed.isoformat()
        if key in result:
            raise ValueError(
                f"Duplicate date {key} in {path.name}"
            )

        result[key] = {
            "sale_date": key,
            "high_lkr_per_kg": prices[0],
            "medium_lkr_per_kg": prices[1],
            "low_lkr_per_kg": prices[2],
            "total_lkr_per_kg": prices[3],
            "source_file": path.name,
            "source_line": line_number,
            "printed_date": printed_date,
        }

    return result


def main(folder):
    with (
        ROOT / "auction_sales_2024_2026.csv"
    ).open(newline="", encoding="utf-8") as handle:
        quantity_rows = list(csv.DictReader(handle))

    expected = {
        row["sale_date"]
        for row in quantity_rows
    }
    if len(expected) != len(quantity_rows):
        raise ValueError(
            "Quantity dataset contains duplicate sale dates"
        )

    prices = {}
    for year in SOURCE_YEARS:
        matches = sorted(
            folder.glob(
                f"Weekly Tea Auction Quantities & Averages "
                f"{year}*.pdf"
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one PDF for {year}; "
                f"found {len(matches)}"
            )

        year_prices = extract(matches[0], year)
        overlap = prices.keys() & year_prices.keys()
        if overlap:
            raise ValueError(
                f"Duplicate dates across PDFs: "
                f"{sorted(overlap)}"
            )
        prices.update(year_prices)

    if set(prices) != expected:
        raise ValueError(
            "Price dates must match quantity dates exactly: "
            f"missing={sorted(expected - prices.keys())}, "
            f"extra={sorted(prices.keys() - expected)}"
        )

    output = ROOT / "auction_prices_2024_2026.csv"
    rows = [
        prices[key]
        for key in sorted(prices)
    ]

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(prices),
        "date_range": [min(prices), max(prices)],
        "unit": (
            "Sri Lankan rupees per kilogram, "
            "average auction sale price"
        ),
        "date_match_with_quantities": True,
        "note": (
            "Prices are observed after the auction; "
            "only prior dated prices may be forecast inputs."
        ),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python "
            "research/extract_auction_prices.py "
            "research/sources"
        )
    main(Path(sys.argv[1]))