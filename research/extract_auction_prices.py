
from pathlib import Path
import csv
import re

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    ROOT
    / "data"
    / "external"
    / "forbes_auction_reports"
    / "WEEKLY TEA MARKET REPORT SALE NO 4-2025.pdf"
)

OUTPUT_DIR = ROOT / "data" / "external"

OUTPUT_PATH = (
    OUTPUT_DIR / "forbes_auction_prices.csv"
)

GRADES = {"BOP", "BOPF", "OP", "PEKOE"}

CATEGORIES = [
    "Select Best",
    "Best",
    "Below Best",
    "Others",
]

PRICE_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$"
)


def parse_price(value):
    value = value.strip()

    match = PRICE_PATTERN.fullmatch(value)

    if not match:
        return None

    low = float(match.group(1))
    high = float(match.group(2))

    if low > high:
        return None

    return low, high


def extract_prices():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    records = []
    rejected = []

    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[7]

        text = page.extract_text(
            layout=True
        ) or ""

        for line in text.splitlines():
            parts = line.strip().split()

            if not parts:
                continue

            grade = parts[0].upper()

            if grade not in GRADES:
                continue

            values = parts[1:]

            if len(values) != 8:
                rejected.append({
                    "grade": grade,
                    "reason": "Unexpected number of price values",
                    "line": line.strip(),
                })
                continue

            for index, category in enumerate(
                CATEGORIES
            ):
                # Current sale is the second value
                # in each pair of previous/current prices.
                price_text = values[index * 2 + 1]

                parsed = parse_price(price_text)

                if parsed is None:
                    rejected.append({
                        "grade": grade,
                        "category": category,
                        "price_text": price_text,
                        "reason": "Invalid quotation",
                    })
                    continue

                low, high = parsed

                records.append({
                    "auction_date": "2025-01-28",
                    "auction_end_date": "2025-01-29",
                    "grade": (
                        "Pekoe"
                        if grade == "PEKOE"
                        else grade
                    ),
                    "elevation": "Low",
                    "quality_category": category,
                    "price_min_lkr": low,
                    "price_max_lkr": high,
                    "price_type": "quoted_range",
                    "source": "Forbes & Walker",
                    "source_report": PDF_PATH.name,
                    "source_page": 8,
                })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "auction_date",
        "auction_end_date",
        "grade",
        "elevation",
        "quality_category",
        "price_min_lkr",
        "price_max_lkr",
        "price_type",
        "source",
        "source_report",
        "source_page",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(records)

    print("Extraction completed.")
    print("Accepted records:", len(records))
    print("Rejected records:", len(rejected))
    print("Output:", OUTPUT_PATH)

    for item in rejected:
        print("REJECTED:", item)


if __name__ == "__main__":
    extract_prices()
