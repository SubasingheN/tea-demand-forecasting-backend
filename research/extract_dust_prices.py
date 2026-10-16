
from pathlib import Path
import csv
import re

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    ROOT / "data" / "external"
    / "forbes_auction_reports"
    / "WEEKLY TEA MARKET REPORT SALE NO 4-2025.pdf"
)

OUTPUT_PATH = (
    ROOT / "data" / "external"
    / "forbes_dust_prices.csv"
)

DUST_CATEGORIES = {
    "BetterPrimaryDust(Orthodox)":
        "Better Primary Dust (Orthodox)",
    "BetterPrimaryDust(CTC)P.Dust":
        "Better Primary Dust (CTC)",
    "BelowBestPrimaryDust(Orthodox)":
        "Below Best Primary Dust (Orthodox)",
    "OtherPrimaryDust(CTC)P.Dust":
        "Other Primary Dust (CTC)",
    "OtherPrimaryDust(Orthodox)":
        "Other Primary Dust (Orthodox)",
    "BetterSecondaryDust":
        "Better Secondary Dust",
    "OtherSecondaryDust":
        "Other Secondary Dust",
}

PRICE_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$"
)

ELEVATIONS = ["High", "Medium", "Low"]


def parse_price(value):
    if value.upper() == "N/A":
        return None

    match = PRICE_PATTERN.fullmatch(value)

    if not match:
        raise ValueError(
            f"Unrecognized quotation: {value}"
        )

    low = float(match.group(1))
    high = float(match.group(2))

    if low > high:
        raise ValueError(
            f"Invalid quotation range: {value}"
        )

    return low, high


def extract_dust_prices():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    records = []
    missing = []

    with pdfplumber.open(PDF_PATH) as pdf:
        # PDF page 7
        page = pdf.pages[6]

        text = page.extract_text(
            layout=True,
            x_tolerance=2,
            y_tolerance=3,
        ) or ""

    for line in text.splitlines():
        parts = line.strip().split()

        if not parts:
            continue

        category_key = parts[0]

        if category_key not in DUST_CATEGORIES:
            continue

        values = parts[1:]

        if len(values) != 6:
            raise ValueError(
                f"Unexpected price count: {line}"
            )

        for index, elevation in enumerate(
            ELEVATIONS
        ):
            current_price = values[index * 2 + 1]
            parsed = parse_price(current_price)

            if parsed is None:
                missing.append({
                    "category": DUST_CATEGORIES[
                        category_key
                    ],
                    "elevation": elevation,
                })
                continue

            low, high = parsed

            records.append({
                "auction_date": "2025-01-28",
                "auction_end_date": "2025-01-29",
                "grade": "Dust",
                "elevation": elevation,
                "quality_category": DUST_CATEGORIES[
                    category_key
                ],
                "price_min_lkr": low,
                "price_max_lkr": high,
                "price_type": "quoted_range",
                "source": "Forbes & Walker",
                "source_report": PDF_PATH.name,
                "source_page": 7,
            })

    expected_categories = set(DUST_CATEGORIES.values())
    extracted_categories = {
        row["quality_category"] for row in records
    } | {
        row["category"] for row in missing
    }

    if extracted_categories != expected_categories:
        raise ValueError(
            "One or more Dust categories were not extracted."
        )

    if len(records) != 19 or len(missing) != 2:
        raise ValueError(
            "Unexpected extraction counts. "
            f"Valid: {len(records)}, missing: {len(missing)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
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
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(records)

    print("Dust extraction completed.")
    print("Accepted records:", len(records))
    print("Unavailable quotations:", len(missing))
    print("Output:", OUTPUT_PATH)

    for item in missing:
        print("N/A:", item)


if __name__ == "__main__":
    extract_dust_prices()
