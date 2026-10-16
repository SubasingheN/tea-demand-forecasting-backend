
from pathlib import Path
import re

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT / "data" / "external"
    / "forbes_auction_reports" / "historical"
)

matches = list(
    REPORT_DIR.glob("forbes_2023_sale_50_*.pdf")
)

if len(matches) != 1:
    raise ValueError(
        f"Expected one sample report; found {len(matches)}"
    )

PDF_PATH = matches[0]

CATEGORIES = {
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
    r"N/A|\d{2,5}(?:\s*-\s*\d{2,5})?",
    re.IGNORECASE,
)


def parse_price(value):
    value = re.sub(
        r"\s+", "", value
    )

    if value.upper() == "N/A":
        return None

    if "-" in value:
        parts = value.split("-")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid price: {value}"
            )

        low, high = map(int, parts)

    else:
        low = high = int(value)

    if low <= 0 or low > high:
        raise ValueError(
            f"Invalid price range: {value}"
        )

    return low, high


def extract():
    extracted = []
    rejected = []

    with pdfplumber.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(
            pdf.pages, start=1
        ):
            text = page.extract_text(
                layout=True,
                x_tolerance=2,
                y_tolerance=3,
            ) or ""

            for line in text.splitlines():
                compact = re.sub(
                    r"\s+", "", line
                )

                category = next(
                    (
                        label
                        for raw, label
                        in CATEGORIES.items()
                        if compact.startswith(raw)
                    ),
                    None
                )

                if not category:
                    continue

                raw_name = next(
                    raw for raw, label
                    in CATEGORIES.items()
                    if label == category
                )

                values_text = compact[
                    len(raw_name):
                ]

                values = PRICE_PATTERN.findall(
                    values_text
                )

                if len(values) != 6:
                    rejected.append({
                        "page": page_number,
                        "category": category,
                        "values": values,
                    })
                    continue

                for elevation, position in [
                    ("High", 1),
                    ("Medium", 3),
                    ("Low", 5),
                ]:
                    value = values[position]

                    try:
                        price = parse_price(value)
                    except ValueError:
                        price = None

                    if price is None:
                        rejected.append({
                            "page": page_number,
                            "category": category,
                            "elevation": elevation,
                            "value": value,
                        })
                        continue

                    extracted.append({
                        "auction_date": "2023-12-18",
                        "grade": "Dust",
                        "elevation": elevation,
                        "quality_category": category,
                        "price_min_lkr": price[0],
                        "price_max_lkr": price[1],
                        "source_page": page_number,
                    })

    print("STEP 72 — HISTORICAL DUST EXTRACTION TEST")
    print("=" * 60)
    print("PDF:", PDF_PATH.name)
    print("Accepted quotations:", len(extracted))
    print("Rejected entries:", len(rejected))

    for record in extracted:
        print(
            record["elevation"],
            "|",
            record["quality_category"],
            "| LKR",
            record["price_min_lkr"],
            "-",
            record["price_max_lkr"],
        )

    if rejected:
        print("\nREJECTED ENTRIES")

        for item in rejected:
            print(item)


if __name__ == "__main__":
    extract()
