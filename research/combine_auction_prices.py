
from pathlib import Path
from collections import Counter
import csv


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / "data" / "external"

INPUT_FILES = [
    EXTERNAL_DIR / "forbes_auction_prices.csv",
    EXTERNAL_DIR / "forbes_dust_prices.csv",
]

OUTPUT_FILE = (
    EXTERNAL_DIR / "forbes_auction_prices_combined.csv"
)

EXPECTED_GRADES = {"BOP", "BOPF", "Dust", "OP", "Pekoe"}

FIELDS = [
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


def combine_prices():
    records = []

    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != FIELDS:
                raise ValueError(
                    f"Unexpected columns in {path.name}"
                )

            records.extend(reader)

    if len(records) != 35:
        raise ValueError(
            f"Expected 35 records, found {len(records)}"
        )

    grades = {r["grade"] for r in records}

    if grades != EXPECTED_GRADES:
        raise ValueError(
            f"Unexpected grade coverage: {grades}"
        )

    keys = set()

    for record in records:
        key = (
            record["auction_date"],
            record["grade"],
            record["elevation"],
            record["quality_category"],
        )

        if key in keys:
            raise ValueError(
                f"Duplicate quotation: {key}"
            )

        keys.add(key)

        low = float(record["price_min_lkr"])
        high = float(record["price_max_lkr"])

        if low <= 0 or low > high:
            raise ValueError(
                f"Invalid quotation: {record}"
            )

        if record["price_type"] != "quoted_range":
            raise ValueError(
                "Unexpected price type"
            )

        if not all(
            str(record[field]).strip()
            for field in FIELDS
        ):
            raise ValueError(
                f"Missing source or quotation data: {record}"
            )

    records.sort(
        key=lambda r: (
            r["auction_date"],
            r["grade"],
            r["elevation"],
            r["quality_category"],
        )
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(records)

    print("Combination completed successfully.")
    print("Total records:", len(records))
    print(
        "Records by grade:",
        dict(Counter(r["grade"] for r in records))
    )
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    combine_prices()
