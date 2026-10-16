
from pathlib import Path
import re

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT / "data" / "external"
    / "forbes_auction_reports"
    / "historical"
)

matches = list(
    REPORT_DIR.glob("forbes_2023_sale_50_*.pdf")
)

if len(matches) != 1:
    raise ValueError("Sample PDF not found uniquely")

PDF_PATH = matches[0]

CATEGORIES = [
    "Better Primary Dust (Orthodox)",
    "Better Primary Dust (CTC) P. Dust",
    "Below Best Primary Dust (Orthodox)",
    "Other Primary Dust (CTC) P. Dust",
    "Other Primary Dust (Orthodox)",
    "Better Secondary Dust",
    "Other Secondary Dust",
]

# Current-sale price column boundaries for this PDF.
CURRENT_COLUMNS = {
    "High": (231, 286),
    "Medium": (340, 391),
    "Low": (445, 510),
}


def normalize(text):
    return re.sub(
        r"\s+", "",
        text
    ).lower()


def parse_price(words):
    value = "".join(
        word["text"]
        for word in sorted(
            words, key=lambda w: w["x0"]
        )
    )

    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", "", value)

    if value.upper() == "N/A":
        return None

    if re.fullmatch(r"\d{2,5}", value):
        number = int(value)
        return number, number

    if re.fullmatch(
        r"\d{2,5}-\d{2,5}",
        value
    ):
        low, high = map(
            int, value.split("-")
        )

        if 0 < low <= high:
            return low, high

    raise ValueError(
        f"Ambiguous quotation: {value}"
    )


def main():
    results = []
    rejected = []

    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[8]  # PDF page 9

        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3
        )

        # Discover category rows using reconstructed labels.
        candidate_y = sorted({
            round(word["top"], 1)
            for word in words
            if 595 <= word["top"] <= 710
            and word["x0"] < 160
        })

        processed_y = []

        for y in candidate_y:
            if any(
                abs(y - previous) < 5
                for previous in processed_y
            ):
                continue

            processed_y.append(y)

            row_words = [
                word for word in words
                if abs(word["top"] - y) < 3
            ]

            label_words = [
                word for word in row_words
                if word["x0"] < 165
            ]

            label = normalize(
                " ".join(
                    word["text"]
                    for word in sorted(
                        label_words,
                        key=lambda w: w["x0"]
                    )
                )
            )

            category = next(
                (
                    item for item in CATEGORIES
                    if normalize(item) == label
                ),
                None
            )

            if not category:
                continue

            for elevation, bounds in (
                CURRENT_COLUMNS.items()
            ):
                x_min, x_max = bounds

                cell_words = [
                    word for word in row_words
                    if x_min <= word["x0"] < x_max
                ]

                try:
                    price = parse_price(cell_words)

                    if price is None:
                        rejected.append(
                            (category, elevation, "N/A")
                        )
                        continue

                    results.append({
                        "category": category,
                        "elevation": elevation,
                        "min": price[0],
                        "max": price[1],
                    })

                except ValueError as error:
                    rejected.append(
                        (category, elevation, str(error))
                    )

    print("STEP 75 — COORDINATE DUST EXTRACTION")
    print("=" * 65)

    for item in results:
        print(
            f"{item['elevation']:7} | "
            f"{item['category']:38} | "
            f"LKR {item['min']:,} - {item['max']:,}"
        )

    print("\nSUMMARY")
    print("Accepted quotations:", len(results))
    print("Rejected quotations:", len(rejected))
    print("Unique categories:", len({
        item["category"] for item in results
    }))

    if rejected:
        print("\nREJECTED ENTRIES")

        for item in rejected:
            print(item)


if __name__ == "__main__":
    main()
