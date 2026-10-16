
from pathlib import Path
import re

import pdfplumber


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "data"
    / "external"
    / "forbes_auction_reports"
    / "historical"
)


# ============================================================
# YEAR-SPECIFIC PDF CONFIGURATION
# ============================================================

SAMPLES = {
    "2021": {
        "pattern": "forbes_2021_sale_36_*.pdf",
        "label_max_x": 225,
        "columns": {
            "High": (283, 335),
            "Medium": (388, 440),
            "Low": (493, 550),
        },
    },
    "2024": {
        "pattern": "forbes_2024_sale_50_*.pdf",
        "label_max_x": 180,
        "columns": {
            "High": (231, 286),
            "Medium": (337, 390),
            "Low": (445, 510),
        },
    },
}


# ============================================================
# DUST QUALITY CATEGORIES
# ============================================================

CATEGORIES = [
    "Better Primary Dust (Orthodox)",
    "Better Primary Dust (CTC) P. Dust",
    "Below Best Primary Dust (Orthodox)",
    "Other Primary Dust (CTC) P. Dust",
    "Other Primary Dust (Orthodox)",
    "Better Secondary Dust",
    "Other Secondary Dust",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(value):
    """
    Normalize category names for reliable matching.
    """

    return re.sub(
        r"\s+",
        "",
        value,
    ).lower()


# ============================================================
# PRICE EXTRACTION
# ============================================================

def parse_price(words):
    """
    Convert PDF words into a validated quotation range.

    Examples:
        ['1100', '-', '1380'] -> (1100, 1380)
        ['900-1100']         -> (900, 1100)
        ['980']              -> (980, 980)
        ['N/A']              -> None
    """

    ordered_words = sorted(
        words,
        key=lambda item: item["x0"],
    )

    value = "".join(
        word["text"]
        for word in ordered_words
    )

    value = value.replace("–", "-")
    value = value.replace("—", "-")
    value = re.sub(r"\s+", "", value)

    if value.upper() == "N/A":
        return None

    # Single quotation value
    if re.fullmatch(r"\d{2,5}", value):
        number = int(value)

        if number <= 0:
            raise ValueError(
                f"Invalid price: {value}"
            )

        return number, number

    # Quotation range
    if re.fullmatch(
        r"\d{2,5}-\d{2,5}",
        value,
    ):
        low, high = map(
            int,
            value.split("-"),
        )

        if 0 < low <= high:
            return low, high

    raise ValueError(
        f"Ambiguous or invalid price: {value!r}"
    )


# ============================================================
# EXTRACT DUST QUOTATIONS FROM ONE PDF
# ============================================================

def inspect_report(year, config):
    """
    Extract current-sale Dust quotations from one
    historical Forbes & Walker auction report.

    Uses year-specific:
        - Category-label boundaries
        - Current-sale price column boundaries
    """

    paths = list(
        REPORT_DIR.glob(
            config["pattern"]
        )
    )

    if len(paths) != 1:
        print(
            f"\n{year}: Expected one PDF, "
            f"found {len(paths)}"
        )
        return

    pdf_path = paths[0]

    results = []
    rejected = []
    found_categories = set()

    label_max_x = config["label_max_x"]

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):

            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
            )

            # Identify possible Dust category rows.
            candidate_y = sorted({
                round(word["top"], 1)
                for word in words
                if word["x0"] < 180
                and word["text"].lower()
                in {"better", "below", "other"}
            })

            processed_y = []

            for y in candidate_y:

                # Avoid processing the same row twice.
                if any(
                    abs(y - previous_y) < 5
                    for previous_y in processed_y
                ):
                    continue

                processed_y.append(y)

                # Collect all words on the same row.
                row = [
                    word
                    for word in words
                    if abs(word["top"] - y) < 3
                ]

                row = sorted(
                    row,
                    key=lambda word: word["x0"],
                )

                # Rebuild the category name without
                # including the price columns.
                label_words = [
                    word["text"]
                    for word in row
                    if word["x0"] < label_max_x
                ]

                label = normalize(
                    " ".join(label_words)
                )

                # Match one of the seven Dust categories.
                category = next(
                    (
                        candidate
                        for candidate in CATEGORIES
                        if normalize(candidate) == label
                    ),
                    None,
                )

                if category is None:
                    continue

                # Prevent duplicate category extraction.
                if category in found_categories:
                    rejected.append(
                        (
                            page_number,
                            category,
                            "Repeated category",
                        )
                    )
                    continue

                found_categories.add(category)

                # Read only current-sale columns.
                for elevation, bounds in (
                    config["columns"].items()
                ):

                    x_min, x_max = bounds

                    cell_words = [
                        word
                        for word in row
                        if x_min <= word["x0"] < x_max
                    ]

                    try:
                        price = parse_price(
                            cell_words
                        )

                        if price is None:
                            rejected.append(
                                (
                                    page_number,
                                    category,
                                    elevation,
                                    "N/A",
                                )
                            )
                            continue

                        low, high = price

                        results.append(
                            {
                                "year": year,
                                "grade": "Dust",
                                "category": category,
                                "elevation": elevation,
                                "price_min_lkr": low,
                                "price_max_lkr": high,
                                "source_page": page_number,
                            }
                        )

                    except ValueError as error:
                        rejected.append(
                            (
                                page_number,
                                category,
                                elevation,
                                str(error),
                            )
                        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("YEAR:", year)
    print("FILE:", pdf_path.name)
    print("=" * 70)

    for item in results:

        print(
            f"{item['elevation']:7} | "
            f"{item['category']:38} | "
            f"LKR {item['price_min_lkr']:,} - "
            f"{item['price_max_lkr']:,} "
            f"| page {item['source_page']}"
        )

    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    print("\nSUMMARY")
    print("Accepted quotations:", len(results))
    print("Rejected entries:", len(rejected))
    print(
        "Categories detected:",
        len(found_categories),
    )

    missing_categories = [
        category
        for category in CATEGORIES
        if category not in found_categories
    ]

    print(
        "Missing categories:",
        len(missing_categories),
    )

    if missing_categories:
        print("\nMISSING CATEGORIES")

        for category in missing_categories:
            print("-", category)

    if rejected:
        print("\nREJECTED ENTRIES")

        for item in rejected:
            print(item)

    if (
        len(results) == 21
        and len(rejected) == 0
        and len(found_categories) == 7
    ):
        print(
            "\nSTATUS: PASS - "
            "All 21 sample quotations extracted."
        )
    else:
        print(
            "\nSTATUS: REVIEW REQUIRED - "
            "Extraction is incomplete or ambiguous."
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "STEP 80 - MULTIYEAR DUST EXTRACTION TEST"
    )

    all_results = []

    for year, config in SAMPLES.items():

        results = inspect_report(
            year,
            config,
        )

        if results:
            all_results.extend(results)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "Total accepted quotations:",
        len(all_results),
    )

    print(
        "Years tested:",
        ", ".join(SAMPLES.keys()),
    )

    print(
        "Expected quotations:",
        len(SAMPLES) * 21,
    )

    print(
        "\nNote: This is a sample extraction test."
    )
    print(
        "No CSV files or trained models were modified."
    )


if __name__ == "__main__":
    main()
