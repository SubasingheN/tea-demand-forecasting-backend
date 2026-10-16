
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
# SAMPLE REPORT CONFIGURATION
# ============================================================

SAMPLES = {
    "2021": {
        "pattern": "forbes_2021_sale_36_*.pdf",
        "label_max_x": 195,
        "columns": {
            "High": {
                "BOP": (238, 281),
                "BOPF": (323, 366),
                "PEKOE/FBOP": (408, 451),
                "OP": (493, 540),
            },
            "Medium": {
                "BOP": (238, 281),
                "BOPF": (323, 366),
                "PEKOE/FBOP": (408, 451),
                "OP": (493, 540),
            },
        },
    },

    "2023": {
        "pattern": "forbes_2023_sale_50_*.pdf",
        "label_max_x": 135,
        "columns": {
            "High": {
                "BOP": (204, 252),
                "BOPF": (297, 346),
                "PEKOE/FBOP": (391, 439),
                "OP": (484, 535),
            },
            "Medium": {
                "BOP": (187, 235),
                "BOPF": (281, 329),
                "PEKOE/FBOP": (374, 422),
                "OP": (468, 520),
            },
        },
    },

    "2024": {
        "pattern": "forbes_2024_sale_50_*.pdf",
        "label_max_x": 135,
        "columns": {
            "High": {
                "BOP": (204, 252),
                "BOPF": (297, 346),
                "PEKOE/FBOP": (391, 439),
                "OP": (484, 535),
            },
            "Medium": {
                "BOP": (187, 235),
                "BOPF": (281, 329),
                "PEKOE/FBOP": (374, 422),
                "OP": (468, 520),
            },
        },
    },
}


# ============================================================
# TARGET QUALITY CATEGORIES
# ============================================================

CATEGORIES = {
    "bestwesterns": {
        "name": "Best Westerns",
        "elevation": "High",
    },
    "goodmediums": {
        "name": "Good Mediums",
        "elevation": "Medium",
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        value.lower(),
    )


# ============================================================
# PRICE PARSING
# ============================================================

def parse_price(words):
    if not words:
        raise ValueError("Empty price column")

    value = "".join(
        word["text"]
        for word in sorted(
            words,
            key=lambda item: item["x0"],
        )
    )

    value = value.replace("–", "-")
    value = value.replace("—", "-")
    value = re.sub(r"\s+", "", value)

    if value.upper() == "N/A":
        return None

    if re.fullmatch(r"\d{2,5}", value):
        number = int(value)

        if number > 0:
            return number, number

    if re.fullmatch(r"\d{2,5}-\d{2,5}", value):
        low, high = map(
            int,
            value.split("-"),
        )

        if 0 < low <= high:
            return low, high

    raise ValueError(
        f"Invalid quotation: {value!r}"
    )


# ============================================================
# SAMPLE EXTRACTION
# ============================================================

def extract_sample(year, config):
    matches = list(
        REPORT_DIR.glob(config["pattern"])
    )

    if len(matches) != 1:
        raise ValueError(
            f"{year}: Expected one PDF, "
            f"found {len(matches)}"
        )

    path = matches[0]

    results = []
    errors = []
    detected = set()

    print("\n" + "=" * 65)
    print("YEAR:", year)
    print("FILE:", path.name)
    print("=" * 65)

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
            )

            candidate_y = sorted({
                round(word["top"], 1)
                for word in words
                if word["x0"] < 135
                and (
                    "best" in word["text"].lower()
                    or "good" in word["text"].lower()
                    or "medium" in word["text"].lower()
                )
            })

            processed_y = []

            for y in candidate_y:
                if any(
                    abs(y - previous) < 3
                    for previous in processed_y
                ):
                    continue

                processed_y.append(y)

                row = sorted(
                    [
                        word
                        for word in words
                        if abs(word["top"] - y) < 3
                    ],
                    key=lambda word: word["x0"],
                )

                label_words = [
                    word["text"]
                    for word in row
                    if word["x0"]
                    < config["label_max_x"]
                ]

                label = normalize(
                    " ".join(label_words)
                )

                if label not in CATEGORIES:
                    continue

                category = CATEGORIES[label]

                if label in detected:
                    errors.append(
                        "Repeated category: "
                        f"{category['name']}"
                    )
                    continue

                detected.add(label)

                elevation = category["elevation"]
                columns = config["columns"][elevation]

                print(
                    f"\nCATEGORY: {category['name']}"
                )
                print("Elevation:", elevation)
                print("PDF page:", page_number)

                for grade, (xmin, xmax) in columns.items():
                    cell = [
                        word
                        for word in row
                        if xmin <= word["x0"] < xmax
                    ]

                    try:
                        price = parse_price(cell)

                        if price is None:
                            errors.append(
                                f"{category['name']}, "
                                f"{grade}: N/A"
                            )
                            print(f"{grade:12} N/A")
                            continue

                        low, high = price

                        results.append({
                            "year": year,
                            "grade_label_original": grade,
                            "quality_category":
                                category["name"],
                            "elevation": elevation,
                            "price_min_lkr": low,
                            "price_max_lkr": high,
                            "source_page": page_number,
                        })

                        print(
                            f"{grade:12} "
                            f"LKR {low:,} - {high:,}"
                        )

                    except ValueError as error:
                        errors.append(
                            f"{category['name']}, "
                            f"{grade}: {error}"
                        )

                        print(
                            f"{grade:12} ERROR: {error}"
                        )

    print("\nSUMMARY")
    print("Accepted quotations:", len(results))
    print("Detected categories:", len(detected))
    print("Errors:", len(errors))

    if errors:
        print("\nERROR DETAILS")

        for error in errors:
            print("-", error)

    return results, errors, detected


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "STEP 99 - CORRECTED FOUR-GRADE "
        "MULTICATEGORY EXTRACTION"
    )

    all_results = []
    all_errors = []
    total_categories = 0

    for year, config in SAMPLES.items():
        results, errors, detected = extract_sample(
            year,
            config,
        )

        all_results.extend(results)
        all_errors.extend(errors)
        total_categories += len(detected)

    expected = (
        len(SAMPLES)
        * len(CATEGORIES)
        * 4
    )

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    print(
        "Total accepted quotations:",
        len(all_results),
    )

    print("Expected quotations:", expected)
    print("Detected category sections:", total_categories)
    print("Total errors:", len(all_errors))

    if (
        len(all_results) == expected
        and total_categories == 6
        and not all_errors
    ):
        print(
            "\nSTATUS: PASS - "
            "All 24 sample quotations extracted."
        )
    else:
        print(
            "\nSTATUS: REVIEW REQUIRED"
        )

    print(
        "\nPEKOE/FBOP remains a combined "
        "quotation category."
    )

    print(
        "No CSV files or trained models were modified."
    )


if __name__ == "__main__":
    main()
