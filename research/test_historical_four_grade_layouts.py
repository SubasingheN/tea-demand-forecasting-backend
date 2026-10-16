
from pathlib import Path
import re

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "data"
    / "external"
    / "forbes_auction_reports"
    / "historical"
)

GRADES = (
    "BOP",
    "BOPF",
    "PEKOE/FBOP",
    "OP",
)

CATEGORIES = {
    "bestwesterns": "High",
    "goodmediums": "Medium",
}


# ============================================================
# SAMPLE REPORTS AND COLUMN POSITIONS
# ============================================================

SAMPLES = {
    "2021_early": {
        "pattern": "forbes_2021_sale_23_*.pdf",
        "High": [245, 327, 408, 489],
        "Medium": [245, 327, 408, 489],
        "cell_width": 39,
    },
    "2021_late": {
        "pattern": "forbes_2021_sale_36_*.pdf",
        "High": [242, 327, 412, 497],
        "Medium": [242, 327, 412, 497],
        "cell_width": 40,
    },
    "2023_early": {
        "pattern": "forbes_2023_date_2023-01-03_*.pdf",
        "High": [247, 321, 394, 468],
        "Medium": [248, 321, 394, 468],
        "cell_width": 35,
    },
    "2023_middle": {
        "pattern": "forbes_2023_date_2023-05-09_*.pdf",
        "High": [239, 322, 405, 487],
        "Medium": [241, 322, 404, 487],
        "cell_width": 38,
    },
    "2023_late": {
        "pattern": "forbes_2023_sale_50_*.pdf",
        "High": [208, 302, 395, 489],
        "Medium": [191, 285, 378, 472],
        "cell_width": 43,
    },
    "2024": {
        "pattern": "forbes_2024_sale_50_*.pdf",
        "High": [208, 302, 395, 489],
        "Medium": [191, 285, 378, 472],
        "cell_width": 43,
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def detect_category(row):
    for boundary in (130, 140, 160, 180, 195):
        label = normalize(
            " ".join(
                word["text"]
                for word in row
                if word["x0"] < boundary
            )
        )

        if label in CATEGORIES:
            return label

    return None


def parse_price_cell(words):
    if not words:
        return None, "MISSING"

    raw = "".join(
        word["text"]
        for word in sorted(
            words,
            key=lambda word: word["x0"],
        )
    )

    raw = (
        raw.replace("–", "-")
        .replace("—", "-")
        .replace(",", "")
        .replace(" ", "")
        .strip()
    )

    if raw.upper() in {"N/A", "NA", "-", ""}:
        return None, "MISSING"

    match = re.fullmatch(
        r"(\d{2,5})-(\d{2,5})",
        raw,
    )

    if match:
        low = int(match.group(1))
        high = int(match.group(2))

        if 0 < low <= high:
            return (low, high), "OK"

    if re.fullmatch(r"\d{2,5}", raw):
        price = int(raw)

        if price > 0:
            return (price, price), "SINGLE_VALUE"

    return None, f"REVIEW: {raw!r}"


# ============================================================
# PRICE EXTRACTION
# ============================================================

def extract_row_prices(row, positions, cell_width):
    results = {}

    for grade, center in zip(
        GRADES,
        positions,
    ):
        xmin = center - 4
        xmax = center + cell_width

        cell = [
            word
            for word in row
            if xmin <= word["x0"] < xmax
        ]

        price, status = parse_price_cell(cell)

        results[grade] = {
            "price": price,
            "status": status,
        }

    return results


# ============================================================
# REPORT TEST
# ============================================================

def test_report(layout_name, config):
    matches = list(
        REPORT_DIR.glob(config["pattern"])
    )

    print("\n" + "=" * 70)
    print("LAYOUT:", layout_name)

    if len(matches) != 1:
        print(
            "ERROR: Expected one PDF, found",
            len(matches),
        )
        return 0, 0, 1

    path = matches[0]
    print("FILE:", path.name)

    found = set()
    accepted = 0
    missing = 0
    reviews = 0

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
                if word["x0"] < 160
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

                category_key = detect_category(row)

                if category_key not in CATEGORIES:
                    continue

                if category_key in found:
                    continue

                found.add(category_key)

                elevation = CATEGORIES[category_key]

                prices = extract_row_prices(
                    row,
                    config[elevation],
                    config["cell_width"],
                )

                print(
                    "\nCATEGORY:",
                    category_key,
                )
                print("ELEVATION:", elevation)
                print("PDF PAGE:", page_number)

                for grade, details in prices.items():
                    price = details["price"]
                    status = details["status"]

                    if status in {
                        "OK",
                        "SINGLE_VALUE",
                    }:
                        low, high = price

                        print(
                            f"{grade:12} "
                            f"LKR {low:,} - {high:,} "
                            f"[{status}]"
                        )

                        accepted += 1

                    elif status == "MISSING":
                        print(
                            f"{grade:12} MISSING"
                        )
                        missing += 1

                    else:
                        print(
                            f"{grade:12} {status}"
                        )
                        reviews += 1

            if len(found) == len(CATEGORIES):
                break

    for category in sorted(
        set(CATEGORIES) - found
    ):
        print(
            "CATEGORY NOT FOUND:",
            category,
        )
        reviews += 4

    print("\nREPORT SUMMARY")
    print("Accepted:", accepted)
    print("Missing:", missing)
    print("Review required:", reviews)

    return accepted, missing, reviews


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "STEP 105 - CORRECTED "
        "HISTORICAL PRICE EXTRACTION TEST"
    )

    total_accepted = 0
    total_missing = 0
    total_reviews = 0

    for layout_name, config in SAMPLES.items():
        accepted, missing, reviews = test_report(
            layout_name,
            config,
        )

        total_accepted += accepted
        total_missing += missing
        total_reviews += reviews

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "Maximum quotation cells:",
        len(SAMPLES) * len(CATEGORIES) * len(GRADES),
    )
    print(
        "Accepted:",
        total_accepted,
    )
    print(
        "Missing:",
        total_missing,
    )
    print(
        "Review required:",
        total_reviews,
    )

    if total_reviews == 0:
        print(
            "\nSTATUS: No parsing errors "
            "in the six sample layouts."
        )
    else:
        print(
            "\nSTATUS: Additional review required."
        )

    print(
        "\nPEKOE/FBOP remains a combined "
        "quotation category."
    )

    print(
        "Factory dataset and TFT model: unchanged."
    )


if __name__ == "__main__":
    main()
