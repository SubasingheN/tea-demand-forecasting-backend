
from pathlib import Path
from collections import Counter
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
# QUALITY CATEGORIES
# ============================================================

CATEGORIES = {
    "bestwesterns": "Best Westerns",
    "belowbestwesterns": "Below Best Westerns",
    "plainerwesterns": "Plainer Westerns",
    "nuwaraeliyas": "Nuwara Eliyas",
    "brighterudapussellawas": "Brighter Udapussellawas",
    "otherudapussellawas": "Other Udapussellawas",
    "bestuvas": "Best Uvas",
    "otheruvas": "Other Uvas",
    "goodmediums": "Good Mediums",
    "othermediums": "Other Mediums",
}

# The category-label boundary must include the longest
# quality-category names while excluding auction prices.
#
# Different reports have slightly different page layouts.
# We use multiple boundaries and require an exact category
# match rather than accepting arbitrary text.

LABEL_BOUNDARIES = [140, 160, 180, 195]


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize(text):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower(),
    )


def identify_year(filename):
    match = re.search(
        r"forbes_(20\d{2})_",
        filename,
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(row):
    """
    Try several label boundaries.

    Return a category only when the reconstructed label
    matches one of the known categories exactly.
    """

    ordered = sorted(
        row,
        key=lambda word: word["x0"],
    )

    for boundary in LABEL_BOUNDARIES:
        label_words = [
            word["text"]
            for word in ordered
            if word["x0"] < boundary
        ]

        key = normalize(
            " ".join(label_words)
        )

        if key in CATEGORIES:
            return key

    return None


# ============================================================
# INSPECT ONE REPORT
# ============================================================

def inspect_report(path):
    """
    Detect high- and medium-grown quotation categories.

    This step checks category presence only.
    It does not extract or validate price values.
    """

    found = set()
    matched_pages = {}

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
                if word["x0"] < 180
            })

            processed = []

            for y in candidate_y:
                if any(
                    abs(y - previous) < 3
                    for previous in processed
                ):
                    continue

                processed.append(y)

                row = [
                    word
                    for word in words
                    if abs(word["top"] - y) < 3
                ]

                category = detect_category(row)

                if category is None:
                    continue

                found.add(category)

                if category not in matched_pages:
                    matched_pages[category] = page_number

    return found, matched_pages


# ============================================================
# HISTORICAL COVERAGE AUDIT
# ============================================================

def main():
    reports = sorted(
        REPORT_DIR.glob("*.pdf")
    )

    stats = {
        "2021": Counter(),
        "2023": Counter(),
        "2024": Counter(),
    }

    reviews = []

    print(
        "STEP 102 - CORRECTED FOUR-GRADE "
        "HISTORICAL COVERAGE AUDIT"
    )
    print("=" * 65)
    print("PDF reports:", len(reports))

    for index, path in enumerate(
        reports,
        start=1,
    ):
        year = identify_year(path.name)

        if year not in stats:
            reviews.append(
                (
                    path.name,
                    "Unknown year",
                )
            )
            continue

        stats[year]["reports"] += 1

        try:
            found, pages = inspect_report(path)

            stats[year][
                "categories_found"
            ] += len(found)

            if len(found) == len(CATEGORIES):
                stats[year]["complete"] += 1

            else:
                stats[year]["review"] += 1

                missing = [
                    CATEGORIES[key]
                    for key in CATEGORIES
                    if key not in found
                ]

                reviews.append(
                    (
                        path.name,
                        f"{len(found)}/10 categories; "
                        f"missing={','.join(missing)}",
                    )
                )

        except Exception as error:
            stats[year]["review"] += 1

            reviews.append(
                (
                    path.name,
                    f"Processing error: {error}",
                )
            )

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(reports)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\nCOVERAGE BY YEAR")
    print("=" * 65)

    for year in sorted(stats):
        values = stats[year]

        print(f"\nYEAR: {year}")
        print(
            "Reports:",
            values["reports"],
        )
        print(
            "All categories found:",
            values["complete"],
        )
        print(
            "Needs review:",
            values["review"],
        )
        print(
            "Category rows detected:",
            values["categories_found"],
        )

    print("\nOVERALL SUMMARY")
    print("=" * 65)

    print(
        "Reports with all 10 categories:",
        sum(
            value["complete"]
            for value in stats.values()
        ),
    )

    print(
        "Reports requiring review:",
        len(reviews),
    )

    print("\nREPORTS REQUIRING REVIEW")
    print("=" * 65)

    if not reviews:
        print("None")

    for filename, reason in reviews[:30]:
        print(
            filename,
            "|",
            reason,
        )

    print(
        "\nIMPORTANT: This checks category "
        "presence, not numerical price accuracy."
    )

    print(
        "No dataset or model files were modified."
    )


if __name__ == "__main__":
    main()
