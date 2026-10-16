
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
# DUST CATEGORIES
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

ELEVATIONS = ["High", "Medium", "Low"]

EXPECTED_QUOTATIONS = 21


# ============================================================
# HISTORICAL LAYOUTS
# ============================================================

LAYOUTS = {
    "2021": [
        {
            "name": "2021_cell_grouping",
            "method": "cell_grouping",
            "label_max_x": 225,
            "price_start_x": 225,
        },
        {
            "name": "2021_standard",
            "method": "coordinates",
            "label_max_x": 225,
            "columns": {
                "High": (283, 335),
                "Medium": (388, 440),
                "Low": (493, 550),
            },
        },
        {
            "name": "2021_shifted",
            "method": "coordinates",
            "label_max_x": 225,
            "columns": {
                "High": (275, 330),
                "Medium": (375, 430),
                "Low": (475, 530),
            },
        },
    ],

    "2023": [
        {
            "name": "2023_early",
            "method": "coordinates",
            "label_max_x": 225,
            "columns": {
                "High": (275, 325),
                "Medium": (375, 425),
                "Low": (470, 525),
            },
        },
        {
            "name": "2023_late",
            "method": "coordinates",
            "label_max_x": 180,
            "columns": {
                "High": (231, 286),
                "Medium": (340, 391),
                "Low": (445, 510),
            },
        },
    ],

    "2024": [
        {
            "name": "2024_standard",
            "method": "coordinates",
            "label_max_x": 180,
            "columns": {
                "High": (231, 286),
                "Medium": (337, 390),
                "Low": (445, 510),
            },
        },
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(value):
    return re.sub(
        r"\s+",
        "",
        str(value),
    ).lower()


def identify_year(filename):
    match = re.search(
        r"forbes_(20\d{2})_",
        filename,
    )

    return match.group(1) if match else None


def parse_price(words):
    if not words:
        raise ValueError("Empty price cell")

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
        f"Invalid quotation: {value!r}"
    )


# ============================================================
# CATEGORY DETECTION
# ============================================================

def find_category_rows(words):
    candidates = sorted({
        round(word["top"], 1)
        for word in words
        if word["x0"] < 200
        and word["text"].lower()
        in {"better", "below", "other"}
    })

    filtered = []

    for y in candidates:
        if any(
            abs(y - old_y) < 4
            for old_y in filtered
        ):
            continue

        filtered.append(y)

    return filtered


def reconstruct_category(
    words,
    row_y,
    label_max_x,
):
    """
    Supports single-line and multiline Dust labels.
    """

    label_words = sorted(
        [
            word
            for word in words
            if word["x0"] < label_max_x
            and -1.5 <= word["top"] - row_y <= 10.5
        ],
        key=lambda word: (
            round(word["top"], 1),
            word["x0"],
        ),
    )

    label = normalize(
        " ".join(
            word["text"]
            for word in label_words
        )
    )

    return next(
        (
            category
            for category in CATEGORIES
            if normalize(category) == label
        ),
        None,
    )


# ============================================================
# PRICE-ROW DETECTION
# ============================================================

def find_price_y(
    words,
    row_y,
    price_start_x,
):
    candidates = Counter(
        round(word["top"], 1)
        for word in words
        if (
            price_start_x <= word["x0"] < 560
            and -2 <= word["top"] - row_y <= 7
            and (
                re.search(r"\d", word["text"])
                or word["text"].upper() == "N/A"
            )
        )
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda y: (
            candidates[y],
            -abs(y - row_y),
        ),
    )


# ============================================================
# PRICE-CELL GROUPING
# ============================================================

def group_price_words(words):
    if not words:
        return []

    words = sorted(
        words,
        key=lambda item: item["x0"],
    )

    groups = []
    current = [words[0]]

    for previous, word in zip(
        words,
        words[1:],
    ):
        gap = word["x0"] - previous["x1"]

        if gap > 12:
            groups.append(current)
            current = [word]
        else:
            current.append(word)

    groups.append(current)

    return groups


# ============================================================
# EXTRACT ONE LAYOUT
# ============================================================

def extract_with_layout(path, layout):
    results = []
    errors = []
    found_categories = set()

    price_start = (
        layout.get("price_start_x")
        if layout["method"] == "cell_grouping"
        else min(
            bounds[0]
            for bounds in layout["columns"].values()
        )
    )

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):
            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
            )

            for y in find_category_rows(words):
                category = reconstruct_category(
                    words,
                    y,
                    layout["label_max_x"],
                )

                if category is None:
                    continue

                if category in found_categories:
                    errors.append(
                        f"Repeated category: {category}"
                    )
                    continue

                found_categories.add(category)

                price_y = find_price_y(
                    words,
                    y,
                    price_start,
                )

                if price_y is None:
                    errors.append(
                        f"No price row: {category}"
                    )
                    continue

                price_words = [
                    word
                    for word in words
                    if abs(word["top"] - price_y) < 2
                ]

                # ------------------------------------
                # METHOD A: PRICE-CELL GROUPING
                # ------------------------------------

                if layout["method"] == "cell_grouping":
                    candidate_words = [
                        word
                        for word in price_words
                        if (
                            layout["price_start_x"]
                            <= word["x0"] < 560
                        )
                    ]

                    groups = group_price_words(
                        candidate_words
                    )

                    if len(groups) != 6:
                        errors.append(
                            f"{category}: Expected 6 "
                            f"price cells; found {len(groups)}"
                        )
                        continue

                    current_cells = {
                        "High": groups[1],
                        "Medium": groups[3],
                        "Low": groups[5],
                    }

                # ------------------------------------
                # METHOD B: COORDINATE COLUMNS
                # ------------------------------------

                else:
                    current_cells = {}

                    for elevation, bounds in (
                        layout["columns"].items()
                    ):
                        x_min, x_max = bounds

                        current_cells[elevation] = [
                            word
                            for word in price_words
                            if (
                                x_min <= word["x0"] < x_max
                            )
                        ]

                # ------------------------------------
                # PARSE CURRENT-SALE PRICES
                # ------------------------------------

                for elevation in ELEVATIONS:
                    cell = current_cells[elevation]

                    try:
                        price = parse_price(cell)

                        if price is None:
                            errors.append(
                                f"N/A: {category}, {elevation}"
                            )
                            continue

                        results.append({
                            "grade": "Dust",
                            "category": category,
                            "elevation": elevation,
                            "min_lkr": price[0],
                            "max_lkr": price[1],
                            "source_page": page_number,
                            "layout": layout["name"],
                        })

                    except ValueError as error:
                        errors.append(
                            f"{category}, {elevation}: "
                            f"{error}"
                        )

    missing_categories = [
        category
        for category in CATEGORIES
        if category not in found_categories
    ]

    return {
        "results": results,
        "errors": errors,
        "missing_categories": missing_categories,
        "detected_categories": len(found_categories),
        "layout": layout["name"],
    }


# ============================================================
# SELECT THE BEST EXTRACTION
# ============================================================

def extract_report(path, layouts):
    """
    Test all applicable layouts.

    A complete extraction is not automatically considered
    independently source-verified.
    """

    if isinstance(layouts, dict):
        layouts = [layouts]

    attempts = [
        extract_with_layout(path, layout)
        for layout in layouts
    ]

    if not attempts:
        raise ValueError("No layouts supplied")

    def score(attempt):
        return (
            attempt["detected_categories"],
            len(attempt["results"]),
            -len(attempt["errors"]),
        )

    best_score = max(
        score(attempt)
        for attempt in attempts
    )

    best_attempts = [
        attempt
        for attempt in attempts
        if score(attempt) == best_score
    ]

    selected = best_attempts[0]

    # If equally ranked layouts disagree on prices,
    # do not silently accept either extraction.
    if len(best_attempts) > 1:
        signatures = set()

        for attempt in best_attempts:
            signature = tuple(sorted(
                (
                    item["category"],
                    item["elevation"],
                    item["min_lkr"],
                    item["max_lkr"],
                )
                for item in attempt["results"]
            ))

            signatures.add(signature)

        if len(signatures) > 1:
            selected["errors"].append(
                "Ambiguous layout: equally ranked "
                "methods produced different prices"
            )

    return selected


# ============================================================
# FULL BATCH AUDIT
# ============================================================

def main():
    reports = sorted(
        REPORT_DIR.glob("*.pdf")
    )

    stats = {
        year: Counter()
        for year in LAYOUTS
    }

    layout_usage = Counter()
    review_reports = []
    total_quotations = 0

    print(
        "STEP 92 - HYBRID HISTORICAL "
        "DUST EXTRACTION AUDIT"
    )
    print("=" * 65)
    print("PDF files:", len(reports))

    for index, path in enumerate(
        reports,
        start=1,
    ):
        year = identify_year(path.name)

        if year not in LAYOUTS:
            review_reports.append(
                (path.name, "Unknown year")
            )
            continue

        stats[year]["reports"] += 1

        try:
            result = extract_report(
                path,
                LAYOUTS[year],
            )

            accepted = len(result["results"])

            stats[year]["quotations"] += accepted
            total_quotations += accepted

            layout_usage[
                result["layout"]
            ] += 1

            complete = (
                accepted == EXPECTED_QUOTATIONS
                and result["detected_categories"] == 7
                and not result["errors"]
                and not result["missing_categories"]
            )

            if complete:
                stats[year]["complete"] += 1

            else:
                stats[year]["review"] += 1

                review_reports.append((
                    path.name,
                    f"{accepted}/21 quotations; "
                    f"{result['detected_categories']}/7 "
                    f"categories; "
                    f"{len(result['errors'])} errors; "
                    f"layout={result['layout']}"
                ))

        except Exception as error:
            stats[year]["review"] += 1

            review_reports.append(
                (path.name, f"Error: {error}")
            )

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(reports)}"
            )

    print("\nCOVERAGE BY YEAR")
    print("=" * 65)

    for year in sorted(stats):
        values = stats[year]

        print(f"\nYEAR: {year}")
        print("Reports:", values["reports"])
        print("Complete:", values["complete"])
        print("Review:", values["review"])
        print("Quotations:", values["quotations"])

    print("\nLAYOUT USAGE")
    print("=" * 65)

    for name, count in sorted(
        layout_usage.items()
    ):
        print(f"{name}: {count}")

    print("\nOVERALL SUMMARY")
    print("=" * 65)

    print(
        "Candidate quotations:",
        total_quotations,
    )

    print(
        "Complete reports:",
        sum(
            values["complete"]
            for values in stats.values()
        ),
    )

    print(
        "Reports needing review:",
        len(review_reports),
    )

    print(
        "Theoretical maximum:",
        len(reports) * EXPECTED_QUOTATIONS,
    )

    print("\nREPORTS REQUIRING REVIEW")
    print("=" * 65)

    for filename, reason in review_reports:
        print(filename, "|", reason)

    print("\nIMPORTANT")
    print(
        "Candidate quotations require source validation."
    )
    print(
        "No CSV files or trained models were modified."
    )


if __name__ == "__main__":
    main()
