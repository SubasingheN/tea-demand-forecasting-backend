
from pathlib import Path
from collections import Counter
import csv
import json
import re

import pdfplumber


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT
    / "data"
    / "external"
    / "forbes_auction_reports"
    / "historical"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "external"
    / "validated_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TEA GRADES AND QUALITY CATEGORIES
# ============================================================

GRADES = (
    "BOP",
    "BOPF",
    "PEKOE/FBOP",
    "OP",
)

CATEGORIES = {
    "bestwesterns": ("Best Westerns", "High"),
    "belowbestwesterns": (
        "Below Best Westerns", "High"
    ),
    "plainerwesterns": ("Plainer Westerns", "High"),
    "nuwaraeliyas": ("Nuwara Eliyas", "High"),
    "brighterudapussellawas": (
        "Brighter Udapussellawas", "High"
    ),
    "otherudapussellawas": (
        "Other Udapussellawas", "High"
    ),
    "bestuvas": ("Best Uvas", "High"),
    "otheruvas": ("Other Uvas", "High"),
    "goodmediums": ("Good Mediums", "Medium"),
    "othermediums": ("Other Mediums", "Medium"),
}


# ============================================================
# LAYOUT CONFIGURATIONS
# ============================================================

LAYOUTS = {
    "2021_early": {
        "year": "2021",
        "high": [245, 327, 408, 489],
        "medium": [245, 327, 408, 489],
        "width": 39,
    },
    "2021_late": {
        "year": "2021",
        "high": [242, 327, 412, 497],
        "medium": [242, 327, 412, 497],
        "width": 40,
    },
    "2023_early": {
        "year": "2023",
        "high": [247, 321, 394, 468],
        "medium": [248, 321, 394, 468],
        "width": 35,
    },
    "2023_middle": {
        "year": "2023",
        "high": [239, 322, 405, 487],
        "medium": [241, 322, 404, 487],
        "width": 38,
    },
    "2023_late": {
        "year": "2023",
        "high": [208, 302, 395, 489],
        "medium": [191, 285, 378, 472],
        "width": 43,
    },
    "2024_standard": {
        "year": "2024",
        "high": [208, 302, 395, 489],
        "medium": [191, 285, 378, 472],
        "width": 43,
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower(),
    )


def report_year(filename):
    match = re.search(
        r"forbes_(20\d{2})_",
        filename,
    )

    return (
        match.group(1)
        if match
        else None
    )


def has_quotation_header(page):
    text = page.extract_text(
        layout=True,
        x_tolerance=2,
        y_tolerance=3,
    ) or ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(lines):
        if "QUOTATION" not in line.upper():
            continue

        window = re.sub(
            r"\s+",
            "",
            " ".join(
                lines[index:index + 4]
            ).upper(),
        )

        if (
            "BOP" in window
            and "BOPF" in window
            and "PEKOE" in window
            and "OP" in window
            and (
                "LKR" in window
                or "RS/KG" in window
            )
        ):
            return True

    return False


def detect_category(row):
    for boundary in (
        130, 140, 160, 180, 195
    ):
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

    if raw.upper() in {
        "",
        "N/A",
        "NA",
        "-",
    }:
        return None, None, "MISSING", raw

    match = re.fullmatch(
        r"(\d{2,5})-(\d{2,5})",
        raw,
    )

    if match:
        low, high = map(
            int,
            match.groups(),
        )

        if 0 < low <= high:
            return low, high, "OK", raw

    if re.fullmatch(r"\d{2,5}", raw):
        price = int(raw)

        if price > 0:
            return (
                price,
                price,
                "SINGLE_VALUE",
                raw,
            )

    return None, None, "REVIEW", raw


# ============================================================
# SELECT LAYOUT
# ============================================================

def select_layout(year, best_westerns_row):
    """
    Compare year-appropriate layout configurations.

    Layout choice is provisional; candidate values
    remain unverified until source checking.
    """

    candidates = []

    for name, config in LAYOUTS.items():
        if config["year"] != year:
            continue

        valid_count = 0
        review_count = 0

        for position in config["high"]:
            cell = [
                word
                for word in best_westerns_row
                if (
                    position - 4
                    <= word["x0"]
                    < position + config["width"]
                )
            ]

            _, _, status, _ = parse_price_cell(cell)

            if status in {
                "OK",
                "SINGLE_VALUE",
            }:
                valid_count += 1
            elif status == "REVIEW":
                review_count += 1

        candidates.append(
            (
                valid_count,
                -review_count,
                name,
            )
        )

    if not candidates:
        return None, "No layout for report year"

    candidates.sort(
        reverse=True
    )

    best = candidates[0]

    equally_scored = [
        item
        for item in candidates
        if item[:2] == best[:2]
    ]

    if len(equally_scored) > 1:
        return (
            best[2],
            "AMBIGUOUS_LAYOUT",
        )

    if best[0] < 4:
        return (
            best[2],
            "PARTIAL_LAYOUT_MATCH",
        )

    return best[2], "MATCHED"


# ============================================================
# EXTRACT REPORT
# ============================================================

def extract_report(path):
    year = report_year(path.name)

    if year is None:
        raise ValueError(
            "Cannot determine report year"
        )

    category_rows = {}

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):
            if not has_quotation_header(page):
                continue

            words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
            )

            candidate_y = sorted({
                round(word["top"], 1)
                for word in words
                if word["x0"] < 180
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

                key = detect_category(row)

                if key is None:
                    continue

                if key not in category_rows:
                    category_rows[key] = (
                        page_number,
                        row,
                    )

    reference = category_rows.get(
        "bestwesterns"
    )

    if reference is None:
        raise ValueError(
            "Best Westerns quotation row missing"
        )

    layout_name, layout_status = select_layout(
        year,
        reference[1],
    )

    if layout_name is None:
        raise ValueError(
            layout_status
        )

    layout = LAYOUTS[layout_name]

    candidates = []
    reviews = []

    if layout_status != "MATCHED":
        reviews.append({
            "source_file": path.name,
            "year": year,
            "category": "",
            "grade": "",
            "reason": layout_status,
            "raw_cell": "",
        })

    for category_key, (
        category_name,
        elevation,
    ) in CATEGORIES.items():

        if category_key not in category_rows:
            reviews.append({
                "source_file": path.name,
                "year": year,
                "category": category_name,
                "grade": "",
                "reason": "CATEGORY_MISSING",
                "raw_cell": "",
            })
            continue

        page_number, row = category_rows[
            category_key
        ]

        positions = layout[
            elevation.lower()
        ]

        for grade, position in zip(
            GRADES,
            positions,
        ):
            cell = [
                word
                for word in row
                if (
                    position - 4
                    <= word["x0"]
                    < position + layout["width"]
                )
            ]

            low, high, status, raw = (
                parse_price_cell(cell)
            )

            record = {
                "source_file": path.name,
                "source_year": year,
                "source_page": page_number,
                "grade": grade,
                "quality_category":
                    category_name,
                "elevation": elevation,
                "price_min_lkr": low,
                "price_max_lkr": high,
                "raw_cell": raw,
                "layout": layout_name,
                "layout_status":
                    layout_status,
                "price_status": status,
                "verification_status":
                    "candidate_unverified",
            }

            candidates.append(record)

            if status != "OK":
                reviews.append({
                    "source_file": path.name,
                    "year": year,
                    "category": category_name,
                    "grade": grade,
                    "reason": status,
                    "raw_cell": raw,
                })

    return candidates, reviews, layout_name


# ============================================================
# EXPORT RESULTS
# ============================================================

def write_csv(path, rows, columns):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    reports = sorted(
        REPORT_DIR.glob("*.pdf")
    )

    all_candidates = []
    all_reviews = []

    layouts_used = Counter()
    report_errors = []

    print(
        "STEP 113 - HISTORICAL FOUR-GRADE "
        "CANDIDATE EXTRACTION"
    )
    print("=" * 70)

    print("PDF reports:", len(reports))

    for index, path in enumerate(
        reports,
        start=1,
    ):
        try:
            candidates, reviews, layout = (
                extract_report(path)
            )

            all_candidates.extend(
                candidates
            )
            all_reviews.extend(
                reviews
            )

            layouts_used[layout] += 1

        except Exception as error:
            report_errors.append({
                "source_file": path.name,
                "year": report_year(path.name),
                "category": "",
                "grade": "",
                "reason": (
                    f"PROCESSING_ERROR: {error}"
                ),
                "raw_cell": "",
            })

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(reports)}"
            )

    all_reviews.extend(
        report_errors
    )

    candidate_path = (
        OUTPUT_DIR
        / "four_grade_candidate_quotations.csv"
    )

    review_path = (
        OUTPUT_DIR
        / "four_grade_extraction_reviews.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "four_grade_extraction_summary.json"
    )

    candidate_columns = [
        "source_file",
        "source_year",
        "source_page",
        "grade",
        "quality_category",
        "elevation",
        "price_min_lkr",
        "price_max_lkr",
        "raw_cell",
        "layout",
        "layout_status",
        "price_status",
        "verification_status",
    ]

    review_columns = [
        "source_file",
        "year",
        "category",
        "grade",
        "reason",
        "raw_cell",
    ]

    write_csv(
        candidate_path,
        all_candidates,
        candidate_columns,
    )

    write_csv(
        review_path,
        all_reviews,
        review_columns,
    )

    status_counts = Counter(
        row["price_status"]
        for row in all_candidates
    )

    summary = {
        "report_count": len(reports),
        "extracted_candidate_cells":
            len(all_candidates),
        "expected_maximum_cells":
            len(reports) * len(CATEGORIES) * len(GRADES),
        "price_status_counts":
            dict(status_counts),
        "layout_usage":
            dict(layouts_used),
        "review_records":
            len(all_reviews),
        "processing_errors":
            len(report_errors),
        "verified_prices": 0,
        "note": (
            "All extracted values are candidates "
            "and require source validation."
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nEXTRACTION SUMMARY")
    print("=" * 70)

    print(
        "Candidate cells:",
        len(all_candidates),
    )
    print(
        "Expected maximum:",
        summary["expected_maximum_cells"],
    )

    for status, count in sorted(
        status_counts.items()
    ):
        print(
            f"{status}: {count}"
        )

    print(
        "Review records:",
        len(all_reviews),
    )
    print(
        "Processing errors:",
        len(report_errors),
    )

    print("\nLAYOUT USAGE")
    print("=" * 70)

    for name, count in sorted(
        layouts_used.items()
    ):
        print(name, ":", count)

    print("\nCREATED FILES")
    print("=" * 70)

    print(candidate_path)
    print(review_path)
    print(summary_path)

    print(
        "\nIMPORTANT: All quotations remain "
        "unverified candidates."
    )
    print(
        "PEKOE/FBOP is not pure Pekoe."
    )
    print(
        "Factory dataset and TFT model: unchanged."
    )


if __name__ == "__main__":
    main()
