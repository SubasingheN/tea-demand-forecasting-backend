
from pathlib import Path
from collections import Counter
import re
import statistics

import pandas as pd
import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT / "data" / "external"
    / "forbes_auction_reports" / "historical"
)

AUDIT_DIR = (
    ROOT / "data" / "external"
    / "validated_audit"
)

SOURCE_CSV = (
    AUDIT_DIR
    / "four_grade_candidate_quotations_v2.csv"
)

OUTPUT_CSV = (
    AUDIT_DIR
    / "2021_bop_recovery_proposals.csv"
)

CATEGORIES = {
    "bestwesterns",
    "belowbestwesterns",
    "plainerwesterns",
    "nuwaraeliyas",
    "brighterudapussellawas",
    "otherudapussellawas",
    "bestuvas",
    "otheruvas",
    "goodmediums",
    "othermediums",
}


def normalize(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def header_page(page):
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
            " ".join(lines[index:index + 4]).upper(),
        )

        if all(
            term in window
            for term in ("BOP", "BOPF", "PEKOE", "OP")
        ) and (
            "RS/KG" in window or "LKR" in window
        ):
            return True

    return False


def detect_category(row):
    for boundary in (130, 140, 160, 180, 195):
        label = normalize(
            "".join(
                word["text"]
                for word in row
                if word["x0"] < boundary
            )
        )

        if label in CATEGORIES:
            return label

    return None


def get_rows(pdf):
    rows = {}

    for page_number, page in enumerate(
        pdf.pages, start=1
    ):
        if not header_page(page):
            continue

        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
        )

        ys = sorted({
            round(word["top"], 1)
            for word in words
            if word["x0"] < 180
        })

        processed = []

        for y in ys:
            if any(
                abs(y - previous) < 3
                for previous in processed
            ):
                continue

            processed.append(y)

            row = sorted(
                [
                    word
                    for word in words
                    if abs(word["top"] - y) < 2
                ],
                key=lambda word: word["x0"],
            )

            category = detect_category(row)

            if category and category not in rows:
                rows[category] = (
                    page_number,
                    row,
                )

    return rows


def quotation_starts(row):
    words = sorted(
        [
            word for word in row
            if word["x0"] >= 180
        ],
        key=lambda word: word["x0"],
    )

    starts = []

    for index, word in enumerate(words):
        if not re.fullmatch(
            r"\d[\d,]*",
            word["text"],
        ):
            continue

        if index + 1 >= len(words):
            continue

        following = words[index + 1]

        if (
            following["text"] in {"-", "–", "—", "¿"}
            and 0 < following["x0"] - word["x0"] < 25
        ):
            starts.append(word["x0"])

    return starts


def estimated_bop_position(rows):
    positions = []

    for _, row in rows.values():
        for x in quotation_starts(row):
            if 225 <= x < 263:
                positions.append(x)

    if len(positions) < 5:
        return None, len(positions)

    return statistics.median(positions), len(positions)


def parse_bop(row, position):
    words = sorted(
        [
            word for word in row
            if position - 5 <= word["x0"] < position + 39
        ],
        key=lambda word: word["x0"],
    )

    raw = "".join(
        word["text"]
        for word in words
    ).replace("–", "-").replace("—", "-")

    match = re.fullmatch(
        r"(\d[\d,]*)-(\d[\d,]*)",
        raw,
    )

    if match:
        minimum = int(
            match.group(1).replace(",", "")
        )
        maximum = int(
            match.group(2).replace(",", "")
        )

        if 0 < minimum <= maximum:
            return "OK", minimum, maximum, raw

    if raw.upper() in {"", "-", "N/A", "NA"}:
        return "MISSING", None, None, raw

    if re.fullmatch(r"\d[\d,]*", raw):
        value = int(raw.replace(",", ""))
        return "SINGLE_VALUE", value, value, raw

    return "REVIEW", None, None, raw


def main():
    print(
        "STEP 133 - EXPORT 2021 BOP "
        "RECOVERY PROPOSALS"
    )
    print("=" * 70)

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)

    df = pd.read_csv(
        SOURCE_CSV,
        keep_default_na=False,
    )

    reviews = df[
        df["source_file"].str.startswith(
            "forbes_2021_"
        )
        & df["grade"].eq("BOP")
        & df["price_status"].eq("REVIEW")
    ].copy()

    print("BOP records requiring review:", len(reviews))

    proposals = []
    counts = Counter()

    for filename, group in reviews.groupby("source_file"):
        pdf_path = REPORT_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        with pdfplumber.open(pdf_path) as pdf:
            rows = get_rows(pdf)

        position, observations = estimated_bop_position(
            rows
        )

        if position is None:
            raise ValueError(
                f"Insufficient column observations: "
                f"{filename} ({observations})"
            )

        for _, record in group.iterrows():
            category = str(
                record["quality_category"]
            )

            key = normalize(category)

            if key not in rows:
                raise ValueError(
                    f"Missing category {category}: "
                    f"{filename}"
                )

            page_number, row = rows[key]

            status, minimum, maximum, raw = parse_bop(
                row,
                position,
            )

            counts[status] += 1

            # Recovered values remain proposals.
            # Do not automatically modify the
            # original quotation dataset.
            proposals.append({
                "source_file": filename,
                "pdf_page": page_number,
                "quality_category": category,
                "grade": "BOP",
                "original_status": record["price_status"],
                "original_raw_cell": record["raw_cell"],
                "proposed_status": status,
                "proposed_raw_cell": raw,
                "proposed_min_lkr": minimum,
                "proposed_max_lkr": maximum,
                "estimated_column_x": round(
                    position, 2
                ),
                "column_observations": observations,
                "verification_status": (
                    "candidate_unverified"
                ),
            })

    result = pd.DataFrame(proposals)

    expected_columns = [
        "source_file",
        "pdf_page",
        "quality_category",
        "grade",
        "original_status",
        "original_raw_cell",
        "proposed_status",
        "proposed_raw_cell",
        "proposed_min_lkr",
        "proposed_max_lkr",
        "estimated_column_x",
        "column_observations",
        "verification_status",
    ]

    result = result[expected_columns]

    if len(result) != len(reviews):
        raise ValueError(
            "Proposal count does not match review count"
        )

    if result.duplicated(
        subset=[
            "source_file",
            "quality_category",
            "grade",
        ]
    ).any():
        raise ValueError(
            "Duplicate correction proposals detected"
        )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPROPOSAL SUMMARY")
    print("=" * 70)

    print("Original REVIEW records:", len(reviews))
    print("Proposed valid ranges:", counts["OK"])
    print(
        "Proposed single values:",
        counts["SINGLE_VALUE"],
    )
    print("Still malformed:", counts["REVIEW"])
    print("Missing quotations:", counts["MISSING"])
    print("Total proposals:", len(result))

    print("\nUNRESOLVED PROPOSALS")
    print("=" * 70)

    unresolved = result[
        result["proposed_status"].isin([
            "REVIEW",
            "MISSING",
        ])
    ]

    if unresolved.empty:
        print("None")
    else:
        print(
            unresolved[
                [
                    "source_file",
                    "quality_category",
                    "proposed_raw_cell",
                    "proposed_status",
                ]
            ].to_string(index=False)
        )

    print("\nOUTPUT FILE")
    print("=" * 70)
    print(OUTPUT_CSV)

    print(
        "\nOriginal and v2 quotation CSVs: unchanged."
    )
    print(
        "All proposals require source verification."
    )
    print(
        "TFT dataset and trained model: unchanged."
    )


if __name__ == "__main__":
    main()
