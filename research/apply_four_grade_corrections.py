
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = (
    ROOT
    / "data"
    / "external"
    / "validated_audit"
)

SOURCE_FILE = (
    AUDIT_DIR
    / "four_grade_candidate_quotations.csv"
)

OUTPUT_FILE = (
    AUDIT_DIR
    / "four_grade_candidate_quotations_v2.csv"
)


# ============================================================
# STEP 129 CORRECTIONS
# ============================================================

# Format:
# (sale number, quality category, grade):
#     (minimum price, maximum price)

CORRECTIONS = {
    ("09", "Best Westerns", "BOP"): (740, 1300),

    ("11", "Below Best Westerns", "BOP"): (630, 660),
    ("11", "Plainer Westerns", "BOP"): (580, 620),
    ("11", "Nuwara Eliyas", "BOP"): (940, 1000),
    ("11", "Brighter Udapussellawas", "BOP"): (590, 680),
    ("11", "Other Udapussellawas", "BOP"): (560, 580),
    ("11", "Best Uvas", "BOP"): (680, 700),
    ("11", "Other Uvas", "BOP"): (610, 650),
    ("11", "Good Mediums", "BOP"): (740, 830),
    ("11", "Other Mediums", "BOP"): (600, 630),
}


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "STEP 130 - APPLY AUDITED "
        "FOUR-GRADE CORRECTIONS"
    )
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {SOURCE_FILE}"
        )

    # Read existing data without altering source CSV.
    df = pd.read_csv(
        SOURCE_FILE,
        keep_default_na=False,
    )

    print("Source records:", len(df))
    print("Corrections requested:", len(CORRECTIONS))

    required_columns = [
        "source_file",
        "quality_category",
        "grade",
        "price_min_lkr",
        "price_max_lkr",
        "price_status",
        "raw_cell",
        "verification_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Work on a copy. The original remains unchanged.
    corrected = df.copy()

    # FIX:
    # Explicitly convert the price columns to
    # nullable numeric types before assigning integers.
    #
    # Empty cells become <NA>, not zero.
    for column in [
        "price_min_lkr",
        "price_max_lkr",
    ]:
        corrected[column] = pd.to_numeric(
            corrected[column].replace("", pd.NA),
            errors="raise",
        ).astype("Int64")

    corrected["correction_status"] = "unchanged"

    # Validate every requested correction before
    # applying any change.
    validated = []

    for key, (minimum, maximum) in CORRECTIONS.items():
        sale, category, grade = key

        if not (0 < minimum <= maximum):
            raise ValueError(
                f"Invalid correction values for {key}"
            )

        sale_prefix = (
            f"forbes_2021_sale_{sale}_"
        )

        mask = (
            corrected["source_file"]
            .str.startswith(sale_prefix)
            & (
                corrected["quality_category"]
                == category
            )
            & (
                corrected["grade"]
                == grade
            )
        )

        matching_indices = corrected.index[mask]

        if len(matching_indices) != 1:
            raise ValueError(
                f"Expected exactly one record for "
                f"{sale}/{category}/{grade}; "
                f"found {len(matching_indices)}"
            )

        row_index = matching_indices[0]

        previous_status = str(
            corrected.at[row_index, "price_status"]
        )

        if previous_status != "REVIEW":
            raise ValueError(
                f"Unexpected price status for "
                f"{sale}/{category}/{grade}: "
                f"{previous_status}"
            )

        validated.append(
            (row_index, minimum, maximum)
        )

    # All ten records passed validation.
    # Now apply the corrections.
    for row_index, minimum, maximum in validated:
        corrected.at[
            row_index, "price_min_lkr"
        ] = minimum

        corrected.at[
            row_index, "price_max_lkr"
        ] = maximum

        corrected.at[
            row_index, "price_status"
        ] = "OK"

        corrected.at[
            row_index, "correction_status"
        ] = "corrected_step129_candidate"

        # Retain the original raw_cell for auditing.
        # These prices are NOT yet source-verified.
        corrected.at[
            row_index, "verification_status"
        ] = "candidate_unverified"

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(corrected) != len(df):
        raise ValueError(
            "Record count changed unexpectedly"
        )

    duplicate_columns = [
        "source_file",
        "quality_category",
        "grade",
    ]

    if corrected.duplicated(
        subset=duplicate_columns
    ).any():
        raise ValueError(
            "Duplicate source/category/grade records"
        )

    applied = int(
        (
            corrected["correction_status"]
            == "corrected_step129_candidate"
        ).sum()
    )

    if applied != len(CORRECTIONS):
        raise ValueError(
            f"Expected {len(CORRECTIONS)} corrections, "
            f"but applied {applied}"
        )

    changed = (
        corrected["correction_status"]
        == "corrected_step129_candidate"
    )

    for _, row in corrected.loc[changed].iterrows():
        minimum = row["price_min_lkr"]
        maximum = row["price_max_lkr"]

        if (
            pd.isna(minimum)
            or pd.isna(maximum)
            or minimum <= 0
            or minimum > maximum
        ):
            raise ValueError(
                "Invalid corrected price range"
            )

    # ========================================================
    # EXPORT NEW CSV
    # ========================================================

    corrected.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\nCORRECTION SUMMARY")
    print("=" * 70)

    print("Source records:", len(df))
    print("Output records:", len(corrected))
    print("Corrections applied:", applied)

    print("\nPRICE STATUS BEFORE")
    print(
        df["price_status"]
        .value_counts()
        .to_string()
    )

    print("\nPRICE STATUS AFTER")
    print(
        corrected["price_status"]
        .value_counts()
        .to_string()
    )

    print("\nOUTPUT FILE")
    print("=" * 70)
    print(OUTPUT_FILE)

    print("\nVALIDATION")
    print("=" * 70)

    print("Duplicate records: 0")
    print("Record count preserved: YES")
    print("Original CSV preserved: YES")

    print(
        "\nAll corrected quotations remain "
        "unverified candidates."
    )

    print(
        "Factory dataset and TFT model: unchanged."
    )


if __name__ == "__main__":
    main()
