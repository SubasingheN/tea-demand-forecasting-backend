
from pathlib import Path
from decimal import Decimal, InvalidOperation

import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = (
    ROOT
    / "data"
    / "external"
    / "validated_audit"
)

SOURCE_CSV = (
    AUDIT_DIR
    / "four_grade_candidate_quotations_v2.csv"
)

PROPOSALS_CSV = (
    AUDIT_DIR
    / "2021_bop_recovery_proposals.csv"
)

OUTPUT_CSV = (
    AUDIT_DIR
    / "four_grade_candidate_quotations_v3.csv"
)

ALLOWED_STATUSES = {"OK", "SINGLE_VALUE"}

EXPECTED_PROPOSALS = 136
EXPECTED_CORRECTIONS = 134
EXPECTED_UNRESOLVED = 2


# ============================================================
# SAFE PRICE CONVERSION
# ============================================================

def parse_integer_price(value, record_key):
    """
    Accept integral values represented as:
        670
        670.0
        670.00

    Reject:
        670.5
        empty values
        invalid strings

    This prevents accidental rounding.
    """

    text = str(value).strip().replace(",", "")

    if not text:
        raise ValueError(
            f"Missing proposed price for {record_key}"
        )

    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"Invalid price {text!r} for {record_key}"
        ) from error

    if not number.is_finite():
        raise ValueError(
            f"Non-finite price for {record_key}"
        )

    if number != number.to_integral_value():
        raise ValueError(
            f"Non-integer price {text!r} "
            f"for {record_key}"
        )

    result = int(number)

    if result <= 0:
        raise ValueError(
            f"Non-positive price for {record_key}"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "STEP 134 - VALIDATE AND APPLY "
        "2021 BOP RECOVERIES"
    )
    print("=" * 70)

    for path in (SOURCE_CSV, PROPOSALS_CSV):
        if not path.exists():
            raise FileNotFoundError(path)

    # Read as text to preserve original raw quotations.
    source = pd.read_csv(
        SOURCE_CSV,
        keep_default_na=False,
        dtype=str,
    )

    proposals = pd.read_csv(
        PROPOSALS_CSV,
        keep_default_na=False,
        dtype=str,
    )

    print("Source records:", len(source))
    print("Recovery proposals:", len(proposals))

    # ========================================================
    # COLUMN VALIDATION
    # ========================================================

    required_source = {
        "source_file",
        "quality_category",
        "grade",
        "price_status",
        "price_min_lkr",
        "price_max_lkr",
        "raw_cell",
    }

    required_proposals = {
        "source_file",
        "quality_category",
        "grade",
        "original_status",
        "original_raw_cell",
        "proposed_status",
        "proposed_raw_cell",
        "proposed_min_lkr",
        "proposed_max_lkr",
        "pdf_page",
    }

    missing_source = required_source - set(source.columns)
    missing_proposals = (
        required_proposals - set(proposals.columns)
    )

    if missing_source:
        raise ValueError(
            f"Missing source columns: {missing_source}"
        )

    if missing_proposals:
        raise ValueError(
            f"Missing proposal columns: {missing_proposals}"
        )

    keys = [
        "source_file",
        "quality_category",
        "grade",
    ]

    if source.duplicated(subset=keys).any():
        raise ValueError("Duplicate source records")

    if proposals.duplicated(subset=keys).any():
        raise ValueError("Duplicate proposals")

    if len(proposals) != EXPECTED_PROPOSALS:
        raise ValueError(
            f"Expected {EXPECTED_PROPOSALS} proposals, "
            f"found {len(proposals)}"
        )

    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    updated = source.copy()

    if "correction_status" not in updated.columns:
        updated["correction_status"] = "unchanged"

    updated["recovery_raw_cell"] = ""
    updated["recovery_pdf_page"] = ""
    updated["recovery_verification_status"] = ""

    lookup = {
        tuple(row[column] for column in keys): index
        for index, row in updated.iterrows()
    }

    validated = []
    unresolved = []

    # ========================================================
    # VALIDATE EVERY PROPOSAL FIRST
    # ========================================================

    for _, proposal in proposals.iterrows():
        key = tuple(
            proposal[column]
            for column in keys
        )

        if key not in lookup:
            raise ValueError(
                f"No matching source record: {key}"
            )

        index = lookup[key]
        old = updated.loc[index]

        if (
            old["price_status"]
            != proposal["original_status"]
        ):
            raise ValueError(
                f"Original status mismatch: {key}"
            )

        if (
            old["raw_cell"]
            != proposal["original_raw_cell"]
        ):
            raise ValueError(
                f"Original raw quotation mismatch: {key}"
            )

        if old["price_status"] != "REVIEW":
            raise ValueError(
                f"Source record is not REVIEW: {key}"
            )

        if not str(key[0]).startswith("forbes_2021_"):
            raise ValueError(
                f"Unexpected source year: {key}"
            )

        if key[2] != "BOP":
            raise ValueError(
                f"Unexpected grade: {key}"
            )

        status = proposal["proposed_status"]

        if status not in ALLOWED_STATUSES:
            unresolved.append(key)
            continue

        # FIX: Supports strings such as '670.0'.
        minimum = parse_integer_price(
            proposal["proposed_min_lkr"],
            key,
        )

        maximum = parse_integer_price(
            proposal["proposed_max_lkr"],
            key,
        )

        if minimum > maximum:
            raise ValueError(
                f"Minimum exceeds maximum: {key}"
            )

        if (
            status == "SINGLE_VALUE"
            and minimum != maximum
        ):
            raise ValueError(
                f"Single value has unequal prices: {key}"
            )

        # Verify the proposed text agrees with the
        # numerical values before accepting it.
        raw = str(
            proposal["proposed_raw_cell"]
        ).replace("–", "-").replace("—", "-")

        expected_raw = (
            f"{minimum}-{maximum}"
            if status == "OK"
            else str(minimum)
        )

        normalized_raw = raw.replace(",", "")

        if normalized_raw != expected_raw:
            raise ValueError(
                f"Quotation text/number mismatch "
                f"for {key}: {raw!r}"
            )

        validated.append(
            (
                index,
                status,
                minimum,
                maximum,
                raw,
                proposal["pdf_page"],
            )
        )

    if len(validated) != EXPECTED_CORRECTIONS:
        raise ValueError(
            f"Expected {EXPECTED_CORRECTIONS} "
            f"corrections, found {len(validated)}"
        )

    if len(unresolved) != EXPECTED_UNRESOLVED:
        raise ValueError(
            f"Expected {EXPECTED_UNRESOLVED} "
            f"unresolved proposals, found "
            f"{len(unresolved)}"
        )

    # ========================================================
    # APPLY VALIDATED CORRECTIONS
    # ========================================================

    for (
        index,
        status,
        minimum,
        maximum,
        proposed_raw,
        pdf_page,
    ) in validated:

        # Assign strings to string columns.
        updated.at[
            index, "price_min_lkr"
        ] = str(minimum)

        updated.at[
            index, "price_max_lkr"
        ] = str(maximum)

        updated.at[
            index, "price_status"
        ] = status

        updated.at[
            index, "correction_status"
        ] = "corrected_step134_candidate"

        # Preserve the original raw_cell.
        # Store the corrected quotation separately.
        updated.at[
            index, "recovery_raw_cell"
        ] = proposed_raw

        updated.at[
            index, "recovery_pdf_page"
        ] = pdf_page

        updated.at[
            index, "recovery_verification_status"
        ] = "candidate_unverified"

        if "verification_status" in updated.columns:
            updated.at[
                index, "verification_status"
            ] = "candidate_unverified"

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(updated) != len(source):
        raise ValueError(
            "Record count changed unexpectedly"
        )

    if updated.duplicated(subset=keys).any():
        raise ValueError(
            "Duplicate records after correction"
        )

    changed_mask = (
        updated["correction_status"]
        == "corrected_step134_candidate"
    )

    changed_count = int(changed_mask.sum())

    if changed_count != EXPECTED_CORRECTIONS:
        raise ValueError(
            "Correction count mismatch"
        )

    changed_indices = {
        item[0]
        for item in validated
    }

    unchanged_indices = [
        index
        for index in source.index
        if index not in changed_indices
    ]

    for column in source.columns:
        if not updated.loc[
            unchanged_indices, column
        ].equals(
            source.loc[
                unchanged_indices, column
            ]
        ):
            raise ValueError(
                f"Unexpected change in column: {column}"
            )

    # Confirm all original raw quotations remain intact.
    if not updated["raw_cell"].equals(
        source["raw_cell"]
    ):
        raise ValueError(
            "Original raw quotations were changed"
        )

    # ========================================================
    # SAVE NEW V3 CSV
    # ========================================================

    updated.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\nCORRECTION SUMMARY")
    print("=" * 70)

    print("Original records:", len(source))
    print("Output records:", len(updated))
    print("Proposals checked:", len(proposals))
    print("Corrections applied:", changed_count)
    print(
        "Unresolved proposals preserved:",
        len(unresolved),
    )

    print("\nPRICE STATUS BEFORE")
    print("=" * 70)

    print(
        source["price_status"]
        .value_counts()
        .to_string()
    )

    print("\nPRICE STATUS AFTER")
    print("=" * 70)

    print(
        updated["price_status"]
        .value_counts()
        .to_string()
    )

    print("\nUNRESOLVED RECORDS")
    print("=" * 70)

    for key in unresolved:
        print(key)

    print("\nOUTPUT FILE")
    print("=" * 70)
    print(OUTPUT_CSV)

    print("\nOriginal and V2 CSV files preserved.")
    print(
        "Recovered quotations remain "
        "unverified candidates."
    )
    print(
        "TFT dataset and trained model: unchanged."
    )


if __name__ == "__main__":
    main()
