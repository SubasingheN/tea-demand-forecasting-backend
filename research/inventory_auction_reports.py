
from pathlib import Path
import csv
import hashlib
import re

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    ROOT / "data" / "external"
    / "forbes_auction_reports"
)

OUTPUT_FILE = (
    ROOT / "data" / "external"
    / "auction_report_inventory.csv"
)

DATE_PATTERN = re.compile(
    r"(\d{1,2})(?:ST|ND|RD|TH)?\s*/\s*"
    r"(\d{1,2})(?:ST|ND|RD|TH)?\s+"
    r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|"
    r"JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
    r"\s+(\d{4})",
    re.IGNORECASE,
)

SALE_PATTERN = re.compile(
    r"SALE\s*NO\s*:?\s*(\d+)",
    re.IGNORECASE,
)


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_report(path):
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)

        cover_text = " ".join(
            (pdf.pages[0].extract_text() or "").split()
        )

        first_pages_text = " ".join(
            " ".join(
                (page.extract_text() or "").split()
            )
            for page in pdf.pages[:3]
        )

    date_match = DATE_PATTERN.search(
        first_pages_text
    )

    sale_match = SALE_PATTERN.search(
        first_pages_text
    )

    return {
        "filename": path.name,
        "pages": page_count,
        "sale_number": (
            sale_match.group(1)
            if sale_match else ""
        ),
        "auction_date_text": (
            date_match.group(0)
            if date_match else ""
        ),
        "date_found": bool(date_match),
        "sha256": file_hash(path),
        "cover_preview": cover_text[:150],
    }


def build_inventory():
    if not REPORT_DIR.exists():
        raise FileNotFoundError(REPORT_DIR)

    reports = sorted(REPORT_DIR.glob("*.pdf"))

    if not reports:
        raise ValueError(
            "No PDF reports found."
        )

    records = []

    for path in reports:
        print("Inspecting:", path.name)

        try:
            records.append(inspect_report(path))
        except Exception as error:
            print("ERROR:", error)

    hashes = [r["sha256"] for r in records]

    fields = [
        "filename",
        "pages",
        "sale_number",
        "auction_date_text",
        "date_found",
        "sha256",
        "cover_preview",
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nInventory completed.")
    print("PDF files found:", len(reports))
    print("Successfully inspected:", len(records))
    print("Duplicate files:", len(hashes) - len(set(hashes)))
    print("Output:", OUTPUT_FILE)

    for row in records:
        print(
            row["filename"],
            "| Sale:", row["sale_number"],
            "| Date:", row["auction_date_text"],
            "| Pages:", row["pages"],
        )


if __name__ == "__main__":
    build_inventory()
