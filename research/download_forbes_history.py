
from pathlib import Path
from collections import Counter
from datetime import datetime
from io import BytesIO
import csv
import hashlib
import re
import time

import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
MANIFEST = EXTERNAL / "forbes_download_manifest.csv"

REPORT_DIR = (
    EXTERNAL
    / "forbes_auction_reports"
    / "historical"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEARS = ["2021", "2023", "2024"]

# Controlled test: three pending URLs per year.
REPORTS_PER_YEAR = 50

REQUEST_DELAY_SECONDS = 3
MAX_PDF_BYTES = 30 * 1024 * 1024

HEADERS = {
    "User-Agent": "AcademicResearch/1.0"
}

MONTHS = (
    "JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|"
    "JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER"
)

DATE_PAIR = re.compile(
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s*/\s*"
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s+"
    rf"({MONTHS})\s+(\d{{4}})",
    re.IGNORECASE,
)

SINGLE_DATE = re.compile(
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s+"
    rf"({MONTHS})\s+(\d{{4}})",
    re.IGNORECASE,
)

SALE_NUMBER = re.compile(
    r"SALE\s*NO\s*:?\s*(\d+)",
    re.IGNORECASE,
)

FILENAME_DATE_PAIR = re.compile(
    r"Sale\s+of\s+(\d{1,2})\s*&\s*"
    r"(\d{1,2})\s+([A-Za-z]+)",
    re.IGNORECASE,
)

FILENAME_FIRST_MONTH = re.compile(
    r"Sale\s+of\s+(\d{1,2})\s+([A-Za-z]+)",
    re.IGNORECASE,
)


def read_manifest():
    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def save_manifest(rows, fields):
    temporary = MANIFEST.with_suffix(".tmp")

    with temporary.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(MANIFEST)


def extract_quotation_text(pages):
    sections = []

    for page in pages:
        text = page.extract_text() or ""

        if "QUOTATION" in text.upper():
            normalized = " ".join(text.split())
            sections.append(normalized)

    return "\n".join(sections)


def verify_pdf(content):
    if not content.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature")

    with pdfplumber.open(BytesIO(content)) as pdf:
        if not pdf.pages:
            raise ValueError("PDF has no pages")

        page_count = len(pdf.pages)

        first_text = " ".join(
            " ".join(
                (page.extract_text() or "").split()
            )
            for page in pdf.pages[:3]
        )

        quotation_text = extract_quotation_text(
            pdf.pages
        )

    sale_match = SALE_NUMBER.search(first_text)
    date_match = DATE_PAIR.search(first_text)

    if date_match:
        day = int(date_match.group(1))
        end_day = int(date_match.group(2))
        month = date_match.group(3)
        year = int(date_match.group(4))

        start_date = datetime.strptime(
            f"{day} {month} {year}",
            "%d %B %Y",
        )

        end_date = datetime.strptime(
            f"{end_day} {month} {year}",
            "%d %B %Y",
        )

        if end_date < start_date:
            raise ValueError("Invalid auction date range")

    else:
        date_match = SINGLE_DATE.search(first_text)

        if not date_match:
            raise ValueError(
                "Auction date not detected"
            )

        day = int(date_match.group(1))
        month = date_match.group(2)
        year = int(date_match.group(3))

        start_date = datetime.strptime(
            f"{day} {month} {year}",
            "%d %B %Y",
        )

    if not sale_match:
        raise ValueError("Sale number not detected")

    return {
        "auction_date": start_date.strftime("%Y-%m-%d"),
        "year": str(year),
        "sale_number": int(sale_match.group(1)),
        "pages": page_count,
        "quotation_text": quotation_text,
    }


def filename_date_mismatch(filename, verified_date):
    """
    Compare the first auction day in the filename
    with the verified first auction date.

    Return None if the filename cannot be parsed.
    """
    first_month = FILENAME_FIRST_MONTH.search(
        filename
    )

    pair = FILENAME_DATE_PAIR.search(filename)

    try:
        if first_month:
            day = int(first_month.group(1))
            month = first_month.group(2)
        elif pair:
            day = int(pair.group(1))
            month = pair.group(3)
        else:
            return None

        year = int(verified_date[:4])

        filename_date = datetime.strptime(
            f"{day} {month} {year}",
            "%d %B %Y",
        ).strftime("%Y-%m-%d")

        return filename_date != verified_date

    except ValueError:
        return None


def download_pdf(session, url):
    with session.get(
        url,
        headers=HEADERS,
        timeout=60,
        stream=True,
    ) as response:
        response.raise_for_status()

        chunks = []
        total = 0

        for chunk in response.iter_content(
            chunk_size=65536
        ):
            if not chunk:
                continue

            total += len(chunk)

            if total > MAX_PDF_BYTES:
                raise ValueError(
                    "PDF exceeds 30 MB limit"
                )

            chunks.append(chunk)

    return b"".join(chunks)


def load_existing_reports():
    hashes = set()
    identities = {}

    for path in sorted(REPORT_DIR.glob("*.pdf")):
        content = path.read_bytes()

        digest = hashlib.sha256(
            content
        ).hexdigest()

        metadata = verify_pdf(content)

        identity = (
            metadata["year"],
            metadata["sale_number"],
        )

        hashes.add(digest)

        identities.setdefault(
            identity, []
        ).append({
            "sha256": digest,
            "quotation_text": (
                metadata["quotation_text"]
            ),
            "filename": path.name,
        })

    return hashes, identities


def main():
    rows, fields = read_manifest()

    if not fields:
        raise ValueError(
            "Manifest is empty or invalid"
        )

    if "review_notes" not in fields:
        fields.append("review_notes")

    for row in rows:
        row.setdefault("review_notes", "")

    selected = []

    for year in TARGET_YEARS:
        pending = [
            row for row in rows
            if row["filename_year"] == year
            and row["download_status"] == "pending"
        ]

        selected.extend(
            pending[:REPORTS_PER_YEAR]
        )

    hashes, identities = load_existing_reports()

    print("STEP 60 — IMPROVED HISTORICAL DOWNLOADER")
    print("=" * 60)
    print("Selected pending URLs:", len(selected))
    print("Existing unique PDFs:", len(hashes))

    saved = 0
    duplicates = 0
    failures = 0
    mismatches = 0

    with requests.Session() as session:
        for index, row in enumerate(
            selected, start=1
        ):
            print(
                f"\n[{index}/{len(selected)}] "
                f"{row['filename']}"
            )

            try:
                content = download_pdf(
                    session, row["source_url"]
                )

                metadata = verify_pdf(content)

                digest = hashlib.sha256(
                    content
                ).hexdigest()

                identity = (
                    metadata["year"],
                    metadata["sale_number"],
                )

                if (
                    metadata["year"]
                    != row["filename_year"]
                ):
                    raise ValueError(
                        "Verified auction year differs "
                        "from filename year"
                    )

                row["verified_auction_date"] = (
                    metadata["auction_date"]
                )

                row["verified_auction_year"] = (
                    metadata["year"]
                )

                date_mismatch = filename_date_mismatch(
                    row["filename"],
                    metadata["auction_date"],
                )

                if date_mismatch:
                    mismatches += 1

                    row["review_notes"] = (
                        "Filename date differs from "
                        "verified PDF auction date."
                    )

                    print(
                        "WARNING: Filename date mismatch"
                    )

                existing_versions = identities.get(
                    identity, []
                )

                identical_quotations = any(
                    metadata["quotation_text"]
                    and metadata["quotation_text"]
                    == existing["quotation_text"]
                    for existing in existing_versions
                )

                if digest in hashes:
                    row["download_status"] = "duplicate"
                    row["validation_status"] = (
                        "duplicate_content"
                    )
                    duplicates += 1
                    print("Duplicate: identical PDF bytes")

                elif identical_quotations:
                    row["download_status"] = "duplicate"
                    row["validation_status"] = (
                        "duplicate_auction_quotations"
                    )
                    row["review_notes"] = (
                        row["review_notes"]
                        + " Same auction and identical "
                        "extracted quotation text."
                    ).strip()
                    duplicates += 1

                    print(
                        "Duplicate: identical quotation text"
                    )

                elif existing_versions:
                    row["download_status"] = "failed"
                    row["validation_status"] = (
                        "conflicting_sale_requires_review"
                    )
                    row["review_notes"] = (
                        row["review_notes"]
                        + " Same sale number but different "
                        "quotation text; manual review needed."
                    ).strip()

                    failures += 1

                    print(
                        "REQUIRES REVIEW: conflicting "
                        "quotation text for same sale"
                    )

                else:
                    filename = (
                        f"forbes_{metadata['year']}_"
                        f"sale_{metadata['sale_number']:02d}_"
                        f"{digest[:12]}.pdf"
                    )

                    output = REPORT_DIR / filename

                    if output.exists():
                        raise FileExistsError(output)

                    output.write_bytes(content)

                    hashes.add(digest)

                    identities.setdefault(
                        identity, []
                    ).append({
                        "sha256": digest,
                        "quotation_text": (
                            metadata["quotation_text"]
                        ),
                        "filename": filename,
                    })

                    row["download_status"] = "downloaded"

                    row["validation_status"] = (
                        "metadata_verified_filename_mismatch"
                        if date_mismatch
                        else "metadata_verified"
                    )

                    saved += 1

                    print("Saved:", filename)

                print(
                    "Verified auction date:",
                    metadata["auction_date"]
                )
                print(
                    "Sale number:",
                    metadata["sale_number"]
                )

            except Exception as error:
                row["download_status"] = "failed"
                row["validation_status"] = (
                    "requires_review"
                )
                row["review_notes"] = str(error)

                failures += 1

                print("ERROR:", error)

            finally:
                save_manifest(rows, fields)

                if index < len(selected):
                    time.sleep(
                        REQUEST_DELAY_SECONDS
                    )

    counts = Counter(
        row["download_status"]
        for row in rows
    )

    print("\n" + "=" * 60)
    print("STEP 60 DOWNLOAD SUMMARY")
    print("=" * 60)

    print("New unique PDFs:", saved)
    print("New duplicate URLs:", duplicates)
    print("New filename-date mismatches:", mismatches)
    print("New records requiring review:", failures)
    print("Total downloaded:", counts["downloaded"])
    print("Total duplicate:", counts["duplicate"])
    print("Remaining pending:", counts["pending"])
    print("Total failed:", counts["failed"])
    print("Total manifest records:", len(rows))
    print("Reports folder:", REPORT_DIR)


if __name__ == "__main__":
    main()
