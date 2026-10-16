
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
    EXTERNAL / "forbes_auction_reports" / "historical"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RECOVERY_LIMIT = 2
REQUEST_DELAY = 3
MAX_BYTES = 30 * 1024 * 1024

MONTHS = (
    "JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|"
    "JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER"
)

# Example: SALE OF - 17TH/18TH OCTOBER 2023
SAME_MONTH = re.compile(
    rf"SALE\s+OF\s*[-:]?\s*"
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s*/\s*"
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s+"
    rf"({MONTHS})\s+(\d{{4}})",
    re.IGNORECASE,
)

# Example: SALE OF - 31ST JULY/02ND AUGUST 2023
CROSS_MONTH = re.compile(
    rf"SALE\s+OF\s*[-:]?\s*"
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s+"
    rf"({MONTHS})\s*/\s*"
    rf"(\d{{1,2}})(?:ST|ND|RD|TH)?\s+"
    rf"({MONTHS})\s+(\d{{4}})",
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


def download_pdf(session, url):
    with session.get(
        url,
        timeout=60,
        stream=True,
        headers={
            "User-Agent": "AcademicResearch/1.0"
        },
    ) as response:
        response.raise_for_status()

        chunks = []
        size = 0

        for chunk in response.iter_content(65536):
            if not chunk:
                continue

            size += len(chunk)

            if size > MAX_BYTES:
                raise ValueError(
                    "PDF exceeds 30 MB limit"
                )

            chunks.append(chunk)

    content = b"".join(chunks)

    if not content.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature")

    return content


def extract_metadata(content):
    dates = []
    quotation_sections = []

    with pdfplumber.open(BytesIO(content)) as pdf:
        if not pdf.pages:
            raise ValueError("Empty PDF")

        page_count = len(pdf.pages)

        for page in pdf.pages[:6]:
            text = " ".join(
                (page.extract_text() or "").split()
            )

            # Check cross-month patterns first.
            for match in CROSS_MONTH.finditer(text):
                day1 = int(match.group(1))
                month1 = match.group(2)

                day2 = int(match.group(3))
                month2 = match.group(4)

                year = int(match.group(5))

                start = datetime.strptime(
                    f"{day1} {month1} {year}",
                    "%d %B %Y",
                ).date()

                end = datetime.strptime(
                    f"{day2} {month2} {year}",
                    "%d %B %Y",
                ).date()

                if end >= start:
                    dates.append((
                        start.isoformat(),
                        end.isoformat(),
                    ))

            for match in SAME_MONTH.finditer(text):
                day1 = int(match.group(1))
                day2 = int(match.group(2))
                month = match.group(3)
                year = int(match.group(4))

                start = datetime.strptime(
                    f"{day1} {month} {year}",
                    "%d %B %Y",
                ).date()

                end = datetime.strptime(
                    f"{day2} {month} {year}",
                    "%d %B %Y",
                ).date()

                if end >= start:
                    dates.append((
                        start.isoformat(),
                        end.isoformat(),
                    ))

        for page in pdf.pages:
            text = page.extract_text() or ""

            if "QUOTATION" in text.upper():
                quotation_sections.append(
                    " ".join(text.split())
                )

    counts = Counter(dates)

    if not counts:
        raise ValueError(
            "No current-auction date candidates"
        )

    ranked = counts.most_common()
    selected, match_count = ranked[0]

    if match_count < 2:
        raise ValueError(
            "Insufficient repeated date evidence"
        )

    if (
        len(ranked) > 1
        and ranked[1][1] == match_count
    ):
        raise ValueError(
            "Conflicting auction date candidates"
        )

    if selected[0][:4] != "2023":
        raise ValueError(
            "Verified auction year is not 2023"
        )

    if not quotation_sections:
        raise ValueError(
            "No extractable quotation sections"
        )

    return {
        "start": selected[0],
        "end": selected[1],
        "matches": match_count,
        "pages": page_count,
    }


def main():
    rows, fields = read_manifest()

    if "review_notes" not in fields:
        fields.append("review_notes")

    for row in rows:
        row.setdefault("review_notes", "")

    candidates = [
        row for row in rows
        if row["filename_year"] == "2023"
        and row["download_status"] == "failed"
        and row["validation_status"]
        == "recovery_requires_review"
    ][:RECOVERY_LIMIT]

    print("STEP 68 — CROSS-MONTH RECOVERY")
    print("=" * 55)
    print("Selected failed reports:", len(candidates))

    existing_hashes = {
        hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in REPORT_DIR.glob("*.pdf")
    }

    recovered = 0
    duplicates = 0
    unresolved = 0

    with requests.Session() as session:
        for index, row in enumerate(
            candidates, start=1
        ):
            print(
                f"\n[{index}/{len(candidates)}]"
                f" {row['filename']}"
            )

            try:
                content = download_pdf(
                    session, row["source_url"]
                )

                metadata = extract_metadata(content)

                digest = hashlib.sha256(
                    content
                ).hexdigest()

                row["verified_auction_date"] = (
                    metadata["start"]
                )
                row["verified_auction_year"] = "2023"

                if digest in existing_hashes:
                    row["download_status"] = "duplicate"
                    row["validation_status"] = (
                        "duplicate_content"
                    )

                    duplicates += 1
                    print("Duplicate PDF content")

                else:
                    filename = (
                        f"forbes_2023_date_"
                        f"{metadata['start']}_"
                        f"{digest[:12]}.pdf"
                    )

                    output = REPORT_DIR / filename

                    if output.exists():
                        raise FileExistsError(
                            "Destination already exists"
                        )

                    output.write_bytes(content)
                    existing_hashes.add(digest)

                    row["download_status"] = "downloaded"
                    row["validation_status"] = (
                        "date_verified_sale_number_unknown"
                    )

                    recovered += 1
                    print("Saved:", filename)

                row["review_notes"] = (
                    "Cross-month auction date verified "
                    f"from {metadata['matches']} repeated "
                    "SALE OF references. Sale number "
                    "not independently verified."
                )

                print(
                    "Auction start:", metadata["start"]
                )
                print(
                    "Auction end:", metadata["end"]
                )
                print("Pages:", metadata["pages"])
                print("Sale number: UNVERIFIED")

            except Exception as error:
                row["download_status"] = "failed"
                row["validation_status"] = (
                    "recovery_requires_review"
                )
                row["review_notes"] = str(error)

                unresolved += 1
                print("REQUIRES REVIEW:", error)

            finally:
                save_manifest(rows, fields)

                if index < len(candidates):
                    time.sleep(REQUEST_DELAY)

    counts = Counter(
        row["download_status"]
        for row in rows
    )

    print("\nSTEP 68 RECOVERY SUMMARY")
    print("=" * 55)
    print("Recovered:", recovered)
    print("Duplicate PDFs:", duplicates)
    print("Unresolved:", unresolved)
    print("Total downloaded:", counts["downloaded"])
    print("Total duplicates:", counts["duplicate"])
    print("Total failed:", counts["failed"])
    print("Total pending:", counts["pending"])
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
