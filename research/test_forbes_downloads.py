
from pathlib import Path
from urllib.parse import urlsplit
import csv
import hashlib
import io
import time

import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]

EXTERNAL = ROOT / "data" / "external"
MANIFEST = EXTERNAL / "forbes_download_manifest.csv"

OUTPUT_DIR = (
    EXTERNAL
    / "forbes_auction_reports"
    / "download_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = ["2021", "2023", "2024"]

HEADERS = {
    "User-Agent": "AcademicResearch/1.0"
}


def download_report(session, row, year):
    url = row["source_url"]

    response = session.get(
        url,
        timeout=60,
        headers=HEADERS,
    )

    response.raise_for_status()

    content = response.content

    if not content.startswith(b"%PDF-"):
        raise ValueError("Response is not a PDF")

    if len(content) > 30 * 1024 * 1024:
        raise ValueError("PDF exceeds 30 MB")

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = len(pdf.pages)

        if pages < 1:
            raise ValueError("PDF has no pages")

        cover = (
            pdf.pages[0].extract_text() or ""
        ).replace("\n", " ")

    sha256 = hashlib.sha256(content).hexdigest()

    output = OUTPUT_DIR / f"forbes_test_{year}.pdf"

    output.write_bytes(content)

    print(f"\nYear candidate: {year}")
    print("HTTP status:", response.status_code)
    print("PDF pages:", pages)
    print("Size (KB):", round(len(content) / 1024, 1))
    print("SHA-256:", sha256)
    print("Cover:", cover[:250])
    print("Saved:", output)


def main():
    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    session = requests.Session()

    for year in YEARS:
        candidates = [
            row for row in rows
            if row["filename_year"] == year
        ]

        if not candidates:
            print(f"No candidate for {year}")
            continue

        # Test the first link for each year.
        row = candidates[0]

        try:
            download_report(session, row, year)
        except Exception as error:
            print(
                f"\nDownload failed for {year}:",
                error
            )

        # Conservative delay between requests.
        time.sleep(3)

    print("\nTHREE-YEAR DOWNLOAD TEST FINISHED")


if __name__ == "__main__":
    main()
