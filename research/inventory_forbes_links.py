
from pathlib import Path
from collections import Counter
from urllib.parse import urljoin, urlparse, unquote
import csv
import re

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    ROOT / "data" / "external"
    / "forbes_historical_pdf_links.csv"
)

SOURCE_URL = (
    "https://www.forbestea.com/"
    "statistics-tea-market-reports"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 AcademicResearch/1.0"
}


def extract_year(filename):
    years = re.findall(
        r"\b20(?:21|22|23|24|25|26)\b",
        filename
    )

    return years[-1] if years else "Unknown"


def main():
    response = requests.get(
        SOURCE_URL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    records = []
    seen_urls = set()

    for link in soup.find_all("a", href=True):
        url = urljoin(
            response.url,
            link["href"].strip()
        )

        parsed = urlparse(url)

        if not parsed.path.lower().endswith(".pdf"):
            continue

        if parsed.netloc.lower().removeprefix(
            "www."
        ) != "forbestea.com":
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        filename = unquote(
            parsed.path.split("/")[-1]
        )

        year = extract_year(filename)

        records.append({
            "filename": filename,
            "filename_year": year,
            "source_url": url,
            "verified_auction_year": "",
            "download_status": "not_downloaded",
            "validation_status": "pending",
        })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "filename",
        "filename_year",
        "source_url",
        "verified_auction_year",
        "download_status",
        "validation_status",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(records)

    counts = Counter(
        record["filename_year"]
        for record in records
    )

    print("LINK INVENTORY COMPLETED")
    print("=" * 50)
    print("Total unique PDF links:", len(records))

    print("\nPDF LINKS BY FILENAME YEAR")

    for year in [
        "2021", "2022", "2023",
        "2024", "2025", "2026", "Unknown"
    ]:
        print(f"{year}: {counts.get(year, 0)}")

    print("\nHistorical target years:")

    for year in ["2021", "2022", "2023", "2024"]:
        print(
            f"{year}: {counts.get(year, 0)} candidate links"
        )

    print("\nOutput:", OUTPUT_FILE)

    print(
        "\nNote: These are filename-based counts."
        "\nAuction dates require PDF verification."
    )


if __name__ == "__main__":
    main()
