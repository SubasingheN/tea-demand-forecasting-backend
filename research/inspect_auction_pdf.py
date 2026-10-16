
from pathlib import Path
import pdfplumber


ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    ROOT
    / "data"
    / "external"
    / "forbes_auction_reports"
    / "WEEKLY TEA MARKET REPORT SALE NO 4-2025.pdf"
)

# PDF page numbers, starting at 1.
PAGES_TO_INSPECT = [5, 6, 7, 8]

KEYWORDS = [
    "BOP",
    "BOPF",
    "DUST",
    "PEKOE",
    "QUOTATIONS",
    "LOW GROWN",
    "HIGH GROWN",
    "MEDIUM GROWN",
]


def inspect_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    with pdfplumber.open(PDF_PATH) as pdf:
        print("PDF opened successfully.")
        print("Total pages:", len(pdf.pages))

        for page_number in PAGES_TO_INSPECT:
            page = pdf.pages[page_number - 1]

            print("\n" + "=" * 70)
            print(f"PDF PAGE {page_number}")
            print("=" * 70)

            text = page.extract_text(
                layout=False
            ) or ""

            for line in text.splitlines():
                if any(
                    keyword in line.upper()
                    for keyword in KEYWORDS
                ):
                    print(line)

            tables = page.extract_tables()

            print(
                "\nDetected tables:",
                len(tables)
            )

            for table_number, table in enumerate(
                tables, start=1
            ):
                print(
                    f"\nTable {table_number}"
                )

                for row in table[:15]:
                    print(row)


if __name__ == "__main__":
    inspect_pdf()
