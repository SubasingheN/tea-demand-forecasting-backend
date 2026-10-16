
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup


URL = (
    "https://www.forbestea.com/"
    "statistics-tea-market-reports"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; AcademicResearch/1.0)"
    )
}


def inspect_links():
    response = requests.get(
        URL,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    print("Website status:", response.status_code)
    print("Final URL:", response.url)

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    all_links = soup.find_all(
        "a",
        href=True,
    )

    print("Total HTML links:", len(all_links))

    pdf_links = []

    for link in all_links:
        href = link.get("href", "").strip()

        absolute_url = urljoin(
            response.url,
            href,
        )

        parsed = urlparse(absolute_url)

        if not parsed.path.lower().endswith(".pdf"):
            continue

        if parsed.netloc.lower().removeprefix(
            "www."
        ) != "forbestea.com":
            continue

        pdf_links.append({
            "text": link.get_text(
                " ", strip=True
            ),
            "url": absolute_url,
        })

    unique_links = {
        item["url"]: item
        for item in pdf_links
    }

    print(
        "\nUnique PDF links found:",
        len(unique_links),
    )

    for index, item in enumerate(
        unique_links.values(),
        start=1,
    ):
        print(f"\n{index}. {item['text']}")
        print(item["url"])


if __name__ == "__main__":
    inspect_links()
