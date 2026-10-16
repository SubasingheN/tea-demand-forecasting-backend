
from pathlib import Path
from collections import Counter
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from batch_extract_historical_dust import (
    REPORT_DIR,
    LAYOUTS,
    identify_year,
    extract_report,
)

OUTPUT_DIR = (
    ROOT / "data" / "external" / "validated_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CANDIDATE_CSV = (
    OUTPUT_DIR / "dust_candidate_quotations.csv"
)

REVIEW_CSV = (
    OUTPUT_DIR / "dust_reports_requiring_review.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR / "dust_extraction_summary.json"
)


def main():
    candidates = []
    reviews = []
    year_stats = {}

    reports = sorted(
        REPORT_DIR.glob("*.pdf")
    )

    for year in LAYOUTS:
        year_stats[year] = {
            "reports": 0,
            "candidate_quotations": 0,
            "complete_reports": 0,
            "review_reports": 0,
        }

    print(
        "STEP 93 — HISTORICAL DUST AUDIT EXPORT"
    )
    print("=" * 65)

    for index, path in enumerate(reports, 1):
        year = identify_year(path.name)

        if year not in LAYOUTS:
            reviews.append({
                "source_file": path.name,
                "year": year or "",
                "reason": "Unknown report year",
            })
            continue

        stats = year_stats[year]
        stats["reports"] += 1

        try:
            audit = extract_report(
                path,
                LAYOUTS[year],
            )

            accepted = len(audit["results"])

            complete = (
                accepted == 21
                and audit["detected_categories"] == 7
                and not audit["errors"]
                and not audit["missing_categories"]
            )

            stats["candidate_quotations"] += accepted

            if complete:
                stats["complete_reports"] += 1
            else:
                stats["review_reports"] += 1

            for result in audit["results"]:
                candidates.append({
                    "source_file": path.name,
                    "source_year": year,
                    "source_page": result["source_page"],
                    "grade": result["grade"],
                    "quality_category": result["category"],
                    "elevation": result["elevation"],
                    "price_min_lkr": result["min_lkr"],
                    "price_max_lkr": result["max_lkr"],
                    "extraction_layout": result["layout"],
                    "report_complete": complete,
                    "verification_status":
                        "candidate_unverified",
                })

            if not complete:
                reasons = (
                    audit["errors"]
                    + [
                        f"Missing category: {category}"
                        for category
                        in audit["missing_categories"]
                    ]
                )

                reviews.append({
                    "source_file": path.name,
                    "year": year,
                    "reason": " | ".join(reasons),
                })

        except Exception as error:
            stats["review_reports"] += 1

            reviews.append({
                "source_file": path.name,
                "year": year,
                "reason": f"Processing error: {error}",
            })

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(reports)} PDFs"
            )

    candidate_fields = [
        "source_file",
        "source_year",
        "source_page",
        "grade",
        "quality_category",
        "elevation",
        "price_min_lkr",
        "price_max_lkr",
        "extraction_layout",
        "report_complete",
        "verification_status",
    ]

    with CANDIDATE_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=candidate_fields
        )
        writer.writeheader()
        writer.writerows(candidates)

    with REVIEW_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_file",
                "year",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(reviews)

    summary = {
        "reports_examined": len(reports),
        "candidate_quotations": len(candidates),
        "review_reports": len(reviews),
        "year_stats": year_stats,
        "verification_status":
            "candidate_unverified",
        "note": (
            "These are historical quotation candidates. "
            "They are not independently verified, "
            "not demand observations, and are not "
            "weighted-average auction prices."
        ),
    }

    with SUMMARY_JSON.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nEXPORT SUMMARY")
    print("=" * 65)

    print(
        "Candidate quotations:",
        len(candidates),
    )
    print(
        "Reports requiring review:",
        len(reviews),
    )

    print("\nCreated files:")
    print(CANDIDATE_CSV)
    print(REVIEW_CSV)
    print(SUMMARY_JSON)

    print(
        "\nFactory dataset and TFT model: unchanged"
    )


if __name__ == "__main__":
    main()
