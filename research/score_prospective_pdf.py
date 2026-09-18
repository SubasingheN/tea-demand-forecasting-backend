"""Score the frozen September forecasts directly from a later 2026 auction PDF.

This command does not overwrite the original 134-sale research dataset or
the retrospective evaluation outputs.
"""

import argparse
import csv
import json
import tempfile
from pathlib import Path

from auction_audit import extract_year
from score_prospective import OUTPUTS, score
from weekly_auction import make_weeks


SNAPSHOT = (
    OUTPUTS
    / "prospective"
    / "auction_forecast_20260917T222937Z.json"
)


def write_rows(path, rows):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "source_pdf",
        type=Path,
        help="The latest published 2026 weekly auction PDF",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=SNAPSHOT,
    )
    args = parser.parse_args()

    if not args.source_pdf.is_file():
        parser.error(
            f"PDF does not exist: {args.source_pdf}"
        )

    sales, issues = extract_year(
        args.source_pdf,
        2026,
    )
    if not sales:
        raise ValueError(
            "The PDF contains no recognizable 2026 sale records"
        )

    dates = [sale["sale_date"] for sale in sales]
    if len(set(dates)) != len(dates):
        raise ValueError(
            "Duplicate dated sales in the updated PDF"
        )

    if any(
        "cannot parse sale row" in issue
        or "invalid date" in issue
        or "unexpected year" in issue
        or "rounded component sum differs" in issue
        for issue in issues
    ):
        raise ValueError(
            "Resolve auction extraction issues before scoring: "
            f"{issues}"
        )

    weeks = make_weeks(sales)

    with tempfile.TemporaryDirectory(
        prefix="auction-prospective-"
    ) as temp:
        scratch = Path(temp)
        calendar = scratch / "calendar.csv"
        sale_records = scratch / "sales.csv"

        write_rows(calendar, weeks)
        write_rows(sale_records, sales)

        result = score(
            args.snapshot,
            calendar,
            args.source_pdf,
            sale_records,
        )

    result["latest_source_sale_date"] = max(dates)
    result["extraction_corrections"] = issues
    result["retrospective_files_overwritten"] = False

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()