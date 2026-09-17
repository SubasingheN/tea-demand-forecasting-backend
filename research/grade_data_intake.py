"""Audit real dated five-grade orders or sales before TFT training.

Input CSV: event_date,tea_type,quantity_kg,target_kind,source_document
target_kind is customer_order or auction_sale. Several rows in one week and
grade are summed. The output is never substituted for the prototype dataset.
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


GRADES = ("BOP", "BOPF", "Dust", "OP", "Pekoe")
TARGETS = ("customer_order", "auction_sale")
REQUIRED = ("event_date", "tea_type", "quantity_kg", "target_kind", "source_document")


def audit(source):
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(REQUIRED) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns: {sorted(missing_columns)}")
        weekly = defaultdict(float)
        sources = defaultdict(set)
        kinds, invalid = set(), []
        for row_number, row in enumerate(reader, 2):
            try:
                event = date.fromisoformat(row["event_date"].strip())
                grade = row["tea_type"].strip()
                kind = row["target_kind"].strip()
                document = row["source_document"].strip()
                quantity = float(row["quantity_kg"])
                if grade not in GRADES or kind not in TARGETS:
                    raise ValueError("unknown grade or target_kind")
                if not document or not 0 <= quantity < float("inf"):
                    raise ValueError("missing source or invalid quantity")
                kinds.add(kind)
                monday = event - timedelta(days=event.weekday())
                key = (monday.isoformat(), grade)
                weekly[key] += quantity
                sources[key].add(document)
            except (ValueError, TypeError, KeyError) as error:
                invalid.append({"row": row_number, "reason": str(error)})
    if invalid or len(kinds) != 1 or not weekly:
        raise ValueError(json.dumps({
            "invalid_rows": invalid[:25], "target_kinds": sorted(kinds),
            "message": "Use exactly one target definition and correct invalid rows.",
        }, indent=2))
    all_weeks = sorted({week for week, _ in weekly})
    weeks = [date.fromisoformat(x) for x in all_weeks]
    expected = [weeks[0] + timedelta(weeks=i)
                for i in range((weeks[-1] - weeks[0]).days // 7 + 1)]
    missing = {grade: [day.isoformat() for day in expected
                       if (day.isoformat(), grade) not in weekly]
               for grade in GRADES}
    cleaned = [
        {"week_start_date": week, "tea_type": grade,
         "observed_quantity_kg": round(weekly[(week, grade)], 3),
         "target_kind": next(iter(kinds)),
         "source_documents": "; ".join(sorted(sources[(week, grade)]))}
        for week, grade in sorted(weekly)
    ]
    summary = {
        "target_kind": next(iter(kinds)),
        "target_label": ("Customer orders (kg)" if "customer_order" in kinds
                         else "Observed auction sales (kg); demand proxy"),
        "first_week": all_weeks[0], "last_week": all_weeks[-1],
        "weeks_with_any_record": len(all_weeks),
        "rows_by_grade": {g: sum(key[1] == g for key in weekly) for g in GRADES},
        "missing_weeks_by_grade": missing,
        "ready_for_research_training": all(len(missing[g]) == 0 and
            sum(key[1] == g for key in weekly) >= 104 for g in GRADES),
        "readiness_note": "104 observed weeks per grade is an initial screening threshold, not a guarantee that TFT will outperform simpler methods.",
    }
    return cleaned, summary


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python research/grade_data_intake.py source.csv")
    cleaned, summary = audit(Path(sys.argv[1]))
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "grade_weekly_observed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cleaned[0])
        writer.writeheader()
        writer.writerows(cleaned)
    (output / "grade_data_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
