"""Score a frozen forecast only against subsequently published auction sales."""

import argparse
import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path


OUTPUTS = Path(__file__).resolve().parent / "outputs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(snapshot_path, calendar_path, source_pdf):
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    issued = datetime.fromisoformat(snapshot["issued_at_utc"])
    if issued.utcoffset() is None:
        raise ValueError("Snapshot issuance must include a UTC offset")
    if snapshot["target_kind"] != "observed_auction_sales_by_elevation":
        raise ValueError("Unexpected target kind")
    if snapshot["unit"] != "thousand kilograms":
        raise ValueError("Unexpected unit")
    if date.fromisoformat(snapshot["last_source_sale_date"]) > issued.date():
        raise ValueError("Snapshot source sale postdates issuance")

    frozen_pdf_hash = snapshot["sha256"].get(source_pdf.name)
    if frozen_pdf_hash is None:
        raise ValueError("The supplied source PDF filename is absent from the snapshot")
    source_updated = sha256(source_pdf) != frozen_pdf_hash

    with calendar_path.open(newline="", encoding="utf-8-sig") as handle:
        calendar = {row["week_start"]: row for row in csv.DictReader(handle)}

    keys = set()
    scored = []
    pending = []
    for prediction in snapshot["forecasts"]:
        target = prediction["target"]
        week = prediction["target_week_start"]
        key = (target, week)
        if key in keys:
            raise ValueError(f"Duplicate prediction: {key}")
        keys.add(key)
        if target not in ("high_1000kg", "medium_1000kg", "low_1000kg"):
            raise ValueError(f"Unexpected elevation: {target}")
        if date.fromisoformat(week) <= issued.date():
            raise ValueError(f"Target week was already underway at issuance: {week}")
        p10 = float(prediction["tft_p10_1000kg"])
        p50 = float(prediction["tft_p50_1000kg"])
        p90 = float(prediction["tft_p90_1000kg"])
        if not p10 <= p50 <= p90:
            raise ValueError(f"Invalid prediction interval: {key}")

        row = calendar.get(week)
        if row is None or row["sale_count"] != "1" or not row[target]:
            pending.append({"target": target, "target_week_start": week,
                            "reason": "No single observed auction sale in this week"})
            continue
        source_date = date.fromisoformat(row["source_sale_dates"])
        if source_date <= issued.date() or source_date < date.fromisoformat(week):
            raise ValueError(f"Observed sale date is inconsistent with snapshot: {key}")
        if not source_updated:
            pending.append({"target": target, "target_week_start": week,
                            "reason": "Published source PDF has not changed since issuance"})
            continue
        actual = float(row[target])
        scored.append({
            "target": target, "target_week_start": week,
            "source_sale_date": source_date.isoformat(),
            "horizon_week": int(prediction["horizon_week"]),
            "actual_1000kg": actual,
            "tft_abs_error_1000kg": round(abs(actual - p50), 4),
            "last_abs_error_1000kg": round(abs(actual - float(prediction["last_baseline_1000kg"])), 4),
            "mean4_abs_error_1000kg": round(abs(actual - float(prediction["mean4_baseline_1000kg"])), 4),
            "inside_tft_10_90": p10 <= actual <= p90,
        })

    summary = {}
    for target in sorted({p["target"] for p in snapshot["forecasts"]}):
        rows = [r for r in scored if r["target"] == target]
        if rows:
            summary[target] = {
                "n_weeks": len(rows),
                "mae_1000kg": {
                    method: round(sum(r[f"{method}_abs_error_1000kg"] for r in rows) / len(rows), 2)
                    for method in ("tft", "last", "mean4")
                },
                "tft_interval_coverage": round(sum(r["inside_tft_10_90"] for r in rows) / len(rows), 3),
            }
    return {
        "snapshot": snapshot_path.name,
        "source_pdf_sha256": sha256(source_pdf),
        "source_pdf_changed_since_snapshot": source_updated,
        "scored_predictions": scored,
        "pending_predictions": pending,
        "by_elevation": summary,
        "status": "complete" if not pending else "partial_or_waiting",
        "caution": "Two target weeks share one forecast origin; results are descriptive, not independent trials.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--calendar", type=Path, default=OUTPUTS / "auction_calendar_weeks.csv")
    parser.add_argument("--source-pdf", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.snapshot, args.calendar, args.source_pdf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
