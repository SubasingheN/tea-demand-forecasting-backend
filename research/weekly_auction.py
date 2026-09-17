"""Build and evaluate an observed weekly auction series without inventing sales.

Usage: python research/weekly_auction.py
Input: research/outputs/auction_sales_2024_2026.csv from auction_audit.py.
"""

import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "auction_sales_2024_2026.csv"
OUTPUT = ROOT / "outputs"
TARGETS = ("high_1000kg", "medium_1000kg", "low_1000kg", "total_1000kg")


def make_weeks(sales):
    grouped = defaultdict(list)
    for sale in sales:
        day = date.fromisoformat(sale["sale_date"])
        grouped[day - timedelta(days=day.weekday())].append(sale)
    start, end = min(grouped), max(grouped)
    weeks = []
    week = start
    while week <= end:
        entries = grouped.get(week, [])
        row = {"week_start": week.isoformat(), "sale_count": len(entries)}
        for column in TARGETS:
            row[column] = sum(int(e[column]) for e in entries) if entries else None
        row["source_sale_dates"] = ";".join(e["sale_date"] for e in entries)
        weeks.append(row)
        week += timedelta(days=7)
    return weeks


def backtest(weeks, cutoff=date(2026, 1, 1)):
    """Origins precede their four target weeks; missing targets are excluded.

    A forecast is issued after an observed origin week. Training input consists
    of four consecutive observed calendar weeks. All four future weeks must
    have observations for inclusion. This does not assume zero demand in
    weeks without an auction.
    """
    predictions = []
    excluded = defaultdict(int)
    for i in range(3, len(weeks) - 4):
        history, future = weeks[i - 3:i + 1], weeks[i + 1:i + 5]
        if date.fromisoformat(future[0]["week_start"]) < cutoff:
            continue
        if any(w["sale_count"] == 0 for w in history):
            excluded["missing_history"] += 1
            continue
        if any(w["sale_count"] == 0 for w in future):
            excluded["missing_future"] += 1
            continue
        for target in TARGETS:
            last = history[-1][target]
            mean4 = sum(w[target] for w in history) / 4
            for horizon, actual in enumerate(future, 1):
                for method, predicted in (("last", last), ("mean4", mean4)):
                    predictions.append({
                        "origin_week": weeks[i]["week_start"],
                        "target_week": actual["week_start"],
                        "horizon_week": horizon,
                        "target": target,
                        "method": method,
                        "actual_1000kg": actual[target],
                        "predicted_1000kg": round(predicted, 3),
                        "absolute_error_1000kg": round(abs(actual[target] - predicted), 3),
                    })
    metrics = {}
    for target in TARGETS:
        metrics[target] = {}
        for method in ("last", "mean4"):
            samples = [p for p in predictions if p["target"] == target and p["method"] == method]
            metrics[target][method] = {
                str(h): {
                    "n_origins": len(selected := [p for p in samples if p["horizon_week"] == h]),
                    "mae_1000kg": round(sum(p["absolute_error_1000kg"] for p in selected) / len(selected), 2) if selected else None,
                }
                for h in range(1, 5)
            }
    return predictions, {"cutoff": cutoff.isoformat(), "excluded_origins": dict(excluded),
                         "metrics": metrics, "eligible_origins": len(predictions) // (len(TARGETS) * 2 * 4)}


def main():
    with INPUT.open(newline="") as handle:
        sales = list(csv.DictReader(handle))
    if not sales or len({r["sale_date"] for r in sales}) != len(sales):
        raise ValueError("Sales must contain distinct dated source observations")
    weeks = make_weeks(sales)
    predictions, summary = backtest(weeks)
    summary.update({
        "unit": "thousand kilograms", "target_kind": "auction_sales_by_elevation",
        "weeks": len(weeks), "observed_weeks": sum(w["sale_count"] > 0 for w in weeks),
        "missing_weeks": [w["week_start"] for w in weeks if w["sale_count"] == 0],
        "note": "Missing weeks have blank targets; they are not zero sales or imputed demand. TFT is not evaluated here.",
    })
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "auction_calendar_weeks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=weeks[0].keys())
        writer.writeheader()
        writer.writerows(weeks)
    with (OUTPUT / "weekly_baseline_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("origin_week", "target_week", "horizon_week", "target", "method", "actual_1000kg", "predicted_1000kg", "absolute_error_1000kg"))
        writer.writeheader()
        writer.writerows(predictions)
    (OUTPUT / "weekly_baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
