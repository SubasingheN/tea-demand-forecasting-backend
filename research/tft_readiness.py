"""Audit honest four-calendar-week TFT windows before fitting a model.

Usage: python research/tft_readiness.py
Reads auction_calendar_weeks.csv; does not train a model or impute missing sales.
"""

import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs"
INPUT = ROOT / "auction_calendar_weeks.csv"
LENGTHS = (4, 8, 12, 16, 24, 32, 52)
VALIDATION_START = date(2025, 7, 1)
TEST_START = date(2026, 1, 1)


def audit(weeks):
    results = {}
    for encoder in LENGTHS:
        windows = {"train": [], "validation": [], "test": []}
        for origin in range(encoder - 1, len(weeks) - 4):
            segment = weeks[origin - encoder + 1:origin + 5]
            if any(int(w["sale_count"]) == 0 for w in segment):
                continue
            first_future = date.fromisoformat(weeks[origin + 1]["week_start"])
            last_future = date.fromisoformat(weeks[origin + 4]["week_start"])
            if last_future < VALIDATION_START:
                bucket = "train"
            elif first_future >= VALIDATION_START and last_future < TEST_START:
                bucket = "validation"
            elif first_future >= TEST_START:
                bucket = "test"
            else:
                continue  # Exclude four-week horizons crossing either boundary.
            windows[bucket].append(weeks[origin]["week_start"])
        results[str(encoder)] = {
            name: {"origins": len(origins), "first_origin": origins[0] if origins else None,
                   "last_origin": origins[-1] if origins else None}
            for name, origins in windows.items()
        }
    return {
        "target": "observed weekly auction sales by elevation, in thousand kg",
        "horizon": "four consecutive calendar weeks with observed auction sales",
        "evaluation_cutoffs": {"validation": VALIDATION_START.isoformat(),
                               "test": TEST_START.isoformat()},
        "counts_are_forecast_origins_per_elevation": True,
        "overlapping_windows_are_not_independent": True,
        "encoder_lengths": results,
        "conclusion": "The existing 52-week TFT encoder has no eligible windows. An 8-week encoder permits a limited experiment; performance must be measured before making any model claim.",
    }


def main():
    with INPUT.open(newline="") as handle:
        weeks = list(csv.DictReader(handle))
    summary = audit(weeks)
    samples = []
    for origin in range(7, len(weeks) - 4):
        segment = weeks[origin - 7:origin + 5]
        if any(int(w["sale_count"]) == 0 for w in segment):
            continue
        start = date.fromisoformat(weeks[origin + 1]["week_start"])
        end = date.fromisoformat(weeks[origin + 4]["week_start"])
        split = ("train" if end < VALIDATION_START else
                 "validation" if start >= VALIDATION_START and end < TEST_START else
                 "test" if start >= TEST_START else None)
        if split is None:
            continue
        for target in ("high_1000kg", "medium_1000kg", "low_1000kg"):
            samples.append({
                "split": split, "origin_week": weeks[origin]["week_start"],
                "target": target,
                **{f"history_week_{j}_1000kg": weeks[origin - 8 + j][target]
                   for j in range(1, 9)},
                **{f"future_week_{j}_date": weeks[origin + j]["week_start"]
                   for j in range(1, 5)},
                **{f"future_week_{j}_1000kg": weeks[origin + j][target]
                   for j in range(1, 5)},
            })
    with (ROOT / "tft_8week_windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)
    summary["window_file"] = "tft_8week_windows.csv"
    with (ROOT / "weekly_baseline_predictions.csv").open(newline="") as handle:
        baseline = list(csv.DictReader(handle))
    eligible = set()
    for index in range(7, len(weeks) - 4):
        if (date.fromisoformat(weeks[index + 1]["week_start"]) >= TEST_START
                and all(int(w["sale_count"]) > 0 for w in weeks[index - 7:index + 5])):
            eligible.add(weeks[index]["week_start"])
    matched = [p for p in baseline if p["origin_week"] in eligible]
    with (ROOT / "baseline_matched_8week.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=baseline[0].keys())
        writer.writeheader()
        writer.writerows(matched)
    if len(eligible) != summary["encoder_lengths"]["8"]["test"]["origins"]:
        raise ValueError("Matched baseline and TFT readiness windows differ")
    summary["matched_baseline_file"] = "baseline_matched_8week.csv"
    (ROOT / "tft_readiness.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
