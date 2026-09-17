"""Score externally generated TFT forecasts against exactly matched baselines.

Usage: python research/score_tft.py path/to/tft_predictions.csv
Input columns: origin_week,target_week,horizon_week,target,predicted_1000kg
Predictions must come from a separately trained model using only information
known by each origin week; this checker cannot prove model training provenance.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "outputs"
KEYS = ("origin_week", "target_week", "horizon_week", "target")
REQUIRED = (*KEYS, "predicted_1000kg")


def key(row):
    return tuple(row[k] for k in KEYS)


def score(path):
    with (ROOT / "baseline_matched_8week.csv").open(newline="") as handle:
        baseline = [row for row in csv.DictReader(handle) if row["target"] in {"high_1000kg", "medium_1000kg", "low_1000kg"}]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(k not in reader.fieldnames for k in REQUIRED):
            raise ValueError(f"Expected columns: {', '.join(REQUIRED)}")
        provided = list(reader)
    if not provided:
        raise ValueError("No TFT predictions provided")
    actual = {}
    for row in baseline:
        identifier = key(row)
        if identifier in actual and actual[identifier] != float(row["actual_1000kg"]):
            raise ValueError("Inconsistent baseline actual values")
        actual[identifier] = float(row["actual_1000kg"])
    seen = {}
    for row in provided:
        identifier = key(row)
        if identifier in seen:
            raise ValueError(f"Repeated prediction key: {identifier}")
        value = float(row["predicted_1000kg"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Prediction must be finite and nonnegative: {identifier}")
        seen[identifier] = value
    if seen.keys() != actual.keys():
        raise ValueError(f"Prediction keys differ: {len(actual.keys() - seen.keys())} missing, "
                         f"{len(seen.keys() - actual.keys())} extra")

    grouped = defaultdict(list)
    for row in baseline:
        identifier = key(row)
        grouped[(row["target"], row["method"], row["horizon_week"])].append(
            (float(row["actual_1000kg"]), float(row["predicted_1000kg"])))
    for identifier, prediction in seen.items():
        grouped[(identifier[3], "tft", identifier[2])].append((actual[identifier], prediction))
    results = []
    for (target, method, horizon), values in sorted(grouped.items()):
        results.append({"target": target, "method": method, "horizon_week": int(horizon),
                        "n_origins": len(values),
                        "mae_1000kg": round(sum(abs(y - p) for y, p in values) / len(values), 2),
                        "rmse_1000kg": round((sum((y - p) ** 2 for y, p in values) / len(values)) ** .5, 2)})
    output = {"target_kind": "observed_auction_sales_by_elevation",
              "comparison": "matched eight-week encoder origins, four calendar-week horizons",
              "origin_count_per_target": len(seen) // (3 * 4), "metrics": results,
              "provenance_warning": "Scoring does not verify that TFT was trained without future information."}
    (ROOT / "tft_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python research/score_tft.py path/to/tft_predictions.csv")
    print(json.dumps(score(Path(sys.argv[1])), indent=2))
