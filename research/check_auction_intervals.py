"""Check empirical coverage of TFT 10th–90th percentile auction forecasts."""

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

from train_auction_tft import ROOT, dataset, make_frames, read_windows


def main():
    rows = read_windows()
    frames = make_frames(rows)
    reference = dataset(frames["train"])
    checkpoint = ROOT / "checkpoints" / "auction-tft.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    records = []
    for row in rows:
        if row["split"] != "test":
            continue
        window_id = f"test:{row['target']}:{row['origin_week']}"
        frame = frames["test"].loc[frames["test"]["window_id"] == window_id].copy()
        sample = dataset(frame, reference, predict=True)
        if len(sample) != 1:
            raise ValueError(f"Expected one test sample for {window_id}")
        with torch.no_grad():
            values = model.predict(sample.to_dataloader(train=False, batch_size=1, num_workers=0),
                                   mode="quantiles")
        values = values.detach().cpu().numpy()
        if values.shape != (1, 4, 3):
            raise ValueError(f"Expected [one sample, four weeks, three quantiles], got {values.shape}")
        for h in range(1, 5):
            low, median, high = (float(x) for x in values[0, h - 1])
            if not np.isfinite([low, median, high]).all():
                raise ValueError(f"Nonfinite quantile for {window_id}, week {h}")
            actual = float(row[f"future_week_{h}_1000kg"])
            records.append({
                "origin_week": row["origin_week"],
                "target_week": row[f"future_week_{h}_date"],
                "target": row["target"], "horizon_week": h,
                "actual_1000kg": actual,
                "p10_1000kg": low, "p50_1000kg": median, "p90_1000kg": high,
                "covered": int(low <= actual <= high),
                "width_1000kg": high - low,
            })
    details_path = ROOT / "tft_interval_predictions.csv"
    with details_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["target"], record["horizon_week"])].append(record)
    summary = {
        "interval": "model 10th–90th percentiles (nominal 80%)",
        "unit": "thousand kg of observed auction sales",
        "note": "Nine overlapping test origins per elevation. Empirical coverage is descriptive, not a calibrated confidence guarantee.",
        "results": [{
            "target": target, "horizon_week": horizon, "n": len(group),
            "coverage": round(sum(r["covered"] for r in group) / len(group), 4),
            "mean_width_1000kg": round(sum(r["width_1000kg"] for r in group) / len(group), 2),
        } for (target, horizon), group in sorted(grouped.items())],
    }
    (ROOT / "tft_interval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
