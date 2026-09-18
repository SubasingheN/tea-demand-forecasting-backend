"""Generate a dated four-week auction forecast from the real-data TFT checkpoint.

Uses the latest eight consecutive observed calendar weeks. Future target
placeholders are required by TimeSeriesDataSet and checked for leakage.
This command does not retrain the model or change the held-out evaluation.
"""

import csv
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from train_auction_tft import ROOT, dataset, make_frames, read_windows


TARGETS = (
    "high_1000kg",
    "medium_1000kg",
    "low_1000kg",
)


def latest_history():
    with (ROOT / "auction_calendar_weeks.csv").open(newline="") as handle:
        weeks = list(csv.DictReader(handle))

    if len(weeks) < 8:
        raise ValueError("Need at least eight calendar weeks")

    history = weeks[-8:]

    if any(int(row["sale_count"]) == 0 for row in history):
        raise ValueError(
            "Latest eight calendar weeks contain a missing auction observation"
        )

    dates = [
        date.fromisoformat(row["week_start"])
        for row in history
    ]

    if any(
        b - a != timedelta(weeks=1)
        for a, b in zip(dates, dates[1:])
    ):
        raise ValueError(
            "History is not eight consecutive calendar weeks"
        )

    return history, dates[-1]


def predict(model, reference, history, target, placeholder):
    frame = pd.DataFrame([
        {
            "window_id": "live",
            "elevation": target.replace("_1000kg", ""),
            "time_idx": index,
            "sale_1000kg": (
                float(history[index][target])
                if index < 8
                else placeholder
            ),
        }
        for index in range(12)
    ])

    sample = dataset(frame, reference, predict=True)

    if len(sample) != 1:
        raise ValueError(
            "Expected one four-week prediction window"
        )

    with torch.no_grad():
        output = model.predict(
            sample.to_dataloader(
                train=False,
                batch_size=1,
                num_workers=0,
            ),
            mode="quantiles",
        )

    values = output.detach().cpu().numpy()

    if values.shape != (1, 4, 3) or not np.isfinite(values).all():
        raise ValueError(
            f"Invalid quantile output: {values.shape}"
        )

    return values[0]


def main():
    history, origin = latest_history()
    reference = dataset(
        make_frames(read_windows())["train"]
    )

    checkpoint = (
        ROOT / "checkpoints" / "auction-tft.ckpt"
    )

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )
    model.eval()

    forecasts = []

    for target in TARGETS:
        values = predict(
            model,
            reference,
            history,
            target,
            placeholder=0.0,
        )

        independent = predict(
            model,
            reference,
            history,
            target,
            placeholder=12345.0,
        )

        if not np.allclose(
            values,
            independent,
            rtol=1e-5,
            atol=1e-3,
        ):
            raise ValueError(
                f"Future target placeholders changed "
                f"{target} predictions"
            )

        for horizon, (lower, median, upper) in enumerate(
            values, 1
        ):
            forecasts.append({
                "target": target,
                "week_start": (
                    origin + timedelta(weeks=horizon)
                ).isoformat(),
                "horizon_week": horizon,
                "p10_1000kg": round(float(lower), 2),
                "p50_1000kg": round(float(median), 2),
                "p90_1000kg": round(float(upper), 2),
            })

    result = {
        "target_kind": (
            "observed_auction_sales_by_elevation"
        ),
        "unit": "thousand kg",
        "origin_week_start": origin.isoformat(),
        "last_source_sale_date": (
            history[-1]["source_sale_dates"]
        ),
        "encoder_observed_weeks": 8,
        "scope": (
            "Research forecast from the last available "
            "eight observed calendar weeks"
        ),
        "caution": (
            "Historical-source forecast until the auction "
            "table is refreshed. Nominal 10th–90th percentile "
            "ranges are poorly calibrated for medium-grown "
            "sales. No grade-level customer demand is forecast here."
        ),
        "forecasts": forecasts,
    }

    (ROOT / "latest_tft_forecast.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()