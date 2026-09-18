"""Measure local TFT response to a 5% change in each past sale.

Retrospective model-behavior diagnostic only. These 2026 origins were already
inspected during development; the diagnostic is not an independent test or
a causal explanation of auction sales.
"""

import json
from collections import defaultdict

import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

from forecast_auction_tft import predict
from train_auction_tft import ROOT, dataset, make_frames, read_windows


TARGETS = (
    "high_1000kg",
    "medium_1000kg",
    "low_1000kg",
)
CHANGE = 0.05


def median_forecast(model, reference, history, target):
    result = predict(
        model,
        reference,
        history,
        target,
        placeholder=0.0,
    )
    if result.shape != (4, 3) or not np.isfinite(result).all():
        raise ValueError(
            f"Unexpected forecast shape or values for {target}"
        )
    return result[:, 1]


def main():
    torch.set_num_threads(2)
    rows = read_windows()
    reference = dataset(make_frames(rows)["train"])

    checkpoint = ROOT / "checkpoints" / "auction-tft.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )
    model.eval()

    changes = defaultdict(list)
    origin_counts = defaultdict(set)

    for row in rows:
        if (
            row["split"] != "test"
            or row["target"] not in TARGETS
        ):
            continue

        target = row["target"]
        origin = row["origin_week"]
        origin_counts[target].add(origin)

        # predict() reads the current target from each history week.
        history = [
            {
                target: float(
                    row[
                        f"history_week_{week}_1000kg"
                    ]
                )
            }
            for week in range(1, 9)
        ]
        original = median_forecast(
            model,
            reference,
            history,
            target,
        )

        for week in range(1, 9):
            changed_history = [
                entry.copy()
                for entry in history
            ]
            changed_history[week - 1][target] *= (
                1 + CHANGE
            )

            adjusted = median_forecast(
                model,
                reference,
                changed_history,
                target,
            )

            for horizon in range(1, 5):
                tft_delta = float(
                    adjusted[horizon - 1]
                    - original[horizon - 1]
                )

                # The simple baselines repeat their values
                # at every future horizon.
                sale = history[week - 1][target]
                last_delta = (
                    CHANGE * sale
                    if week == 8
                    else 0.0
                )
                mean4_delta = (
                    CHANGE * sale / 4
                    if week >= 5
                    else 0.0
                )

                changes[
                    (target, week, horizon)
                ].append(
                    (
                        tft_delta,
                        last_delta,
                        mean4_delta,
                    )
                )

    expected = (
        len(next(iter(origin_counts.values())))
        if origin_counts
        else 0
    )
    if (
        expected < 1
        or any(
            len(origins) != expected
            for origins in origin_counts.values()
        )
        or set(origin_counts) != set(TARGETS)
    ):
        raise ValueError(
            f"Missing or unequal test origins: "
            f"{dict(origin_counts)}"
        )

    summary = []
    for target in TARGETS:
        for week in range(1, 9):
            for horizon in range(1, 5):
                values = changes[
                    (target, week, horizon)
                ]
                if len(values) != expected:
                    raise ValueError(
                        "Missing perturbation: "
                        f"{target}, {week}, {horizon}"
                    )

                summary.append({
                    "target": target,
                    "history_week": week,
                    "horizon_week": horizon,
                    "n_origins": expected,
                    "mean_abs_tft_change_1000kg": round(
                        sum(
                            abs(value[0])
                            for value in values
                        )
                        / expected,
                        3,
                    ),
                    "mean_abs_last_change_1000kg": round(
                        sum(
                            abs(value[1])
                            for value in values
                        )
                        / expected,
                        3,
                    ),
                    "mean_abs_mean4_change_1000kg": round(
                        sum(
                            abs(value[2])
                            for value in values
                        )
                        / expected,
                        3,
                    ),
                    "tft_positive_response_count": sum(
                        value[0] > 0
                        for value in values
                    ),
                    "tft_negative_response_count": sum(
                        value[0] < 0
                        for value in values
                    ),
                })

    result = {
        "target_kind": (
            "observed_auction_sales_by_elevation"
        ),
        "history_weeks": 8,
        "forecast_weeks": 4,
        "test_origins_per_elevation": expected,
        "intervention": (
            "Raise one observed history week's sale by 5%; "
            "keep other history fixed"
        ),
        "scope": (
            "Model sensitivity, not causes of sales "
            "or effects of changing supply"
        ),
        "caution": (
            "Test origins overlap and were previously "
            "inspected; results are descriptive."
        ),
        "results": summary,
    }

    path = ROOT / "tft_responsiveness.json"
    path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    overview = {
        target: round(
            sum(
                item[
                    "mean_abs_tft_change_1000kg"
                ]
                for item in summary
                if item["target"] == target
            )
            / 32,
            3,
        )
        for target in TARGETS
    }

    print(
        json.dumps(
            {
                "output": str(path),
                "test_origins_per_elevation": expected,
                "mean_abs_tft_response_across_8_lags_and_4_horizons_1000kg": overview,
                "note": result["caution"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()