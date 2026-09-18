"""Measure local input sensitivity of one dated TFT auction forecast.

Each history week is increased by 5% in a separate model run. This is a
model response to a hypothetical change, not a cause of auction sales.
"""

import json

import numpy as np
from pytorch_forecasting import TemporalFusionTransformer

from forecast_auction_tft import (
    ROOT,
    TARGETS,
    latest_history,
    predict,
)
from train_auction_tft import (
    dataset,
    make_frames,
    read_windows,
)


def main():
    history, origin = latest_history()

    reference = dataset(
        make_frames(read_windows())["train"]
    )

    model = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            str(
                ROOT
                / "checkpoints"
                / "auction-tft.ckpt"
            ),
            map_location="cpu",
        )
    )
    model.eval()

    results = {}

    for target in TARGETS:
        baseline = predict(
            model,
            reference,
            history,
            target,
            placeholder=0.0,
        )[:, 1]

        if (
            len(baseline) != 4
            or not np.isfinite(baseline).all()
        ):
            raise ValueError(
                "Invalid reference forecast "
                f"for {target}"
            )

        effects = []

        for index in range(8):
            modified = [
                row.copy()
                for row in history
            ]

            original = float(
                modified[index][target]
            )

            modified[index][target] = str(
                original * 1.05
            )

            changed = predict(
                model,
                reference,
                modified,
                target,
                placeholder=0.0,
            )[:, 1]

            delta = changed - baseline

            effects.append({
                "input_week_start":
                    history[index]["week_start"],
                "observed_1000kg":
                    original,
                "hypothetical_change":
                    "+5% to this one observed week",
                "median_forecast_change_1000kg_by_horizon": [
                    round(float(value), 2)
                    for value in delta
                ],
                "mean_absolute_change_1000kg":
                    round(
                        float(
                            np.mean(
                                np.abs(delta)
                            )
                        ),
                        2,
                    ),
            })

        results[target] = {
            "original_median_1000kg_by_horizon": [
                round(float(value), 2)
                for value in baseline
            ],
            "input_sensitivity": sorted(
                effects,
                key=lambda row: row[
                    "mean_absolute_change_1000kg"
                ],
                reverse=True,
            ),
        }

    output = {
        "source_week_start":
            origin.isoformat(),
        "method": (
            "Increase one historical input by 5%, "
            "hold the other seven fixed, rerun "
            "the same checkpoint"
        ),
        "scope": (
            "Local sensitivity of this dated "
            "model forecast only"
        ),
        "caution": (
            "These are model responses to hypothetical "
            "inputs. They do not identify causes of "
            "market sales or show what an industry "
            "user understood."
        ),
        "unit": "thousand kg",
        "results": results,
    }

    (
        ROOT / "local_forecast_sensitivity.json"
    ).write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()