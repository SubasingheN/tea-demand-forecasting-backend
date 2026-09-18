"""Export aggregate TFT interpretation for the real auction test windows.

This describes model-internal variable reliance. It does not establish
causes of auction sales or explain an individual forecast.
"""

import json

import torch
from pytorch_forecasting import TemporalFusionTransformer

from train_auction_tft import ROOT, dataset, make_frames, read_windows


TARGETS = ("high_1000kg", "medium_1000kg", "low_1000kg")
WEIGHT_GROUPS = (
    "encoder_variables",
    "decoder_variables",
    "static_variables",
)


def weights(names, values):
    numbers = (
        torch.as_tensor(values)
        .detach()
        .cpu()
        .float()
        .flatten()
        .tolist()
    )
    if len(names) != len(numbers):
        raise ValueError(
            f"Variable labels ({len(names)}) and weights "
            f"({len(numbers)}) differ"
        )
    return {
        name: round(float(value), 6)
        for name, value in zip(names, numbers)
    }


def main():
    rows = read_windows()
    frames = make_frames(rows)
    reference = dataset(frames["train"])

    checkpoint = ROOT / "checkpoints" / "auction-tft.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    model.eval()

    findings = {}
    for target in TARGETS:
        target_rows = [
            row for row in rows
            if row["split"] == "test" and row["target"] == target
        ]
        combined = None

        for row in target_rows:
            window_id = f"test:{target}:{row['origin_week']}"
            frame = frames["test"].loc[
                frames["test"]["window_id"] == window_id
            ].copy()
            sample = dataset(frame, reference, predict=True)

            if len(sample) != 1:
                raise ValueError(
                    f"Expected one prediction window for {window_id}"
                )

            with torch.no_grad():
                raw = model.predict(
                    sample.to_dataloader(
                        train=False, batch_size=1, num_workers=0
                    ),
                    mode="raw",
                    return_x=True,
                )
                interpretation = model.interpret_output(
                    raw.output, reduction="sum"
                )

            if combined is None:
                combined = {
                    name: interpretation[name]
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                    for name in WEIGHT_GROUPS
                }
            else:
                for name in WEIGHT_GROUPS:
                    combined[name] += (
                        interpretation[name].detach().cpu().float()
                    )

        if combined is None:
            raise ValueError(f"No test windows for {target}")

        findings[target] = {
            "test_origins": len(target_rows),
            "encoder_variables": weights(
                model.encoder_variables,
                combined["encoder_variables"],
            ),
            "decoder_variables": weights(
                model.decoder_variables,
                combined["decoder_variables"],
            ),
            "static_variables": weights(
                model.static_variables,
                combined["static_variables"],
            ),
        }

    counts = {
        target: findings[target]["test_origins"]
        for target in TARGETS
    }

    result = {
        "scope": (
            "Aggregate TFT variable reliance across observed test "
            "origins for each elevation"
        ),
        "test_origins_per_target": counts,
        "caution": (
            "Model-internal weights are not causal effects, forecast "
            "accuracy, or explanations of individual forecasts. "
            "Only historical sales, relative time, and elevation "
            "were available as inputs."
        ),
        "checkpoint": str(checkpoint),
        "results": findings,
    }

    output = ROOT / "tft_explanation.json"
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()