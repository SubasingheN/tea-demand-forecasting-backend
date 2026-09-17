"""Export aggregate TFT interpretation for the real auction test windows.

Run from the backend root after train_auction_tft.py. This is model-internal
variable reliance, not a causal explanation of auction sales.
"""

import json
from pathlib import Path

import torch
from pytorch_forecasting import TemporalFusionTransformer

from train_auction_tft import ROOT, dataset, make_frames, read_windows


def weights(names, values):
    numbers = torch.as_tensor(values).detach().cpu().float().flatten().tolist()
    if len(names) != len(numbers):
        raise ValueError(f"Variable labels ({len(names)}) and weights ({len(numbers)}) differ")
    return {name: round(float(value), 6) for name, value in zip(names, numbers)}


def main():
    rows = read_windows()
    frames = make_frames(rows)
    reference = dataset(frames["train"])
    checkpoint = ROOT / "checkpoints" / "auction-tft.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    findings = {}
    for target in ("high_1000kg", "medium_1000kg", "low_1000kg"):
        target_rows = [row for row in rows if row["split"] == "test" and row["target"] == target]
        combined = None
        for row in target_rows:
            window_id = f"test:{target}:{row['origin_week']}"
            frame = frames["test"].loc[frames["test"]["window_id"] == window_id].copy()
            sample = dataset(frame, reference, predict=True)
            if len(sample) != 1:
                raise ValueError(f"Expected one prediction window for {window_id}")
            with torch.no_grad():
                raw = model.predict(sample.to_dataloader(train=False, batch_size=1, num_workers=0),
                                    mode="raw", return_x=True)
                interpretation = model.interpret_output(raw.output, reduction="sum")
            if combined is None:
                combined = {name: interpretation[name].detach().cpu().float().clone()
                            for name in ("encoder_variables", "decoder_variables", "static_variables")}
            else:
                for name in combined:
                    combined[name] += interpretation[name].detach().cpu().float()
        if combined is None:
            raise ValueError(f"No test windows for {target}")
        findings[target] = {
            "test_origins": len(target_rows),
            "encoder_variables": weights(model.encoder_variables, combined["encoder_variables"]),
            "decoder_variables": weights(model.decoder_variables, combined["decoder_variables"]),
            "static_variables": weights(model.static_variables, combined["static_variables"]),
        }
    result = {
        "scope": "Aggregate TFT variable reliance across nine observed test origins per elevation",
        "caution": "Model-internal weights are not causal effects, forecast accuracy, or per-origin reasons."
                   " Only historical sales, relative time, and elevation were available as inputs.",
        "checkpoint": str(checkpoint),
        "results": findings,
    }
    output = ROOT / "tft_explanation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
