"""Select a method on 2025 validation data and report 2026 test results.

The 2026 test results have already been inspected during development.
This analysis is exploratory; later untouched data is needed to confirm it.
"""

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer

from train_auction_tft import ROOT, dataset, make_frames, read_windows


METHODS = ("last", "mean4", "tft")
TARGETS = ("high_1000kg", "medium_1000kg", "low_1000kg")
HORIZON = 4


def key(row):
    return (
        row["origin_week"],
        row["target_week"],
        int(row["horizon_week"]),
        row["target"],
    )


def read_csv(filename):
    with (ROOT / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_errors(expected_origins):
    baseline_rows = read_csv("baseline_matched_8week.csv")
    tft_rows = read_csv("tft_predictions.csv")

    actual = {}
    baseline_predictions = {}

    for row in baseline_rows:
        if row["target"] not in TARGETS:
            continue

        identifier = key(row)
        method = row["method"]
        if method not in ("last", "mean4"):
            raise ValueError(f"Unexpected baseline method: {method}")

        observed = float(row["actual_1000kg"])
        if identifier in actual and actual[identifier] != observed:
            raise ValueError(f"Conflicting actual values: {identifier}")
        actual[identifier] = observed

        prediction_key = (identifier, method)
        if prediction_key in baseline_predictions:
            raise ValueError(f"Duplicate baseline prediction: {prediction_key}")
        baseline_predictions[prediction_key] = float(
            row["predicted_1000kg"]
        )

    tft_predictions = {}
    for row in tft_rows:
        if row["target"] not in TARGETS:
            continue

        identifier = key(row)
        if identifier in tft_predictions:
            raise ValueError(f"Duplicate TFT prediction: {identifier}")
        tft_predictions[identifier] = float(
            row["predicted_1000kg"]
        )

    if set(tft_predictions) != set(actual):
        raise ValueError(
            "TFT predictions and matched baselines use different test dates"
        )

    errors = defaultdict(list)
    for identifier, observed in actual.items():
        target = identifier[3]

        for method in ("last", "mean4"):
            prediction_key = (identifier, method)
            if prediction_key not in baseline_predictions:
                raise ValueError(
                    f"Missing baseline prediction: {prediction_key}"
                )
            errors[(target, method)].append(
                abs(baseline_predictions[prediction_key] - observed)
            )

        errors[(target, "tft")].append(
            abs(tft_predictions[identifier] - observed)
        )

    expected_predictions = expected_origins * HORIZON
    for target in TARGETS:
        for method in METHODS:
            count = len(errors[(target, method)])
            if count != expected_predictions:
                raise ValueError(
                    f"{target}/{method}: expected "
                    f"{expected_predictions} predictions, found {count}"
                )

    return errors


def main():
    rows = read_windows()
    frames = make_frames(rows)
    reference = dataset(frames["train"])

    checkpoint = ROOT / "checkpoints" / "auction-tft.ckpt"
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    model.eval()

    validation_errors = defaultdict(list)

    for row in rows:
        if row["split"] != "validation":
            continue

        target = row["target"]
        history = np.array(
            [
                float(row[f"history_week_{week}_1000kg"])
                for week in range(1, 9)
            ]
        )
        actual = np.array(
            [
                float(row[f"future_week_{week}_1000kg"])
                for week in range(1, HORIZON + 1)
            ]
        )

        window_id = f"validation:{target}:{row['origin_week']}"
        frame = frames["validation"].loc[
            frames["validation"]["window_id"] == window_id
        ].copy()
        sample = dataset(frame, reference, predict=True)

        if len(sample) != 1:
            raise ValueError(
                f"Expected one validation window: {window_id}"
            )

        with torch.no_grad():
            prediction = model.predict(
                sample.to_dataloader(
                    train=False,
                    batch_size=1,
                    num_workers=0,
                ),
                mode="prediction",
            ).detach().cpu().numpy().reshape(-1)

        if len(prediction) != HORIZON or not np.isfinite(
            prediction
        ).all():
            raise ValueError(
                f"Invalid validation prediction: {window_id}"
            )

        candidates = {
            "last": np.repeat(history[-1], HORIZON),
            "mean4": np.repeat(history[-4:].mean(), HORIZON),
            "tft": np.maximum(prediction, 0),
        }

        for method, values in candidates.items():
            validation_errors[(target, method)].extend(
                np.abs(values - actual).tolist()
            )

    validation_counts = {
        target: sum(
            row["split"] == "validation" and row["target"] == target
            for row in rows
        )
        for target in TARGETS
    }
    test_counts = {
        target: sum(
            row["split"] == "test" and row["target"] == target
            for row in rows
        )
        for target in TARGETS
    }

    if len(set(test_counts.values())) != 1:
        raise ValueError(
            f"Different test-origin counts by elevation: {test_counts}"
        )

    test = test_errors(next(iter(test_counts.values())))

    result = {
        "selection_period": "2025 validation",
        "report_period": "2026 test",
        "unit": "thousand kg MAE across four forecast weeks",
        "caution": (
            "Exploratory analysis: the 2026 test results had already "
            "been inspected. Confirm the selection rule on later "
            "untouched auction observations."
        ),
        "elevations": {},
    }

    for target in TARGETS:
        validation_origins = validation_counts[target]
        test_origins = test_counts[target]

        if validation_origins < 1 or test_origins < 1:
            raise ValueError(f"No validation or test origins for {target}")

        expected_count = validation_origins * HORIZON
        for method in METHODS:
            count = len(validation_errors[(target, method)])
            if count != expected_count:
                raise ValueError(
                    f"{target}/{method}: expected {expected_count} "
                    f"validation predictions, found {count}"
                )

        validation_mae = {
            method: round(
                float(np.mean(validation_errors[(target, method)])),
                2,
            )
            for method in METHODS
        }
        selected = min(
            METHODS,
            key=lambda method: validation_mae[method],
        )
        test_mae = {
            method: round(float(np.mean(test[(target, method)])), 2)
            for method in METHODS
        }

        result["elevations"][target] = {
            "validation_origins": validation_origins,
            "test_origins": test_origins,
            "validation_mae_1000kg": validation_mae,
            "selected_method": selected,
            "test_mae_1000kg": test_mae,
            "selected_test_mae_1000kg": test_mae[selected],
        }

    output = ROOT / "validation_model_selection.json"
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()