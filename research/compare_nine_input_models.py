
"""
Compare nine-input TFT checkpoints using identical
chronological validation forecast windows.

Models:
- Original nine-input TFT
- Candidate A
- Candidate B
- Seasonal 52-week baseline

Selection metric: validation MAE in kilograms.

This script does not:
- Retrain any model
- Evaluate the held-out test partition
- Modify checkpoints
- Replace the factory forecasting model
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer

from build_eight_input_dataset import (
    load_eight_input_data,
    create_eight_input_datasets,
)

from tft_eight_input_config import (
    OUTPUT_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    TEA_GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    QUANTILES,
    VALIDATION_START,
    TEST_START,
    HISTORICAL_ONLY_REALS,
    validate_feature_configuration,
)


ORIGINAL_REPORT = (
    OUTPUT_DIR / "nine_input_training_metadata.json"
)

OPTIMIZATION_REPORT = (
    OUTPUT_DIR
    / "optimization"
    / "optimization_summary.json"
)

COMPARISON_DIR = OUTPUT_DIR / "final_comparison"

RESULTS_FILE = COMPARISON_DIR / "validation_comparison.csv"

SUMMARY_FILE = COMPARISON_DIR / "validation_comparison.json"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def get_checkpoints():
    original = read_json(ORIGINAL_REPORT)
    optimized = read_json(OPTIMIZATION_REPORT)

    if original.get("model") != MODEL_NAME:
        raise ValueError("Original model metadata mismatch.")

    if original.get("historical_only_features") != HISTORICAL_ONLY_REALS:
        raise ValueError("Original model feature mismatch.")

    if optimized.get("historical_features") != HISTORICAL_ONLY_REALS:
        raise ValueError("Optimized model feature mismatch.")

    candidates = {
        entry["name"]: entry
        for entry in optimized["all_candidates"]
    }

    if set(candidates) != {"candidate_a", "candidate_b"}:
        raise ValueError("Expected both optimization candidates.")

    paths = {
        "original_tft": Path(original["checkpoint"]),
        "candidate_a": Path(candidates["candidate_a"]["checkpoint"]),
        "candidate_b": Path(candidates["candidate_b"]["checkpoint"]),
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {name} is missing: {path}"
            )

    return paths


def collect_forecasts(checkpoint, validation_dataset):
    model = TemporalFusionTransformer.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )

    model.eval()
    model.cpu()

    model_quantiles = getattr(model.loss, "quantiles", None)

    if model_quantiles is not None and not np.allclose(
        model_quantiles, QUANTILES
    ):
        raise ValueError("Checkpoint quantiles mismatch.")

    loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    rows = []

    with torch.inference_mode():
        for x, y in loader:
            output = model(x)

            raw = output.prediction.detach().cpu()

            actual = (
                y[0] if isinstance(y, (list, tuple)) else y
            ).detach().cpu()

            if raw.ndim != 3 or raw.shape[1:] != (
                PREDICTION_LENGTH, len(QUANTILES)
            ):
                raise ValueError("Invalid model output dimensions.")

            if actual.shape != raw.shape[:2]:
                raise ValueError("Actual/prediction shape mismatch.")

            if not torch.isfinite(raw).all():
                raise ValueError("Non-finite model predictions.")

            if not torch.isfinite(actual).all():
                raise ValueError("Non-finite demand values.")

            crossing = (
                (raw[..., 0] > raw[..., 1])
                | (raw[..., 1] > raw[..., 2])
            )

            ordered = torch.sort(raw, dim=-1).values

            encoder = x["encoder_target"].detach().cpu()

            if encoder.shape[1] != ENCODER_LENGTH:
                raise ValueError("Unexpected encoder length.")

            indices = validation_dataset.x_to_index(x)

            for sample in range(len(indices)):
                grade = str(indices.iloc[sample]["grade"])
                start = int(indices.iloc[sample]["time_idx"])

                if grade not in TEA_GRADES:
                    raise ValueError(f"Unknown tea grade: {grade}")

                for h in range(1, PREDICTION_LENGTH + 1):
                    step = h - 1
                    target_idx = start + step
                    seasonal_position = h - 53

                    if not (
                        -ENCODER_LENGTH <= seasonal_position < 0
                    ):
                        raise ValueError(
                            "Seasonal baseline index invalid."
                        )

                    rows.append({
                        "grade": grade,
                        "target_time_idx": target_idx,
                        "horizon": h,
                        "actual_kg": float(actual[sample, step]),
                        "p10_kg": float(ordered[sample, step, 0]),
                        "p50_kg": float(ordered[sample, step, 1]),
                        "p90_kg": float(ordered[sample, step, 2]),
                        "seasonal_kg": float(
                            encoder[sample, seasonal_position]
                        ),
                        "crossing": bool(crossing[sample, step]),
                    })

    result = pd.DataFrame(rows)

    if len(result) != len(validation_dataset) * PREDICTION_LENGTH:
        raise ValueError("Incomplete validation forecasts.")

    keys = ["grade", "target_time_idx", "horizon"]

    if result.duplicated(keys).any():
        raise ValueError("Duplicate validation forecast keys.")

    return result.sort_values(keys).reset_index(drop=True)


def calculate_metrics(frame, column):
    actual = frame["actual_kg"].to_numpy(dtype=float)
    predicted = frame[column].to_numpy(dtype=float)

    if len(actual) == 0:
        raise ValueError("Empty comparison data.")

    difference = predicted - actual

    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))

    total = float(np.sum(np.abs(actual)))

    wape = (
        float(100 * np.abs(difference).sum() / total)
        if total > 0 else None
    )

    return {
        "mae_kg": round(mae, 4),
        "rmse_kg": round(rmse, 4),
        "wape_percent": (
            round(wape, 4) if wape is not None else None
        ),
    }


def evaluate_model(name, frame, baseline=False):
    column = "seasonal_kg" if baseline else "p50_kg"

    metrics = calculate_metrics(frame, column)

    result = {
        "model": name,
        "samples": len(frame),
        **metrics,
        "interval_coverage_percent": None,
        "crossing_percent": None,
    }

    if not baseline:
        actual = frame["actual_kg"].to_numpy()
        lower = frame["p10_kg"].to_numpy()
        upper = frame["p90_kg"].to_numpy()

        result["interval_coverage_percent"] = round(
            100 * float(np.mean(
                (actual >= lower) & (actual <= upper)
            )),
            4,
        )

        result["crossing_percent"] = round(
            100 * float(frame["crossing"].mean()),
            4,
        )

    return result


def main():
    print("\n=== FINAL NINE-INPUT VALIDATION COMPARISON ===")

    validate_feature_configuration()

    if ENCODER_LENGTH != 52 or PREDICTION_LENGTH != 4:
        raise ValueError("Unexpected forecasting configuration.")

    if list(QUANTILES) != [0.1, 0.5, 0.9]:
        raise ValueError("Expected P10, P50 and P90.")

    for path in [RESULTS_FILE, SUMMARY_FILE]:
        if path.exists():
            raise FileExistsError(
                f"Comparison output already exists: {path}"
            )

    checkpoints = get_checkpoints()

    df = load_eight_input_data()
    datasets = create_eight_input_datasets(df)

    validation = datasets["validation"]

    print("Validation windows:", len(validation))

    reference = None
    results = []

    keys = ["grade", "target_time_idx", "horizon"]

    for name, checkpoint in checkpoints.items():
        print(f"\nEvaluating: {name}")

        predictions = collect_forecasts(
            checkpoint, validation
        )

        # All checkpoints must be evaluated on
        # the same targets and actual demand.
        identity = predictions[
            keys + ["actual_kg", "seasonal_kg"]
        ].copy()

        if reference is None:
            reference = identity
        elif not reference.equals(identity):
            raise ValueError(
                "Models were not evaluated on "
                "identical validation observations."
            )

        results.append(
            evaluate_model(name, predictions)
        )

    seasonal_frame = reference.rename(
        columns={"seasonal_kg": "seasonal_kg"}
    )

    results.append(
        evaluate_model(
            "seasonal_52w",
            seasonal_frame,
            baseline=True,
        )
    )

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        "mae_kg"
    ).reset_index(drop=True)

    selected = comparison.iloc[0]["model"]

    print("\n=== MATCHED VALIDATION RESULTS ===")
    print(comparison.to_string(index=False))

    print("\n=== LOWEST VALIDATION MAE ===")
    print("Model:", selected)
    print("MAE:", comparison.iloc[0]["mae_kg"])

    COMPARISON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary = {
        "comparison": "matched_chronological_validation",
        "validation_start": str(VALIDATION_START.date()),
        "validation_end_exclusive": str(TEST_START.date()),
        "selection_metric": "mae_kg",
        "selected_model": selected,
        "validation_forecast_count": len(reference),
        "results": comparison.where(
            pd.notna(comparison), None
        ).to_dict(orient="records"),
        "checkpoints": {
            name: str(path)
            for name, path in checkpoints.items()
        },
        "test_evaluated": False,
        "model_retrained": False,
        "factory_model_replaced": False,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", RESULTS_FILE)
    print("Saved:", SUMMARY_FILE)

    print("\nVALIDATION COMPARISON: PASS")
    print("Factory model replaced: NO")


if __name__ == "__main__":
    main()
