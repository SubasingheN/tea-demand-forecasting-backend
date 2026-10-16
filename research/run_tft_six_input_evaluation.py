
"""
Run six-input TFT evaluation with JSON-safe values.

Preserves:
- Existing model checkpoints
- Original datasets
- Original evaluation script

Converts NaN and Infinity to JSON null.
"""

import json
import math

import numpy as np
import pandas as pd

import evaluate_tft_six_input as evaluation


def clean_json_value(value):
    """
    Recursively convert non-finite numeric values
    into JSON-compatible null values.
    """

    if isinstance(value, dict):
        return {
            key: clean_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            clean_json_value(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def install_json_safe_encoder():
    original_dumps = json.dumps

    def safe_dumps(obj, *args, **kwargs):
        cleaned = clean_json_value(obj)

        return original_dumps(
            cleaned,
            *args,
            **kwargs,
        )

    evaluation.json.dumps = safe_dumps


def main():
    print(
        "\n=== JSON-SAFE SIX-INPUT EVALUATION ==="
    )

    install_json_safe_encoder()

    evaluation.main()

    summary_path = evaluation.SUMMARY_FILE

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file,
            parse_constant=lambda value: (
                _raise_invalid_json(value)
            ),
        )

    assert summary["model"] == "six_input_tft"
    assert summary["test_windows"] == 255
    assert summary["prediction_rows"] == 1020

    print("\nJSON SUMMARY VALIDATION: PASS")
    print("Invalid JSON numbers: NONE")
    print("Model retrained: NO")


def _raise_invalid_json(value):
    raise ValueError(
        f"Invalid JSON constant: {value}"
    )


if __name__ == "__main__":
    main()
