
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.data import GroupNormalizer

from build_tft_datasets import (
    GRADES,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    load_features,
    load_manifests,
    validate_datasets,
)

from evaluate_tft import (
    evaluate_predictions,
    build_metric_tables,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "research" / "outputs" / "tft_ablation"
)

PRICE_CHECKPOINT = (
    ROOT / "research" / "outputs" / "tft_full_training"
    / "checkpoints" / "tft-epoch09-val496.3011.ckpt"
)

NO_PRICE_CHECKPOINT = (
    ROOT / "research" / "outputs" / "tft_no_price"
    / "checkpoints" / "tft-no-price-epoch09-val470.2676.ckpt"
)

KNOWN_REALS = [
    "time_idx",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
]

NO_PRICE_UNKNOWN_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
]

VALIDATION_START = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")


def build_no_price_validation(df, manifests):
    columns = (
        ["week_start_date", "grade"]
        + KNOWN_REALS
        + NO_PRICE_UNKNOWN_REALS
    )

    no_price_df = df[columns].copy()

    train_df = no_price_df[
        no_price_df["week_start_date"] < VALIDATION_START
    ].copy()

    val_df = no_price_df[
        no_price_df["week_start_date"] < TEST_START
    ].copy()

    validation_start_idx = int(
        no_price_df.loc[
            no_price_df["week_start_date"]
            >= VALIDATION_START,
            "time_idx",
        ].min()
    )

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="demand_kg",
        group_ids=["grade"],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["grade"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=NO_PRICE_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(
            groups=["grade"],
            transformation="softplus",
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
        allow_missing_timesteps=False,
        randomize_length=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        min_prediction_idx=validation_start_idx,
        stop_randomization=True,
        predict=False,
    )

    # Verify the window identities directly.
    decoded = validation.decoded_index

    actual = {
        (str(row.grade), int(row.time_idx_first_prediction))
        for row in decoded.itertuples(index=False)
    }

    date_to_idx = (
        no_price_df[
            ["week_start_date", "time_idx"]
        ]
        .drop_duplicates()
        .set_index("week_start_date")["time_idx"]
        .to_dict()
    )

    expected = {
        (
            str(row.grade),
            int(date_to_idx[pd.Timestamp(row.target_start)]),
        )
        for row in manifests["validation"].itertuples(index=False)
    }

    if actual != expected:
        raise ValueError(
            "No-price validation windows do not match "
            "the Step 8 validation manifest."
        )

    return validation


def make_date_mapping(df):
    pairs = df[
        ["time_idx", "week_start_date"]
    ].drop_duplicates()

    if pairs["time_idx"].duplicated().any():
        raise ValueError("Invalid shared time index.")

    return {
        int(row.time_idx): pd.Timestamp(
            row.week_start_date
        )
        for row in pairs.itertuples(index=False)
    }


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(path)

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(path),
        map_location="cpu",
        weights_only=False,
    )

    model.eval()
    model.cpu()

    return model


def evaluate_one(
    label,
    checkpoint,
    dataset,
    manifest,
    index_to_date,
):
    print(f"\nEVALUATING {label}")
    print("-" * 65)

    model = load_checkpoint(checkpoint)

    predictions, crossings = evaluate_predictions(
        model,
        dataset,
        manifest,
        index_to_date,
    )

    if crossings:
        print(
            f"Warning: {crossings} crossing "
            "quantile observations."
        )

    grade, horizon, detail, overall = (
        build_metric_tables(predictions)
    )

    predictions = predictions.rename(
        columns={
            "tft_q10_kg": f"{label}_q10_kg",
            "tft_q50_kg": f"{label}_q50_kg",
            "tft_q90_kg": f"{label}_q90_kg",
            "within_80_percent_interval":
                f"{label}_interval_covered",
        }
    )

    return predictions, grade, horizon, detail, overall


def main():
    print("STEP 14 - MATCHED TFT AUCTION-PRICE ABLATION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    manifests = load_manifests()

    # Rebuild the price-aware validation dataset.
    from build_tft_datasets import create_datasets

    training, price_validation, testing = (
        create_datasets(df)
    )

    validate_datasets(
        {
            "train": training,
            "validation": price_validation,
            "test": testing,
        },
        manifests,
        df,
    )

    no_price_validation = build_no_price_validation(
        df, manifests
    )

    manifest = manifests["validation"]
    index_to_date = make_date_mapping(df)

    (
        with_price,
        price_grade,
        price_horizon,
        price_detail,
        price_overall,
    ) = evaluate_one(
        "with_price",
        PRICE_CHECKPOINT,
        price_validation,
        manifest,
        index_to_date,
    )

    (
        no_price,
        no_price_grade,
        no_price_horizon,
        no_price_detail,
        no_price_overall,
    ) = evaluate_one(
        "no_price",
        NO_PRICE_CHECKPOINT,
        no_price_validation,
        manifest,
        index_to_date,
    )

    keys = [
        "grade",
        "forecast_origin",
        "target_start",
        "target_week",
        "horizon",
    ]

    selected = keys + [
        "actual_demand_kg",
        "no_price_q10_kg",
        "no_price_q50_kg",
        "no_price_q90_kg",
        "no_price_interval_covered",
    ]

    matched = with_price.merge(
        no_price[selected],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_other"),
    )

    if len(matched) != 1000:
        raise ValueError(
            f"Expected 1000 matched predictions, "
            f"found {len(matched)}."
        )

    if not np.allclose(
        matched["actual_demand_kg"],
        matched["actual_demand_kg_other"],
        rtol=0,
        atol=1e-5,
    ):
        raise ValueError(
            "Model targets differ on matched windows."
        )

    matched = matched.drop(
        columns=["actual_demand_kg_other"]
    )

    def comparison_row(label, a, b):
        return {
            "group": label,
            "with_price_mae_kg": a["tft_mae_kg"],
            "no_price_mae_kg": b["tft_mae_kg"],
            "with_price_rmse_kg": a["tft_rmse_kg"],
            "no_price_rmse_kg": b["tft_rmse_kg"],
            "with_price_wape_percent":
                a["tft_wape_percent"],
            "no_price_wape_percent":
                b["tft_wape_percent"],
            "with_price_interval_coverage_percent":
                a["interval_80_coverage_percent"],
            "no_price_interval_coverage_percent":
                b["interval_80_coverage_percent"],
            "price_mae_benefit_percent": (
                100 * (
                    b["tft_mae_kg"] - a["tft_mae_kg"]
                ) / b["tft_mae_kg"]
                if b["tft_mae_kg"] > 0 else None
            ),
        }

    rows = [
        comparison_row(
            "overall",
            price_overall,
            no_price_overall,
        )
    ]

    for grade in GRADES:
        a = price_grade.loc[
            price_grade["grade"] == grade
        ].iloc[0]

        b = no_price_grade.loc[
            no_price_grade["grade"] == grade
        ].iloc[0]

        rows.append(
            comparison_row(grade, a, b)
        )

    for horizon in range(1, PREDICTION_LENGTH + 1):
        a = price_horizon.loc[
            price_horizon["horizon"] == horizon
        ].iloc[0]

        b = no_price_horizon.loc[
            no_price_horizon["horizon"] == horizon
        ].iloc[0]

        rows.append(
            comparison_row(
                f"week_{horizon}", a, b
            )
        )

    comparison = pd.DataFrame(rows)

    matched_path = (
        OUTPUT_DIR / "matched_validation_predictions.csv"
    )

    comparison_path = (
        OUTPUT_DIR / "ablation_comparison.csv"
    )

    report_path = (
        OUTPUT_DIR / "ablation_summary.json"
    )

    matched.to_csv(matched_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    report = {
        "status": "PASS",
        "partition": "2024_validation",
        "matched_prediction_rows": len(matched),
        "forecast_windows": 250,
        "with_price_checkpoint": str(
            PRICE_CHECKPOINT
        ),
        "no_price_checkpoint": str(
            NO_PRICE_CHECKPOINT
        ),
        "overall": rows[0],
        "test_partition_evaluated": False,
        "notes": [
            "The two models use identical target dates.",
            "Price ablation removes quotation price "
            "inputs and price availability masks.",
            "Auction-price effects are observational, "
            "not causal.",
            "The held-out test data remains reserved.",
        ],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nMATCHED MODEL COMPARISON")
    print("=" * 65)

    print(
        comparison[
            [
                "group",
                "with_price_mae_kg",
                "no_price_mae_kg",
                "price_mae_benefit_percent",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print(matched_path)
    print(comparison_path)
    print(report_path)

    print("\nMATCHED ABLATION EVALUATION: PASS")


if __name__ == "__main__":
    main()
