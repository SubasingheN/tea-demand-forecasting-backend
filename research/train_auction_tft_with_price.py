"""Compare a price-aware TFT with the volume-only TFT on the same windows.

Only historical auction prices are used. During each forecast, the
latest observed price is carried forward; future observed prices
are never supplied to the model.
"""

import csv
import json
import random
from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import ConcatDataset, DataLoader

from train_auction_tft import ROOT, HISTORY, HORIZON, SEED


TARGETS = (
    "high_1000kg",
    "medium_1000kg",
    "low_1000kg",
)


def read_csv(filename):
    with (
        ROOT / filename
    ).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_frames(windows):
    records = {
        split: []
        for split in ("train", "validation", "test")
    }

    for window in windows:
        split = window["split"]
        target = window["target"]
        origin = window["origin_week"]

        if split not in records or target not in TARGETS:
            raise ValueError(
                f"Unexpected window: {split}, {target}"
            )

        window_id = f"{split}:{target}:{origin}"
        previous_price = float(
            window[
                f"history_price_{HISTORY}_lkr_per_kg"
            ]
        )

        for offset in range(HISTORY + HORIZON):
            if offset < HISTORY:
                sale = float(
                    window[
                        f"history_week_{offset + 1}_1000kg"
                    ]
                )
                price = float(
                    window[
                        f"history_price_{offset + 1}_lkr_per_kg"
                    ]
                )
            else:
                sale = float(
                    window[
                        f"future_week_"
                        f"{offset - HISTORY + 1}_1000kg"
                    ]
                )
                # Known at the forecast origin.
                # Never use the actual future price.
                price = previous_price

            records[split].append({
                "window_id": window_id,
                "elevation": target.replace(
                    "_1000kg", ""
                ),
                "time_idx": offset,
                "sale_1000kg": sale,
                "price_lkr_per_kg": price,
            })

    return {
        split: pd.DataFrame(values)
        for split, values in records.items()
    }


def make_dataset(frame, reference=None, predict=False):
    if reference is not None:
        return TimeSeriesDataSet.from_dataset(
            reference,
            frame,
            predict=predict,
            stop_randomization=True,
        )

    return TimeSeriesDataSet(
        frame,
        time_idx="time_idx",
        target="sale_1000kg",
        group_ids=["window_id"],
        min_encoder_length=HISTORY,
        max_encoder_length=HISTORY,
        min_prediction_length=HORIZON,
        max_prediction_length=HORIZON,
        static_categoricals=["elevation"],
        time_varying_known_reals=[
            "time_idx",
            "price_lkr_per_kg",
        ],
        time_varying_unknown_reals=["sale_1000kg"],
        categorical_encoders={
            "__group_id__window_id":
                NaNLabelEncoder(add_nan=True)
        },
        target_normalizer=TorchNormalizer(
            method="standard"
        ),
        allow_missing_timesteps=False,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=False,
    )


def sample_for(row, frames, train):
    split = row["split"]
    target = row["target"]
    origin = row["origin_week"]
    window_id = f"{split}:{target}:{origin}"

    frame = frames[split].loc[
        frames[split]["window_id"] == window_id
    ].copy()
    sample = make_dataset(
        frame, train, predict=True
    )

    if len(sample) != 1:
        raise ValueError(
            f"Expected one window for {window_id}"
        )

    return sample


def prediction_key(row):
    return (
        row["origin_week"],
        row["target_week"],
        int(row["horizon_week"]),
        row["target"],
    )


def compare(windows, new_predictions):
    expected = {
        (
            row["origin_week"],
            row[f"future_week_{h}_date"],
            h,
            row["target"],
        ): float(
            row[f"future_week_{h}_1000kg"]
        )
        for row in windows
        if row["split"] == "test"
        for h in range(1, HORIZON + 1)
    }

    baseline = read_csv(
        "baseline_matched_8week.csv"
    )
    sources = {
        "last": [
            row for row in baseline
            if row["target"] in TARGETS
            and row["method"] == "last"
        ],
        "mean4": [
            row for row in baseline
            if row["target"] in TARGETS
            and row["method"] == "mean4"
        ],
        "volume_only_tft": read_csv(
            "tft_predictions.csv"
        ),
        "price_aware_tft": new_predictions,
    }

    metrics = []

    for method, source in sources.items():
        predictions = {}

        for row in source:
            key = prediction_key(row)
            if key in predictions:
                raise ValueError(
                    f"Duplicate {method} forecast: {key}"
                )

            predictions[key] = float(
                row["predicted_1000kg"]
            )

            if (
                "actual_1000kg" in row
                and float(row["actual_1000kg"])
                != expected.get(key)
            ):
                raise ValueError(
                    f"Mismatched outcome for "
                    f"{method}: {key}"
                )

        if predictions.keys() != expected.keys():
            raise ValueError(
                f"{method} does not match "
                f"the test dates"
            )

        errors = defaultdict(list)

        for key, actual in expected.items():
            errors[(key[3], key[2])].append(
                abs(actual - predictions[key])
            )

        for target in TARGETS:
            for horizon in range(
                1, HORIZON + 1
            ):
                values = errors[
                    (target, horizon)
                ]
                metrics.append({
                    "target": target,
                    "method": method,
                    "horizon_week": horizon,
                    "n_origins": len(values),
                    "mae_1000kg": round(
                        float(np.mean(values)),
                        2,
                    ),
                })

    return {
        "target_kind":
            "observed_auction_sales_by_elevation",
        "experiment": (
            "historical average auction price "
            "input vs volume-only TFT"
        ),
        "unit": "thousand kg MAE",
        "test_origins_per_elevation":
            len(expected) //
            (len(TARGETS) * HORIZON),
        "warning": (
            "Exploratory ablation: 2026 test "
            "outcomes were inspected before this "
            "experiment. No causal effect of price "
            "or confirmatory improvement can be claimed."
        ),
        "metrics": metrics,
    }


def main():
    pl.seed_everything(SEED, workers=True)
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)

    windows = read_csv(
        "tft_8week_price_windows.csv"
    )
    frames = make_frames(windows)
    train = make_dataset(
        frames["train"]
    )

    validation = [
        sample_for(row, frames, train)
        for row in windows
        if row["split"] == "validation"
    ]

    if len(train) != sum(
        row["split"] == "train"
        for row in windows
    ):
        raise ValueError(
            "Training window count differs"
        )

    if len(validation) != sum(
        row["split"] == "validation"
        for row in windows
    ):
        raise ValueError(
            "Validation window count differs"
        )

    model = TemporalFusionTransformer.from_dataset(
        train,
        learning_rate=0.001,
        hidden_size=8,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        loss=QuantileLoss(
            quantiles=[0.1, 0.5, 0.9]
        ),
        optimizer="adam",
        log_interval=-1,
    )

    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="auction-tft-with-price",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
            checkpoint,
        ],
        gradient_clip_val=0.1,
        deterministic=True,
        logger=False,
        enable_model_summary=False,
    )

    validation_loader = DataLoader(
        ConcatDataset(validation),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=
            TimeSeriesDataSet._collate_fn,
    )

    trainer.fit(
        model,
        train.to_dataloader(
            train=True,
            batch_size=16,
            num_workers=0,
        ),
        validation_loader,
    )

    best = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint.best_model_path
    )
    predictions = []

    for row in windows:
        if row["split"] != "test":
            continue

        sample = sample_for(
            row, frames, train
        )
        output = best.predict(
            sample.to_dataloader(
                train=False,
                batch_size=1,
                num_workers=0,
            ),
            mode="prediction",
        )
        values = (
            output.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if (
            len(values) != HORIZON
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Invalid forecast for "
                f"{row['target']}, "
                f"{row['origin_week']}"
            )

        for h, value in enumerate(
            values, 1
        ):
            predictions.append({
                "origin_week":
                    row["origin_week"],
                "target_week":
                    row[f"future_week_{h}_date"],
                "horizon_week": h,
                "target": row["target"],
                "predicted_1000kg":
                    float(max(0, value)),
            })

    output_path = (
        ROOT / "tft_with_price_predictions.csv"
    )
    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=
                predictions[0].keys(),
        )
        writer.writeheader()
        writer.writerows(predictions)

    result = compare(
        windows, predictions
    )
    result["checkpoint"] = str(
        checkpoint.best_model_path
    )

    (
        ROOT / "price_ablation.json"
    ).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(
        result, indent=2
    ))


if __name__ == "__main__":
    main()