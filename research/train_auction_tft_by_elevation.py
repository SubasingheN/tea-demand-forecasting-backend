"""Test training-only target scaling by elevation.

Uses the original eight-week windows, architecture, seed and epoch budget.
Writes separate checkpoint and prediction files.
The 2026 test outcomes have already been inspected, so this is exploratory.
"""

import csv
import json
import random
from collections import defaultdict

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
)

from train_auction_tft import (
    ROOT,
    HORIZON,
    SEED,
    dataset,
    make_frames,
    read_windows,
)


TARGETS = (
    "high_1000kg",
    "medium_1000kg",
    "low_1000kg",
)


def read_csv(filename):
    with (
        ROOT / filename
    ).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sample(row, frames, reference):
    split = row["split"]
    window_id = (
        f"{split}:"
        f"{row['target']}:"
        f"{row['origin_week']}"
    )

    frame = frames[split].loc[
        frames[split]["window_id"]
        == window_id
    ].copy()

    result = dataset(
        frame,
        reference,
        predict=True,
    )

    if len(result) != 1:
        raise ValueError(
            f"Expected one window "
            f"for {window_id}"
        )

    return result


def normalize_frames(frames):
    # Fit scales using training windows only.
    training = frames["train"]
    stats = {}

    for target in TARGETS:
        elevation = target.replace(
            "_1000kg", ""
        )

        series = training.loc[
            training["elevation"]
            == elevation,
            "sale_1000kg",
        ]

        center = float(
            series.mean()
        )
        scale = float(
            series.std(ddof=0)
        )

        if (
            not np.isfinite(
                [center, scale]
            ).all()
            or scale <= 0
        ):
            raise ValueError(
                f"Invalid training scale "
                f"for {target}"
            )

        stats[elevation] = {
            "mean_1000kg": center,
            "std_1000kg": scale,
        }

    scaled = {}

    for split, frame in frames.items():
        copy = frame.copy()

        for elevation, values in stats.items():
            match = (
                copy["elevation"]
                == elevation
            )

            copy.loc[
                match, "sale_1000kg"
            ] = (
                copy.loc[
                    match, "sale_1000kg"
                ]
                - values["mean_1000kg"]
            ) / values["std_1000kg"]

        scaled[split] = copy

    return scaled, stats


def key(row):
    return (
        row["origin_week"],
        row["target_week"],
        int(row["horizon_week"]),
        row["target"],
    )


def compare(windows, new_predictions):
    actual = {
        (
            row["origin_week"],
            row[
                f"future_week_"
                f"{h}_date"
            ],
            h,
            row["target"],
        ): float(
            row[
                f"future_week_"
                f"{h}_1000kg"
            ]
        )
        for row in windows
        if row["split"] == "test"
        for h in range(
            1, HORIZON + 1
        )
    }

    baseline = read_csv(
        "baseline_matched_8week.csv"
    )

    sources = {
        "last": [
            row
            for row in baseline
            if (
                row["target"]
                in TARGETS
                and row["method"]
                == "last"
            )
        ],
        "mean4": [
            row
            for row in baseline
            if (
                row["target"]
                in TARGETS
                and row["method"]
                == "mean4"
            )
        ],
        "original_tft": read_csv(
            "tft_predictions.csv"
        ),
        "elevation_scaled_tft":
            new_predictions,
    }

    summary = {}

    for method, source in sources.items():
        predictions = {}

        for row in source:
            identifier = key(row)

            if identifier in predictions:
                raise ValueError(
                    f"Repeated {method} "
                    f"forecast for "
                    f"{identifier}"
                )

            predictions[
                identifier
            ] = float(
                row[
                    "predicted_1000kg"
                ]
            )

            if (
                "actual_1000kg" in row
                and float(
                    row["actual_1000kg"]
                )
                != actual.get(identifier)
            ):
                raise ValueError(
                    "Mismatched outcome "
                    f"for {identifier}"
                )

        if (
            predictions.keys()
            != actual.keys()
        ):
            raise ValueError(
                f"{method} predictions "
                "differ from the eligible "
                "test dates"
            )

        errors = defaultdict(list)

        for identifier, observed in actual.items():
            errors[
                identifier[3]
            ].append(
                abs(
                    observed
                    - predictions[
                        identifier
                    ]
                )
            )

        summary[method] = {
            target: round(
                float(
                    np.mean(
                        errors[target]
                    )
                ),
                2,
            )
            for target in TARGETS
        }

    return {
        "test_origins_per_elevation":
            len(actual) //
            (
                len(TARGETS)
                * HORIZON
            ),
        "unit": (
            "thousand kg MAE over "
            "four forecast weeks"
        ),
        "comparison": summary,
        "caution": (
            "Exploratory: 2026 test "
            "outcomes were inspected "
            "before this ablation. "
            "Confirm any improvement "
            "on later untouched data."
        ),
    }


def main():
    pl.seed_everything(
        SEED,
        workers=True,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    windows = read_windows()

    frames, stats = normalize_frames(
        make_frames(windows)
    )

    train = dataset(
        frames["train"]
    )

    validation = [
        sample(
            row,
            frames,
            train,
        )
        for row in windows
        if row["split"]
        == "validation"
    ]

    if len(train) != sum(
        row["split"] == "train"
        for row in windows
    ):
        raise ValueError(
            "Training window count differs"
        )

    if len(validation) != sum(
        row["split"]
        == "validation"
        for row in windows
    ):
        raise ValueError(
            "Validation window count differs"
        )

    model = (
        TemporalFusionTransformer
        .from_dataset(
            train,
            learning_rate=0.001,
            hidden_size=8,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=4,
            loss=QuantileLoss(
                quantiles=[
                    0.1,
                    0.5,
                    0.9,
                ]
            ),
            optimizer="adam",
            log_interval=-1,
        )
    )

    directory = (
        ROOT / "checkpoints"
    )
    directory.mkdir(
        exist_ok=True
    )

    checkpoint = ModelCheckpoint(
        dirpath=directory,
        filename=(
            "auction-tft-by-elevation"
        ),
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
        collate_fn=(
            TimeSeriesDataSet
            ._collate_fn
        ),
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

    best = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            checkpoint.best_model_path
        )
    )

    predictions = []

    for row in windows:
        if row["split"] != "test":
            continue

        test = sample(
            row,
            frames,
            train,
        )

        with torch.no_grad():
            output = best.predict(
                test.to_dataloader(
                    train=False,
                    batch_size=1,
                    num_workers=0,
                ),
                mode="prediction",
            )

        normalized = (
            output.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if (
            len(normalized) != HORIZON
            or not np.isfinite(
                normalized
            ).all()
        ):
            raise ValueError(
                "Invalid prediction for "
                f"{row['target']}, "
                f"{row['origin_week']}"
            )

        scale = stats[
            row["target"].replace(
                "_1000kg", ""
            )
        ]

        volumes = (
            normalized
            * scale["std_1000kg"]
            + scale["mean_1000kg"]
        )

        for h, value in enumerate(
            volumes,
            1,
        ):
            predictions.append({
                "origin_week":
                    row["origin_week"],
                "target_week":
                    row[
                        f"future_week_"
                        f"{h}_date"
                    ],
                "horizon_week": h,
                "target":
                    row["target"],
                "predicted_1000kg":
                    float(
                        max(0, value)
                    ),
            })

    path = (
        ROOT
        / "tft_by_elevation_predictions.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=
                predictions[0].keys(),
        )
        writer.writeheader()
        writer.writerows(
            predictions
        )

    result = compare(
        windows,
        predictions,
    )

    result[
        "training_scales"
    ] = stats

    result["checkpoint"] = str(
        checkpoint.best_model_path
    )

    (
        ROOT
        / "normalization_ablation.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()