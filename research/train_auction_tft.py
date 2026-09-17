"""Exploratory TFT for observed eight-week auction windows.

Run after weekly_auction.py and tft_readiness.py; requires the project's
torch, lightning and pytorch-forecasting dependencies. No synthetic grade
data, notebook figures or future sale prices are used as model inputs.
"""

import csv
import json
import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss


ROOT = Path(__file__).resolve().parent / "outputs"
HISTORY = 8
HORIZON = 4
SEED = 42


def read_windows():
    with (ROOT / "tft_8week_windows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not all(r["split"] in ("train", "validation", "test") for r in rows):
        raise ValueError("Run research/tft_readiness.py first")
    return rows


def make_frames(rows):
    records = {split: [] for split in ("train", "validation", "test")}
    for row in rows:
        # Distinct groups prevent the library from joining unrelated windows.
        window_id = f"{row['split']}:{row['target']}:{row['origin_week']}"
        elevation = row["target"].replace("_1000kg", "")
        for offset in range(HISTORY + HORIZON):
            value = (row[f"history_week_{offset + 1}_1000kg"] if offset < HISTORY
                     else row[f"future_week_{offset - HISTORY + 1}_1000kg"])
            records[row["split"]].append({
                "window_id": window_id, "elevation": elevation,
                "time_idx": offset, "sale_1000kg": float(value),
            })
    return {split: pd.DataFrame(values) for split, values in records.items()}


def dataset(frame, reference=None, predict=False):
    if reference is not None:
        return TimeSeriesDataSet.from_dataset(
            reference, frame, predict=predict, stop_randomization=True)
    return TimeSeriesDataSet(
        frame, time_idx="time_idx", target="sale_1000kg",
        group_ids=["window_id"],
        min_encoder_length=HISTORY, max_encoder_length=HISTORY,
        min_prediction_length=HORIZON, max_prediction_length=HORIZON,
        static_categoricals=["elevation"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["sale_1000kg"],
        categorical_encoders={"__group_id__window_id": NaNLabelEncoder(add_nan=True)},
        target_normalizer=TorchNormalizer(method="standard"),
        allow_missing_timesteps=False,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=False,
    )


def main():
    pl.seed_everything(SEED, workers=True)
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    rows = read_windows()
    frames = make_frames(rows)
    train = dataset(frames["train"])
    validation_parts = []
    for row in rows:
        if row["split"] != "validation":
            continue
        window_id = f"validation:{row['target']}:{row['origin_week']}"
        frame = frames["validation"].loc[
            frames["validation"]["window_id"] == window_id].copy()
        part = dataset(frame, train, predict=True)
        if len(part) != 1:
            raise ValueError(f"Expected one validation window for {window_id}")
        validation_parts.append(part)
    if len(train) != sum(r["split"] == "train" for r in rows):
        raise ValueError("Training window count differs from source")
    if len(validation_parts) != sum(r["split"] == "validation" for r in rows):
        raise ValueError("Validation window count differs from source")
    model = TemporalFusionTransformer.from_dataset(
        train, learning_rate=0.001, hidden_size=8, attention_head_size=2,
        dropout=0.1, hidden_continuous_size=4,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        optimizer="adam", log_interval=-1,
    )
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename="auction-tft",
                                 monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=30, accelerator="auto", devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min"), checkpoint],
        gradient_clip_val=0.1, deterministic=True, logger=False,
        enable_model_summary=False,
    )
    validation_loader = DataLoader(ConcatDataset(validation_parts), batch_size=16,
                                   shuffle=False, num_workers=0,
                                   collate_fn=TimeSeriesDataSet._collate_fn)
    trainer.fit(model, train.to_dataloader(train=True, batch_size=16, num_workers=0),
                validation_loader)
    best = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    predictions = []
    for row in rows:
        if row["split"] != "test":
            continue
        window_id = f"test:{row['target']}:{row['origin_week']}"
        frame = frames["test"].loc[frames["test"]["window_id"] == window_id].copy()
        test_dataset = dataset(frame, train, predict=True)
        if len(test_dataset) != 1:
            raise ValueError(f"Expected exactly one forecast for {window_id}")
        # Decoder targets exist for retrospective scoring but are unknown to
        # the decoder at inference. Prediction sees the eight-week encoder.
        output = best.predict(test_dataset.to_dataloader(train=False, batch_size=1,
                                                         num_workers=0), mode="prediction")
        values = output.detach().cpu().numpy().reshape(-1)
        if len(values) != HORIZON:
            raise ValueError(f"Expected four forecast values for {window_id}")
        for h, value in enumerate(values, 1):
            predictions.append({
                "origin_week": row["origin_week"],
                "target_week": row[f"future_week_{h}_date"],
                "horizon_week": h, "target": row["target"],
                "predicted_1000kg": float(max(0, value)),
            })
    path = ROOT / "tft_predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    metadata = {"seed": SEED, "encoder_weeks": HISTORY, "horizon_weeks": HORIZON,
                "train_windows": len(train), "validation_windows": len(validation_parts),
                "test_windows": len(predictions) // HORIZON,
                "torch_version": torch.__version__, "checkpoint": str(checkpoint.best_model_path),
                "test_predictions": str(path)}
    (ROOT / "tft_training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    from score_tft import score
    print(json.dumps(score(path), indent=2))


if __name__ == "__main__":
    main()
