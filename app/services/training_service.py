from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from app.services.data_service import load_dataset
from app.services.preprocessing_service import preprocess_dataset
from app.services.split_service import (
    split_dataset,
    TRAIN_END_TIME_IDX,
)
from app.services.tft_data_service import (
    create_tft_dataset,
    create_validation_dataset,
    create_dataloaders,
)


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MAX_EPOCHS = 20
TEST_EPOCHS = 1

LEARNING_RATE = 0.03
HIDDEN_SIZE = 16
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 8


def create_tft_model(training_dataset):
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    return model


def create_trainer(
    max_epochs: int = MAX_EPOCHS,
):
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft-best-model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            early_stopping,
            checkpoint_callback,
        ],
        gradient_clip_val=0.1,
        log_every_n_steps=10,
    )

    return trainer


def prepare_training_pipeline():
    dataframe = load_dataset()

    processed_dataframe = preprocess_dataset(
        dataframe
    )

    (
        training_data,
        validation_data,
        test_data,
    ) = split_dataset(
        processed_dataframe
    )

    training_dataset = create_tft_dataset(
        training_data
    )

    validation_start_time_idx = (
        TRAIN_END_TIME_IDX + 1
    )

    validation_context_data = (
        processed_dataframe[
            processed_dataframe["time_idx"]
            <= validation_data["time_idx"].max()
        ].copy()
    )

    validation_dataset = create_validation_dataset(
        training_dataset,
        validation_context_data,
        validation_start_time_idx,
    )

    (
        training_dataloader,
        validation_dataloader,
    ) = create_dataloaders(
        training_dataset,
        validation_dataset,
    )

    return {
        "training_data": training_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "training_dataloader": training_dataloader,
        "validation_dataloader": validation_dataloader,
    }


def train_tft_model(
    training_dataset,
    training_dataloader,
    validation_dataloader,
    max_epochs: int = MAX_EPOCHS,
):
    model = create_tft_model(
        training_dataset
    )

    trainer = create_trainer(
        max_epochs=max_epochs
    )

    trainer.fit(
        model,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )

    best_model_path = (
        trainer.checkpoint_callback.best_model_path
    )

    return {
        "model": model,
        "trainer": trainer,
        "best_model_path": best_model_path,
    }


def run_training_test():
    pipeline = prepare_training_pipeline()

    result = train_tft_model(
        pipeline["training_dataset"],
        pipeline["training_dataloader"],
        pipeline["validation_dataloader"],
        max_epochs=TEST_EPOCHS,
    )

    return {
        "status": "success",
        "message": "One-epoch TFT training test completed.",
        "best_model_path": result[
            "best_model_path"
        ],
    }


def run_training_pipeline():
    pipeline = prepare_training_pipeline()

    training_result = train_tft_model(
        pipeline["training_dataset"],
        pipeline["training_dataloader"],
        pipeline["validation_dataloader"],
        max_epochs=MAX_EPOCHS,
    )

    return {
        "training_rows": len(
            pipeline["training_data"]
        ),
        "validation_rows": len(
            pipeline["validation_data"]
        ),
        "test_rows": len(
            pipeline["test_data"]
        ),
        "training_dataset_length": len(
            pipeline["training_dataset"]
        ),
        "validation_dataset_length": len(
            pipeline["validation_dataset"]
        ),
        "best_model_path": training_result[
            "best_model_path"
        ],
    }