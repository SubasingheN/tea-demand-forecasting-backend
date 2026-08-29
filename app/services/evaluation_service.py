from pathlib import Path

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from pytorch_forecasting import (
    TemporalFusionTransformer,
)

from app.services.data_service import load_dataset
from app.services.preprocessing_service import (
    preprocess_dataset,
)
from app.services.split_service import (
    split_dataset,
    VALIDATION_END_TIME_IDX,
)
from app.services.tft_data_service import (
    create_tft_dataset,
    create_test_dataset,
)


MODEL_PATH = Path(
    "models/tft-best-model-v1.ckpt"
)


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = (
        TemporalFusionTransformer.load_from_checkpoint(
            MODEL_PATH
        )
    )

    return model


def prepare_test_dataset():
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

    test_start_time_idx = (
        VALIDATION_END_TIME_IDX + 1
    )

    test_dataset = create_test_dataset(
        training_dataset,
        processed_dataframe,
        test_start_time_idx,
    )

    return {
        "test_data": test_data,
        "test_dataset": test_dataset,
    }


def calculate_regression_metrics(
    actual_values,
    predicted_values,
):
    actual = np.asarray(
        actual_values
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values
    ).reshape(-1)

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5

    r2 = r2_score(
        actual,
        predicted,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def generate_test_predictions():
    model = load_trained_model()

    test_result = prepare_test_dataset()

    test_dataset = test_result[
        "test_dataset"
    ]

    test_dataloader = (
        test_dataset.to_dataloader(
            train=False,
            batch_size=32,
            num_workers=0,
        )
    )

    predictions = model.predict(
        test_dataloader,
        mode="prediction",
        return_x=True,
    )

    return predictions


def evaluate_trained_model():
    predictions = generate_test_predictions()

    predicted_values = (
        predictions.output
        .detach()
        .cpu()
        .numpy()
    )

    actual_values = (
        predictions.x["decoder_target"]
        .detach()
        .cpu()
        .numpy()
    )

    metrics = calculate_regression_metrics(
        actual_values,
        predicted_values,
    )

    return {
        "status": "success",
        "message": "TFT model evaluated successfully.",
        **metrics,
    }