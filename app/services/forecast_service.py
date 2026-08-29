from app.services.evaluation_service import (
    load_trained_model,
    prepare_test_dataset,
)


LOWER_QUANTILE_INDEX = 1
MEDIAN_QUANTILE_INDEX = 3
UPPER_QUANTILE_INDEX = 5


def generate_quantile_forecasts():
    model = load_trained_model()

    test_result = prepare_test_dataset()

    test_dataset = test_result["test_dataset"]

    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=32,
        num_workers=0,
    )

    predictions = model.predict(
        test_dataloader,
        mode="quantiles",
        return_x=True,
    )

    quantile_tensor = predictions.output

    lower_forecasts = quantile_tensor[
        :, :, LOWER_QUANTILE_INDEX
    ]

    median_forecasts = quantile_tensor[
        :, :, MEDIAN_QUANTILE_INDEX
    ]

    upper_forecasts = quantile_tensor[
        :, :, UPPER_QUANTILE_INDEX
    ]

    return {
        "lower": lower_forecasts,
        "median": median_forecasts,
        "upper": upper_forecasts,
    }