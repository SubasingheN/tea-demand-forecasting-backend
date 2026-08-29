from fastapi import APIRouter

from app.services.training_service import (
    prepare_training_pipeline,
    run_training_test,
    run_training_pipeline,
)
from app.services.evaluation_service import (
    load_trained_model,
    prepare_test_dataset,
    generate_test_predictions,
    evaluate_trained_model,
)
from app.services.forecast_service import (
    generate_quantile_forecasts,
)
from app.services.weekly_forecast_service import (
    generate_future_quantile_predictions,
    format_future_forecasts,
)

router = APIRouter()


@router.get("/training/prepare")
def prepare_training():
    pipeline = prepare_training_pipeline()

    return {
        "status": "success",
        "message": "Training pipeline prepared successfully.",
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
        "training_batches": len(
            pipeline["training_dataloader"]
        ),
        "validation_batches": len(
            pipeline["validation_dataloader"]
        ),
    }


@router.get("/training/test")
def test_training():
    result = run_training_test()

    return result


@router.get("/training/full")
def full_training():
    result = run_training_pipeline()

    return {
        "status": "success",
        "message": "Full TFT training completed successfully.",
        **result,
    }


@router.get("/training/model-load-test")
def test_model_loading():
    model = load_trained_model()

    return {
        "status": "success",
        "message": "Saved TFT model loaded successfully.",
        "model_class": model.__class__.__name__,
    }


@router.get("/training/test-dataset")
def inspect_test_dataset():
    result = prepare_test_dataset()

    return {
        "status": "success",
        "message": "Held-out TFT test dataset created successfully.",
        "test_rows": len(
            result["test_data"]
        ),
        "test_dataset_length": len(
            result["test_dataset"]
        ),
        "first_test_time_idx": int(
            result["test_data"]["time_idx"].min()
        ),
        "last_test_time_idx": int(
            result["test_data"]["time_idx"].max()
        ),
    }


@router.get("/training/prediction-test")
def test_predictions():
    predictions = generate_test_predictions()

    prediction_tensor = predictions.output

    return {
        "status": "success",
        "message": "TFT test predictions generated successfully.",
        "prediction_shape": list(
            prediction_tensor.shape
        ),
    }


@router.get("/training/evaluate")
def evaluate_model():
    result = evaluate_trained_model()

    return result


@router.get("/training/quantile-test")
def test_quantile_predictions():
    forecasts = generate_quantile_forecasts()

    return {
        "status": "success",
        "message": "TFT quantile forecasts generated successfully.",
        "lower_shape": list(
            forecasts["lower"].shape
        ),
        "median_shape": list(
            forecasts["median"].shape
        ),
        "upper_shape": list(
            forecasts["upper"].shape
        ),
        "example": {
            "lower": round(
                float(forecasts["lower"][0][0]),
                2,
            ),
            "median": round(
                float(forecasts["median"][0][0]),
                2,
            ),
            "upper": round(
                float(forecasts["upper"][0][0]),
                2,
            ),
        },
    }


@router.get("/training/future-quantile-test")
def test_future_quantile_predictions():
    predictions = (
        generate_future_quantile_predictions()
    )

    prediction_tensor = predictions.output

    return {
        "status": "success",
        "message": (
            "Future TFT quantile predictions "
            "generated successfully."
        ),
        "prediction_shape": list(
            prediction_tensor.shape
        ),
        "index_rows": len(
            predictions.index
        ),
    }


@router.get("/training/future-forecast-test")
def test_formatted_future_forecasts():
    forecasts = format_future_forecasts()

    return {
        "status": "success",
        "message": (
            "Future weekly demand forecasts "
            "formatted successfully."
        ),
        "tea_type_count": len(
            forecasts
        ),
        "forecasts": forecasts,
    }