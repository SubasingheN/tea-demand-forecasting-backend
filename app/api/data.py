from fastapi import APIRouter

from app.services.data_service import load_dataset
from app.services.validation_service import validate_dataset
from app.services.preprocessing_service import preprocess_dataset
from app.services.tft_data_service import (
    prepare_tft_dataframe,
    create_tft_dataset,
)
from app.services.split_service import split_dataset
from app.services.weekly_forecast_service import (
    get_latest_week_information,
    create_future_calendar_rows,
    create_prediction_dataframe,
    create_future_prediction_dataset,
)

router = APIRouter()


@router.get("/data/summary")
def get_data_summary():
    dataframe = load_dataset()

    return {
        "rows": len(dataframe),
        "columns": list(dataframe.columns),
    }


@router.get("/data/validate")
def validate_data():
    dataframe = load_dataset()

    result = validate_dataset(dataframe)

    return result


@router.get("/data/preprocess")
def preprocess_data():
    dataframe = load_dataset()

    processed_dataframe = preprocess_dataset(dataframe)

    return {
        "rows": len(processed_dataframe),
        "missing_values": int(
            processed_dataframe.isnull().sum().sum()
        ),
        "first_date": processed_dataframe[
            "week_start_date"
        ].min(),
        "last_date": processed_dataframe[
            "week_start_date"
        ].max(),
    }


@router.get("/data/tft-ready")
def get_tft_ready_data():
    dataframe = load_dataset()

    processed_dataframe = preprocess_dataset(dataframe)

    tft_dataframe = prepare_tft_dataframe(
        processed_dataframe
    )

    return {
        "rows": len(tft_dataframe),
        "tea_types": sorted(
            tft_dataframe["series_id"]
            .unique()
            .tolist()
        ),
        "target": "demand_kg",
        "first_time_idx": int(
            tft_dataframe["time_idx"].min()
        ),
        "last_time_idx": int(
            tft_dataframe["time_idx"].max()
        ),
    }


@router.get("/data/tft-dataset")
def create_tft_dataset_test():
    dataframe = load_dataset()

    processed_dataframe = preprocess_dataset(dataframe)

    tft_dataset = create_tft_dataset(
        processed_dataframe
    )

    return {
        "status": "success",
        "message": (
            "TFT TimeSeriesDataSet created successfully."
        ),
        "dataset_length": len(tft_dataset),
    }


@router.get("/data/split")
def get_dataset_split():
    dataframe = load_dataset()

    processed_dataframe = preprocess_dataset(dataframe)

    (
        training_data,
        validation_data,
        test_data,
    ) = split_dataset(processed_dataframe)

    return {
        "training_rows": len(training_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "training_last_time_idx": int(
            training_data["time_idx"].max()
        ),
        "validation_last_time_idx": int(
            validation_data["time_idx"].max()
        ),
        "test_last_time_idx": int(
            test_data["time_idx"].max()
        ),
    }


@router.get("/data/latest-week")
def get_latest_week():
    result = get_latest_week_information()

    return {
        "status": "success",
        "latest_time_idx": result["latest_time_idx"],
        "latest_date": result["latest_date"],
        "tea_types": result["tea_types"],
        "forecast_horizon": result[
            "forecast_horizon"
        ],
    }


@router.get("/data/future-calendar")
def get_future_calendar():
    future_dataframe = create_future_calendar_rows()

    return {
        "status": "success",
        "rows": len(future_dataframe),
        "tea_types": sorted(
            future_dataframe["series_id"]
            .unique()
            .tolist()
        ),
        "first_future_time_idx": int(
            future_dataframe["time_idx"].min()
        ),
        "last_future_time_idx": int(
            future_dataframe["time_idx"].max()
        ),
        "first_future_date": future_dataframe[
            "week_start_date"
        ].min(),
        "last_future_date": future_dataframe[
            "week_start_date"
        ].max(),
    }


@router.get("/data/prediction-dataframe")
def get_prediction_dataframe():
    prediction_dataframe = (
        create_prediction_dataframe()
    )

    latest_time_idx = int(
        prediction_dataframe["time_idx"].max()
    )

    latest_history = get_latest_week_information()

    future_rows = prediction_dataframe[
        prediction_dataframe["time_idx"]
        > latest_history["latest_time_idx"]
    ]

    return {
        "status": "success",
        "message": (
            "Prediction dataframe created successfully."
        ),
        "total_rows": len(
            prediction_dataframe
        ),
        "future_rows": len(
            future_rows
        ),
        "latest_time_idx": latest_time_idx,
        "missing_values": int(
            prediction_dataframe
            .isnull()
            .sum()
            .sum()
        ),
    }


@router.get("/data/future-tft-dataset")
def get_future_tft_dataset():
    future_dataset = (
        create_future_prediction_dataset()
    )

    return {
        "status": "success",
        "message": (
            "Future TFT prediction dataset "
            "created successfully."
        ),
        "dataset_length": len(
            future_dataset
        ),
    }