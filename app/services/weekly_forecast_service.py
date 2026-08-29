import math

import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

from app.services.data_service import load_dataset
from app.services.preprocessing_service import preprocess_dataset
from app.services.evaluation_service import load_trained_model
from app.services.trend_service import calculate_trend


TEA_TYPES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]

FORECAST_HORIZON = 4


def get_latest_weekly_history() -> pd.DataFrame:
    dataframe = load_dataset()

    processed_dataframe = preprocess_dataset(
        dataframe
    )

    processed_dataframe = processed_dataframe.sort_values(
        by=["series_id", "time_idx"]
    ).reset_index(drop=True)

    return processed_dataframe


def get_latest_week_information():
    dataframe = get_latest_weekly_history()

    latest_time_idx = int(
        dataframe["time_idx"].max()
    )

    latest_date = dataframe[
        "week_start_date"
    ].max()

    available_tea_types = sorted(
        dataframe["series_id"]
        .unique()
        .tolist()
    )

    return {
        "latest_time_idx": latest_time_idx,
        "latest_date": latest_date,
        "tea_types": available_tea_types,
        "forecast_horizon": FORECAST_HORIZON,
    }


def create_future_calendar_rows() -> pd.DataFrame:
    latest_information = get_latest_week_information()

    latest_time_idx = latest_information[
        "latest_time_idx"
    ]

    latest_date = latest_information[
        "latest_date"
    ]

    future_rows = []

    for tea_type in TEA_TYPES:
        for step in range(
            1,
            FORECAST_HORIZON + 1,
        ):
            future_date = (
                latest_date
                + pd.Timedelta(
                    weeks=step
                )
            )

            future_time_idx = (
                latest_time_idx + step
            )

            year = future_date.year
            month = future_date.month

            week_of_year = int(
                future_date.isocalendar().week
            )

            month_sin = math.sin(
                2 * math.pi * month / 12
            )

            month_cos = math.cos(
                2 * math.pi * month / 12
            )

            week_sin = math.sin(
                2
                * math.pi
                * week_of_year
                / 52
            )

            week_cos = math.cos(
                2
                * math.pi
                * week_of_year
                / 52
            )

            future_rows.append(
                {
                    "week_start_date": future_date,
                    "time_idx": future_time_idx,
                    "series_id": tea_type,
                    "grade": tea_type,
                    "year": year,
                    "month": month,
                    "week_of_year": week_of_year,
                    "month_sin": month_sin,
                    "month_cos": month_cos,
                    "week_sin": week_sin,
                    "week_cos": week_cos,
                    "festival_flag": 0,
                }
            )

    return pd.DataFrame(
        future_rows
    )


def create_prediction_dataframe() -> pd.DataFrame:
    historical_data = get_latest_weekly_history()

    future_data = create_future_calendar_rows()

    prepared_future_rows = []

    for _, future_row in future_data.iterrows():
        tea_type = future_row["series_id"]

        tea_history = historical_data[
            historical_data["series_id"]
            == tea_type
        ].sort_values(
            by="time_idx"
        )

        latest_record = tea_history.iloc[-1]

        prepared_future_rows.append(
            {
                "week_start_date":
                    future_row["week_start_date"],

                "time_idx":
                    int(future_row["time_idx"]),

                "series_id":
                    tea_type,

                "year":
                    int(future_row["year"]),

                "month":
                    int(future_row["month"]),

                "week_of_year":
                    int(future_row["week_of_year"]),

                "month_sin":
                    float(future_row["month_sin"]),

                "month_cos":
                    float(future_row["month_cos"]),

                "week_sin":
                    float(future_row["week_sin"]),

                "week_cos":
                    float(future_row["week_cos"]),

                "grade":
                    tea_type,

                # Future demand is unknown.
                # This is only a required placeholder.
                "demand_kg":
                    0.0,

                "production_kg":
                    float(
                        latest_record[
                            "production_kg"
                        ]
                    ),

                "inventory_kg":
                    float(
                        latest_record[
                            "inventory_kg"
                        ]
                    ),

                "colombo_auction_price_lkr":
                    float(
                        latest_record[
                            "colombo_auction_price_lkr"
                        ]
                    ),

                "auction_price_lag1w":
                    float(
                        latest_record[
                            "colombo_auction_price_lkr"
                        ]
                    ),

                "auction_price_lag4w":
                    float(
                        tea_history[
                            "colombo_auction_price_lkr"
                        ].tail(4).iloc[0]
                    ),

                "rainfall_index":
                    float(
                        latest_record[
                            "rainfall_index"
                        ]
                    ),

                "usd_lkr_rate":
                    float(
                        latest_record[
                            "usd_lkr_rate"
                        ]
                    ),

                "festival_flag":
                    int(
                        future_row[
                            "festival_flag"
                        ]
                    ),

                "global_commodity_shock":
                    float(
                        latest_record[
                            "global_commodity_shock"
                        ]
                    ),

                "demand_rolling4w_avg":
                    float(
                        tea_history[
                            "demand_kg"
                        ].tail(4).mean()
                    ),

                "demand_rolling12w_avg":
                    float(
                        tea_history[
                            "demand_kg"
                        ].tail(12).mean()
                    ),
            }
        )

    prepared_future_dataframe = pd.DataFrame(
        prepared_future_rows
    )

    prediction_dataframe = pd.concat(
        [
            historical_data,
            prepared_future_dataframe,
        ],
        ignore_index=True,
    )

    prediction_dataframe = (
        prediction_dataframe.sort_values(
            by=[
                "series_id",
                "time_idx",
            ]
        ).reset_index(drop=True)
    )

    return prediction_dataframe


def create_future_prediction_dataset():
    model = load_trained_model()

    prediction_dataframe = (
        create_prediction_dataframe()
    )

    dataset_parameters = (
        model.dataset_parameters
    )

    future_dataset = (
        TimeSeriesDataSet.from_parameters(
            dataset_parameters,
            prediction_dataframe,
            predict=True,
            stop_randomization=True,
        )
    )

    return future_dataset


def generate_future_quantile_predictions():
    model = load_trained_model()

    future_dataset = (
        create_future_prediction_dataset()
    )

    predictions = model.predict(
        future_dataset,
        mode="quantiles",
        return_index=True,
    )

    return predictions


def format_future_forecasts():
    predictions = (
        generate_future_quantile_predictions()
    )

    quantile_predictions = predictions.output
    prediction_index = predictions.index

    # QuantileLoss default quantiles:
    # [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
    #
    # We use:
    # 0.10 -> lower forecast
    # 0.50 -> median forecast
    # 0.90 -> upper forecast

    lower_index = 1
    median_index = 3
    upper_index = 5

    future_calendar = (
        create_future_calendar_rows()
    )

    results = []

    for sequence_index in range(
        len(prediction_index)
    ):
        tea_type = prediction_index.iloc[
            sequence_index
        ]["series_id"]

        tea_calendar = future_calendar[
            future_calendar["series_id"]
            == tea_type
        ].sort_values(
            by="time_idx"
        )

        weekly_forecasts = []

        median_values = []

        for week_position in range(
            FORECAST_HORIZON
        ):
            calendar_row = tea_calendar.iloc[
                week_position
            ]

            lower = float(
                quantile_predictions[
                    sequence_index,
                    week_position,
                    lower_index,
                ]
            )

            median = float(
                quantile_predictions[
                    sequence_index,
                    week_position,
                    median_index,
                ]
            )

            upper = float(
                quantile_predictions[
                    sequence_index,
                    week_position,
                    upper_index,
                ]
            )

            median_values.append(
                median
            )

            weekly_forecasts.append(
                {
                    "week":
                        week_position + 1,

                    "week_start_date":
                        calendar_row[
                            "week_start_date"
                        ].strftime(
                            "%Y-%m-%d"
                        ),

                    "time_idx":
                        int(
                            calendar_row[
                                "time_idx"
                            ]
                        ),

                    "lower_demand_kg":
                        round(
                            lower,
                            2,
                        ),

                    "median_demand_kg":
                        round(
                            median,
                            2,
                        ),

                    "upper_demand_kg":
                        round(
                            upper,
                            2,
                        ),

                    "demand_range":
                        (
                            f"{round(lower):,} kg - "
                            f"{round(upper):,} kg"
                        ),
                }
            )

        trend = calculate_trend(
            median_values
        )

        results.append(
            {
                "tea_type":
                    tea_type,

                "trend":
                    trend,

                "forecasts":
                    weekly_forecasts,
            }
        )

    return results