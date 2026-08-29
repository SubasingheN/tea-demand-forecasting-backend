import math

import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet

from app.schemas.weekly_input import WeeklyBatchInput
from app.services.data_service import load_dataset
from app.services.preprocessing_service import (
    preprocess_dataset,
)
from app.services.evaluation_service import (
    load_trained_model,
)
from app.services.trend_service import (
    calculate_trend,
)


FORECAST_HORIZON = 4

TEA_TYPES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]


def create_calendar_features(input_date):
    input_date = pd.Timestamp(input_date)

    year = input_date.year
    month = input_date.month

    week_of_year = int(
        input_date.isocalendar().week
    )

    month_sin = math.sin(
        2 * math.pi * month / 12
    )

    month_cos = math.cos(
        2 * math.pi * month / 12
    )

    week_sin = math.sin(
        2 * math.pi * week_of_year / 52
    )

    week_cos = math.cos(
        2 * math.pi * week_of_year / 52
    )

    return {
        "year": year,
        "month": month,
        "week_of_year": week_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
    }


def get_historical_data():
    dataframe = load_dataset()

    processed_dataframe = (
        preprocess_dataset(dataframe)
    )

    processed_dataframe = (
        processed_dataframe.sort_values(
            by=[
                "series_id",
                "time_idx",
            ]
        ).reset_index(drop=True)
    )

    return processed_dataframe


def validate_demo_batch(
    batch_input: WeeklyBatchInput,
):
    received_tea_types = {
        record.tea_type
        for record in batch_input.records
    }

    required_tea_types = set(
        TEA_TYPES
    )

    if received_tea_types != required_tea_types:
        missing = (
            required_tea_types
            - received_tea_types
        )

        extra = (
            received_tea_types
            - required_tea_types
        )

        raise ValueError(
            "Invalid tea types. "
            f"Missing: {sorted(missing)}. "
            f"Extra: {sorted(extra)}."
        )


def create_current_input_rows(
    batch_input: WeeklyBatchInput,
    historical_data: pd.DataFrame,
):
    validate_demo_batch(
        batch_input
    )

    latest_time_idx = int(
        historical_data[
            "time_idx"
        ].max()
    )

    current_time_idx = (
        latest_time_idx + 1
    )

    operational_date = pd.Timestamp(
        batch_input.week_start_date
    )

    calendar_features = (
        create_calendar_features(
            operational_date
        )
    )

    current_rows = []

    for weekly_input in batch_input.records:

        tea_history = historical_data[
            historical_data[
                "series_id"
            ]
            == weekly_input.tea_type
        ].copy()

        tea_history = (
            tea_history.sort_values(
                by="time_idx"
            ).reset_index(drop=True)
        )

        if tea_history.empty:
            raise ValueError(
                "No historical data found for "
                f"{weekly_input.tea_type}."
            )

        auction_price_lag1w = float(
            tea_history[
                "colombo_auction_price_lkr"
            ].iloc[-1]
        )

        if len(tea_history) >= 4:
            auction_price_lag4w = float(
                tea_history[
                    "colombo_auction_price_lkr"
                ].iloc[-4]
            )

        else:
            auction_price_lag4w = (
                auction_price_lag1w
            )

        rolling4_values = (
            tea_history[
                "demand_kg"
            ].tail(3).tolist()
            + [
                weekly_input.demand_kg
            ]
        )

        rolling12_values = (
            tea_history[
                "demand_kg"
            ].tail(11).tolist()
            + [
                weekly_input.demand_kg
            ]
        )

        demand_rolling4w_avg = float(
            sum(rolling4_values)
            / len(rolling4_values)
        )

        demand_rolling12w_avg = float(
            sum(rolling12_values)
            / len(rolling12_values)
        )

        current_rows.append(
            {
                "week_start_date":
                    operational_date,

                "time_idx":
                    current_time_idx,

                "series_id":
                    weekly_input.tea_type,

                "year":
                    calendar_features[
                        "year"
                    ],

                "month":
                    calendar_features[
                        "month"
                    ],

                "week_of_year":
                    calendar_features[
                        "week_of_year"
                    ],

                "month_sin":
                    calendar_features[
                        "month_sin"
                    ],

                "month_cos":
                    calendar_features[
                        "month_cos"
                    ],

                "week_sin":
                    calendar_features[
                        "week_sin"
                    ],

                "week_cos":
                    calendar_features[
                        "week_cos"
                    ],

                "grade":
                    weekly_input.tea_type,

                "demand_kg":
                    float(
                        weekly_input.demand_kg
                    ),

                "production_kg":
                    float(
                        weekly_input.production_kg
                    ),

                "inventory_kg":
                    float(
                        weekly_input.inventory_kg
                    ),

                "colombo_auction_price_lkr":
                    float(
                        weekly_input
                        .colombo_auction_price_lkr
                    ),

                "auction_price_lag1w":
                    auction_price_lag1w,

                "auction_price_lag4w":
                    auction_price_lag4w,

                "rainfall_index":
                    float(
                        weekly_input.rainfall_index
                    ),

                "usd_lkr_rate":
                    float(
                        weekly_input.usd_lkr_rate
                    ),

                "festival_flag":
                    int(
                        weekly_input.festival_flag
                    ),

                "global_commodity_shock":
                    float(
                        weekly_input
                        .global_commodity_shock
                    ),

                "demand_rolling4w_avg":
                    demand_rolling4w_avg,

                "demand_rolling12w_avg":
                    demand_rolling12w_avg,
            }
        )

    return pd.DataFrame(
        current_rows
    )


def create_demo_future_rows(
    batch_input: WeeklyBatchInput,
    historical_data: pd.DataFrame,
    current_input_rows: pd.DataFrame,
):
    current_time_idx = int(
        current_input_rows[
            "time_idx"
        ].max()
    )

    current_operational_date = (
        pd.Timestamp(
            batch_input.week_start_date
        )
    )

    future_rows = []

    for tea_type in TEA_TYPES:

        tea_history = historical_data[
            historical_data[
                "series_id"
            ] == tea_type
        ].copy()

        tea_history = (
            tea_history.sort_values(
                by="time_idx"
            )
        )

        current_record = (
            current_input_rows[
                current_input_rows[
                    "series_id"
                ] == tea_type
            ].iloc[0]
        )

        demand_history = (
            tea_history[
                "demand_kg"
            ].tolist()
            + [
                float(
                    current_record[
                        "demand_kg"
                    ]
                )
            ]
        )

        price_history = (
            tea_history[
                "colombo_auction_price_lkr"
            ].tolist()
            + [
                float(
                    current_record[
                        "colombo_auction_price_lkr"
                    ]
                )
            ]
        )

        for step in range(
            1,
            FORECAST_HORIZON + 1,
        ):

            future_date = (
                current_operational_date
                + pd.Timedelta(
                    weeks=step
                )
            )

            future_time_idx = (
                current_time_idx + step
            )

            calendar_features = (
                create_calendar_features(
                    future_date
                )
            )

            auction_price_lag1w = float(
                price_history[-1]
            )

            if len(price_history) >= 4:
                auction_price_lag4w = float(
                    price_history[-4]
                )

            else:
                auction_price_lag4w = (
                    auction_price_lag1w
                )

            demand_rolling4w_avg = float(
                pd.Series(
                    demand_history
                ).tail(4).mean()
            )

            demand_rolling12w_avg = float(
                pd.Series(
                    demand_history
                ).tail(12).mean()
            )

            future_rows.append(
                {
                    "week_start_date":
                        future_date,

                    "time_idx":
                        future_time_idx,

                    "series_id":
                        tea_type,

                    "year":
                        calendar_features[
                            "year"
                        ],

                    "month":
                        calendar_features[
                            "month"
                        ],

                    "week_of_year":
                        calendar_features[
                            "week_of_year"
                        ],

                    "month_sin":
                        calendar_features[
                            "month_sin"
                        ],

                    "month_cos":
                        calendar_features[
                            "month_cos"
                        ],

                    "week_sin":
                        calendar_features[
                            "week_sin"
                        ],

                    "week_cos":
                        calendar_features[
                            "week_cos"
                        ],

                    "grade":
                        tea_type,

                    "demand_kg":
                        0.0,

                    "production_kg":
                        float(
                            current_record[
                                "production_kg"
                            ]
                        ),

                    "inventory_kg":
                        float(
                            current_record[
                                "inventory_kg"
                            ]
                        ),

                    "colombo_auction_price_lkr":
                        float(
                            current_record[
                                "colombo_auction_price_lkr"
                            ]
                        ),

                    "auction_price_lag1w":
                        auction_price_lag1w,

                    "auction_price_lag4w":
                        auction_price_lag4w,

                    "rainfall_index":
                        float(
                            current_record[
                                "rainfall_index"
                            ]
                        ),

                    "usd_lkr_rate":
                        float(
                            current_record[
                                "usd_lkr_rate"
                            ]
                        ),

                    "festival_flag":
                        0,

                    "global_commodity_shock":
                        float(
                            current_record[
                                "global_commodity_shock"
                            ]
                        ),

                    "demand_rolling4w_avg":
                        demand_rolling4w_avg,

                    "demand_rolling12w_avg":
                        demand_rolling12w_avg,
                }
            )

    return pd.DataFrame(
        future_rows
    )


def create_demo_prediction_dataframe(
    batch_input: WeeklyBatchInput,
):
    historical_data = (
        get_historical_data()
    )

    current_input_rows = (
        create_current_input_rows(
            batch_input,
            historical_data,
        )
    )

    future_rows = (
        create_demo_future_rows(
            batch_input,
            historical_data,
            current_input_rows,
        )
    )

    prediction_dataframe = pd.concat(
        [
            historical_data,
            current_input_rows,
            future_rows,
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

    return (
        prediction_dataframe,
        current_input_rows,
        future_rows,
    )


def create_demo_prediction_dataset(
    batch_input: WeeklyBatchInput,
):
    """
    Create the TFT prediction dataset
    for the latest operational weekly
    input.

    The same dataset is reusable for
    forecasting and explainability.
    """

    (
        prediction_dataframe,
        current_input_rows,
        future_rows,
    ) = create_demo_prediction_dataframe(
        batch_input
    )

    model = load_trained_model()

    future_dataset = (
        TimeSeriesDataSet.from_parameters(
            model.dataset_parameters,
            prediction_dataframe,
            predict=True,
            stop_randomization=True,
        )
    )

    return {
        "future_dataset":
            future_dataset,

        "prediction_dataframe":
            prediction_dataframe,

        "current_input_rows":
            current_input_rows,

        "future_rows":
            future_rows,
    }


def generate_demo_forecast(
    batch_input: WeeklyBatchInput,
):
    dataset_result = (
        create_demo_prediction_dataset(
            batch_input
        )
    )

    future_dataset = dataset_result[
        "future_dataset"
    ]

    current_input_rows = dataset_result[
        "current_input_rows"
    ]

    future_rows = dataset_result[
        "future_rows"
    ]

    model = load_trained_model()

    predictions = model.predict(
        future_dataset,
        mode="quantiles",
        return_index=True,
    )

    quantile_predictions = (
        predictions.output
    )

    prediction_index = (
        predictions.index
    )

    lower_index = 1
    median_index = 3
    upper_index = 5

    results = []

    for sequence_index in range(
        len(prediction_index)
    ):

        tea_type = (
            prediction_index.iloc[
                sequence_index
            ]["series_id"]
        )

        tea_future_rows = (
            future_rows[
                future_rows[
                    "series_id"
                ] == tea_type
            ]
            .sort_values(
                by="time_idx"
            )
            .reset_index(drop=True)
        )

        weekly_forecasts = []
        median_values = []

        for week_position in range(
            FORECAST_HORIZON
        ):

            calendar_row = (
                tea_future_rows.iloc[
                    week_position
                ]
            )

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

    return {
        "input_week_start_date":
            str(
                batch_input.week_start_date
            ),

        "internal_input_time_idx":
            int(
                current_input_rows[
                    "time_idx"
                ].max()
            ),

        "forecast_horizon_weeks":
            FORECAST_HORIZON,

        "tea_type_count":
            len(results),

        "forecasts":
            results,
    }