import pandas as pd

from app.db.weekly_input_repository import (
    get_weekly_input_by_date,
)

from app.services.data_service import (
    load_dataset,
)

from app.services.forecast_history_service import (
    get_forecast_history,
)


def get_operational_actual_demand(
    week_start_date: str,
    tea_type: str,
):
    """
    Find actual demand from MongoDB
    operational weekly inputs.
    """

    weekly_input = (
        get_weekly_input_by_date(
            week_start_date
        )
    )

    if weekly_input is None:
        return None

    records = weekly_input.get(
        "records",
        [],
    )

    for record in records:
        if (
            record.get("tea_type")
            == tea_type
        ):
            demand = record.get(
                "demand_kg"
            )

            if demand is not None:
                return float(demand)

    return None


def get_research_actual_demand(
    actual_dataframe,
    week_start_date: str,
    tea_type: str,
):
    """
    Find actual demand from the
    original research dataset.
    """

    forecast_date = pd.Timestamp(
        week_start_date
    )

    actual_row = actual_dataframe[
        (
            actual_dataframe[
                "series_id"
            ] == tea_type
        )
        &
        (
            actual_dataframe[
                "week_start_date"
            ] == forecast_date
        )
    ]

    if actual_row.empty:
        return None

    return float(
        actual_row[
            "demand_kg"
        ].iloc[0]
    )


def compare_forecast_with_actuals(
    forecast_id: str,
) -> dict:
    history = (
        get_forecast_history()
    )

    selected_forecast = next(
        (
            forecast
            for forecast
            in history
            if forecast[
                "forecast_id"
            ] == forecast_id
        ),
        None,
    )

    if selected_forecast is None:
        raise ValueError(
            f"Forecast '{forecast_id}' "
            "was not found."
        )

    actual_dataframe = (
        load_dataset()
    )

    actual_dataframe[
        "week_start_date"
    ] = pd.to_datetime(
        actual_dataframe[
            "week_start_date"
        ],
        format="mixed",
    )

    comparison_results = []

    for tea_forecast in (
        selected_forecast[
            "forecasts"
        ]
    ):
        tea_type = (
            tea_forecast[
                "tea_type"
            ]
        )

        for weekly_forecast in (
            tea_forecast[
                "forecasts"
            ]
        ):
            week_start_date = str(
                weekly_forecast[
                    "week_start_date"
                ]
            )

            # First priority:
            # operational actual demand
            # saved in MongoDB.
            actual_demand = (
                get_operational_actual_demand(
                    week_start_date,
                    tea_type,
                )
            )

            actual_source = None

            if actual_demand is not None:
                actual_source = (
                    "operational_database"
                )

            else:
                # Second priority:
                # original research dataset.
                actual_demand = (
                    get_research_actual_demand(
                        actual_dataframe,
                        week_start_date,
                        tea_type,
                    )
                )

                if actual_demand is not None:
                    actual_source = (
                        "research_dataset"
                    )

            lower_demand = float(
                weekly_forecast[
                    "lower_demand_kg"
                ]
            )

            median_demand = float(
                weekly_forecast[
                    "median_demand_kg"
                ]
            )

            upper_demand = float(
                weekly_forecast[
                    "upper_demand_kg"
                ]
            )

            if actual_demand is None:
                comparison_status = (
                    "Actual demand not available"
                )

                absolute_error = None

            else:
                absolute_error = abs(
                    actual_demand -
                    median_demand
                )

                if (
                    lower_demand
                    <= actual_demand
                    <= upper_demand
                ):
                    comparison_status = (
                        "Within predicted range"
                    )

                elif (
                    actual_demand
                    < lower_demand
                ):
                    comparison_status = (
                        "Below predicted range"
                    )

                else:
                    comparison_status = (
                        "Above predicted range"
                    )

            comparison_results.append(
                {
                    "tea_type":
                        tea_type,

                    "week":
                        weekly_forecast[
                            "week"
                        ],

                    "week_start_date":
                        week_start_date,

                    "predicted_lower_kg":
                        weekly_forecast[
                            "lower_demand_kg"
                        ],

                    "predicted_median_kg":
                        weekly_forecast[
                            "median_demand_kg"
                        ],

                    "predicted_upper_kg":
                        weekly_forecast[
                            "upper_demand_kg"
                        ],

                    "actual_demand_kg":
                        actual_demand,

                    "actual_source":
                        actual_source,

                    "absolute_error_kg":
                        (
                            round(
                                absolute_error,
                                2,
                            )
                            if absolute_error
                            is not None
                            else None
                        ),

                    "comparison_status":
                        comparison_status,
                }
            )

    available_actuals = [
        result
        for result
        in comparison_results
        if result[
            "actual_demand_kg"
        ] is not None
    ]

    within_range_count = sum(
        1
        for result
        in available_actuals
        if result[
            "comparison_status"
        ] ==
        "Within predicted range"
    )

    range_coverage_percentage = (
        (
            within_range_count /
            len(available_actuals)
        ) * 100
        if available_actuals
        else None
    )

    return {
        "forecast_id":
            forecast_id,

        "generated_at":
            selected_forecast[
                "generated_at"
            ],

        "total_forecast_records":
            len(
                comparison_results
            ),

        "actual_records_available":
            len(
                available_actuals
            ),

        "within_range_count":
            within_range_count,

        "range_coverage_percentage":
            (
                round(
                    range_coverage_percentage,
                    2,
                )
                if range_coverage_percentage
                is not None
                else None
            ),

        "comparisons":
            comparison_results,
    }