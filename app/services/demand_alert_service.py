from app.services.weekly_forecast_service import (
    format_future_forecasts,
)
from app.services.weekly_forecast_service import (
    get_latest_weekly_history,
)


LOW_DEMAND_THRESHOLD = -0.15
HIGH_DEMAND_THRESHOLD = 0.15


def get_recent_demand_baseline(
    tea_type: str,
) -> float:
    historical_data = get_latest_weekly_history()

    tea_history = historical_data[
        historical_data["series_id"]
        == tea_type
    ].sort_values(
        by="time_idx"
    )

    recent_average = float(
        tea_history[
            "demand_kg"
        ].tail(12).mean()
    )

    return recent_average


def classify_demand_alert(
    forecast_median: float,
    baseline_demand: float,
) -> dict:
    if baseline_demand == 0:
        return {
            "alert_level": "Normal",
            "percentage_change": 0.0,
        }

    percentage_change = (
        forecast_median
        - baseline_demand
    ) / baseline_demand

    if (
        percentage_change
        <= LOW_DEMAND_THRESHOLD
    ):
        alert_level = (
            "Low Demand Warning"
        )

    elif (
        percentage_change
        >= HIGH_DEMAND_THRESHOLD
    ):
        alert_level = (
            "High Demand Warning"
        )

    else:
        alert_level = "Normal"

    return {
        "alert_level":
            alert_level,

        "percentage_change":
            round(
                percentage_change * 100,
                2,
            ),
    }


def generate_alerts_from_forecasts(
    forecasts: list,
):
    results = []

    for tea_result in forecasts:
        tea_type = tea_result[
            "tea_type"
        ]

        baseline_demand = (
            get_recent_demand_baseline(
                tea_type
            )
        )

        weekly_alerts = []

        for forecast in tea_result[
            "forecasts"
        ]:
            median_forecast = float(
                forecast[
                    "median_demand_kg"
                ]
            )

            alert = (
                classify_demand_alert(
                    median_forecast,
                    baseline_demand,
                )
            )

            weekly_alerts.append(
                {
                    "week":
                        forecast["week"],

                    "week_start_date":
                        forecast[
                            "week_start_date"
                        ],

                    "median_demand_kg":
                        median_forecast,

                    "baseline_demand_kg":
                        round(
                            baseline_demand,
                            2,
                        ),

                    "percentage_change":
                        alert[
                            "percentage_change"
                        ],

                    "alert_level":
                        alert[
                            "alert_level"
                        ],
                }
            )

        results.append(
            {
                "tea_type":
                    tea_type,

                "baseline_demand_kg":
                    round(
                        baseline_demand,
                        2,
                    ),

                "alerts":
                    weekly_alerts,
            }
        )

    return results


def generate_demand_alerts():
    forecasts = (
        format_future_forecasts()
    )

    return (
        generate_alerts_from_forecasts(
            forecasts
        )
    )