from app.services.demand_alert_service import (
    generate_demand_alerts,
)


def get_recommendations_for_alert(
    alert_level: str,
) -> list:
    if alert_level == "Low Demand Warning":
        return [
            "Consider reducing production planning.",
            "Review current inventory levels.",
            "Avoid unnecessary stock accumulation.",
            "Review market release quantities.",
        ]

    elif alert_level == "High Demand Warning":
        return [
            "Consider increasing production planning.",
            "Check whether inventory can support demand.",
            "Prepare sufficient stock for market release.",
            "Monitor demand closely for further increases.",
        ]

    else:
        return [
            "Maintain normal production planning.",
            "Continue monitoring weekly demand forecasts.",
        ]


def generate_recommendations_from_alerts(
    demand_alerts: list,
):
    results = []

    for tea_result in demand_alerts:
        tea_type = tea_result["tea_type"]

        weekly_recommendations = []

        for alert in tea_result["alerts"]:
            recommendations = (
                get_recommendations_for_alert(
                    alert["alert_level"]
                )
            )

            weekly_recommendations.append(
                {
                    "week": alert["week"],
                    "week_start_date":
                        alert["week_start_date"],
                    "median_demand_kg":
                        alert["median_demand_kg"],
                    "baseline_demand_kg":
                        alert["baseline_demand_kg"],
                    "percentage_change":
                        alert["percentage_change"],
                    "alert_level":
                        alert["alert_level"],
                    "recommendations":
                        recommendations,
                }
            )

        results.append(
            {
                "tea_type": tea_type,
                "weekly_recommendations":
                    weekly_recommendations,
            }
        )

    return results


def generate_demand_recommendations():
    demand_alerts = (
        generate_demand_alerts()
    )

    return (
        generate_recommendations_from_alerts(
            demand_alerts
        )
    )