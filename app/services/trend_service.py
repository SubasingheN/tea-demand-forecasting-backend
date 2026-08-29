def calculate_trend(
    median_forecasts: list[float],
    stability_threshold: float = 0.02,
) -> str:
    if len(median_forecasts) < 2:
        return "Stable"

    first_value = median_forecasts[0]
    last_value = median_forecasts[-1]

    if first_value == 0:
        return "Stable"

    percentage_change = (
        last_value - first_value
    ) / first_value

    if percentage_change > stability_threshold:
        return "Increasing"

    if percentage_change < -stability_threshold:
        return "Decreasing"

    return "Stable"