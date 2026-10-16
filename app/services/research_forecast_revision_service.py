
from datetime import date, timedelta
import math


FORECAST_WEEKS = 4

QUANTILE_FIELDS = (
    "lower_demand_kg",
    "median_demand_kg",
    "upper_demand_kg",
)


def validate_forecast_run(run):
    """
    Validate the forecast structure needed
    for comparing two research forecast runs.
    """
    if not isinstance(run, dict):
        raise ValueError(
            "Forecast run must be an object."
        )

    origin = date.fromisoformat(
        run["forecast_origin"]
    )

    forecasts = run["forecasts"]

    if not isinstance(forecasts, dict):
        raise ValueError(
            "Forecasts must be grouped by grade."
        )

    normalized = {}

    for grade, weeks in forecasts.items():
        if not isinstance(weeks, list):
            raise ValueError(
                f"{grade}: invalid weekly forecasts."
            )

        if len(weeks) != FORECAST_WEEKS:
            raise ValueError(
                f"{grade}: expected four weeks."
            )

        normalized[grade] = {}

        for item in weeks:
            week = date.fromisoformat(
                item["week"]
            )

            expected_horizon = (
                week - origin
            ).days // 7

            if (
                week.weekday() != 0
                or (week - origin).days % 7 != 0
                or not 1 <= expected_horizon <= 4
            ):
                raise ValueError(
                    f"{grade}: invalid forecast date."
                )

            if item["horizon"] != expected_horizon:
                raise ValueError(
                    f"{grade}: horizon/date mismatch."
                )

            if week in normalized[grade]:
                raise ValueError(
                    f"{grade}: duplicate forecast week."
                )

            values = {}

            for field in QUANTILE_FIELDS:
                value = float(item[field])

                if not math.isfinite(value):
                    raise ValueError(
                        f"{grade}: nonfinite forecast."
                    )

                values[field] = value

            if not (
                0
                <= values["lower_demand_kg"]
                <= values["median_demand_kg"]
                <= values["upper_demand_kg"]
            ):
                raise ValueError(
                    f"{grade}: invalid uncertainty range."
                )

            normalized[grade][week] = {
                "horizon": expected_horizon,
                **values,
            }

    return origin, normalized


def compare_forecast_runs(
    previous_run,
    new_run,
):
    """
    Compare P50 forecasts by matching
    grade and forecast calendar week.

    Both runs must use the selected
    no-price TFT model.
    """
    for run in (previous_run, new_run):
        if run.get("model") != "no_price_tft":
            raise ValueError(
                "Forecast model mismatch."
            )

        if run.get("input_source") == "test_data":
            raise ValueError(
                "Test forecasts cannot be used "
                "in an operational comparison."
            )

    previous_origin, previous = (
        validate_forecast_run(previous_run)
    )

    new_origin, current = (
        validate_forecast_run(new_run)
    )

    if new_origin <= previous_origin:
        raise ValueError(
            "New forecast origin must be later "
            "than the previous forecast origin."
        )

    shared_grades = sorted(
        set(previous) & set(current)
    )

    results = []

    for grade in shared_grades:
        shared_weeks = sorted(
            set(previous[grade])
            & set(current[grade])
        )

        for week in shared_weeks:
            old = previous[grade][week]
            new = current[grade][week]

            old_p50 = old[
                "median_demand_kg"
            ]

            new_p50 = new[
                "median_demand_kg"
            ]

            difference = new_p50 - old_p50

            percentage = (
                difference / old_p50 * 100
                if old_p50 > 0
                else None
            )

            results.append({
                "tea_grade": grade,
                "forecast_week": (
                    week.isoformat()
                ),
                "previous_horizon": old[
                    "horizon"
                ],
                "new_horizon": new[
                    "horizon"
                ],
                "previous_p50_kg": round(
                    old_p50, 2
                ),
                "new_p50_kg": round(
                    new_p50, 2
                ),
                "difference_kg": round(
                    difference, 2
                ),
                "change_percent": (
                    round(percentage, 2)
                    if percentage is not None
                    else None
                ),
                "direction": (
                    "increased"
                    if difference > 0
                    else "decreased"
                    if difference < 0
                    else "unchanged"
                ),
            })

    return {
        "status": "success",
        "comparison_type": (
            "forecast_revision"
        ),
        "model": "no_price_tft",
        "previous_origin": (
            previous_origin.isoformat()
        ),
        "new_origin": (
            new_origin.isoformat()
        ),
        "overlapping_grade_count": len(
            shared_grades
        ),
        "comparison_count": len(results),
        "comparisons": results,
        "message": (
            "Forecast revisions compare predictions "
            "for the same tea grade and calendar week. "
            "They are not forecast accuracy measurements."
        ),
    }
