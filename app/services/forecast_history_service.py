from app.db.forecast_repository import (
    get_all_forecasts,
    generate_forecast_id as
        generate_database_forecast_id,
    save_forecast_to_database,
)


def load_forecast_history() -> list:
    """
    Load all saved forecast records
    from MongoDB Atlas.
    """

    return get_all_forecasts()


def generate_forecast_id(
    history: list = None,
) -> str:
    """
    Generate the next friendly
    forecast ID from MongoDB.

    Example:
    forecast_1
    forecast_2
    forecast_3

    The history parameter is kept
    for compatibility with the
    previous JSON-based service.
    """

    return (
        generate_database_forecast_id()
    )


def save_forecast_history(
    forecasts: list,
    input_week_start_date=None,
    source: str = "research",
) -> dict:
    """
    Save a forecast record to
    MongoDB Atlas.

    source:
    - research
    - operational

    input_week_start_date:
    Date of the weekly input used
    to generate the forecast.
    """

    saved_forecast = (
        save_forecast_to_database(
            forecasts=forecasts,
            input_week_start_date=
                input_week_start_date,
            source=source,
        )
    )

    return saved_forecast


def get_forecast_history() -> list:
    """
    Return all saved forecast
    records from MongoDB Atlas.
    """

    return get_all_forecasts()