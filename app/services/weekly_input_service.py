import math

import pandas as pd

from app.schemas.weekly_input import (
    WeeklyBatchInput,
    WeeklyTeaInput,
)
from app.services.data_service import load_dataset
from app.services.weekly_forecast_service import (
    get_latest_weekly_history,
)


DATA_FILE_PATH = (
    "data/finalized_tft_weekly_tea_demand_model_data.csv"
)

REQUIRED_TEA_TYPES = {
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
}


def prepare_single_weekly_record(
    weekly_input: WeeklyTeaInput,
    week_start_date,
) -> dict:
    historical_data = get_latest_weekly_history()

    tea_history = historical_data[
        historical_data["series_id"]
        == weekly_input.tea_type
    ].copy()

    if tea_history.empty:
        raise ValueError(
            f"No historical data found for "
            f"{weekly_input.tea_type}."
        )

    tea_history = tea_history.sort_values(
        by="time_idx"
    ).reset_index(drop=True)

    latest_time_idx = int(
        tea_history["time_idx"].max()
    )

    latest_date = pd.to_datetime(
        tea_history["week_start_date"].max()
    )

    input_date = pd.Timestamp(
        week_start_date
    )

    expected_date = (
        latest_date
        + pd.Timedelta(weeks=1)
    )

    if input_date != expected_date:
        raise ValueError(
            f"{weekly_input.tea_type} input must be for "
            f"{expected_date.date()}."
        )

    new_time_idx = latest_time_idx + 1

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

    demand_rolling4w_avg = float(
        tea_history[
            "demand_kg"
        ].tail(4).mean()
    )

    demand_rolling12w_avg = float(
        tea_history[
            "demand_kg"
        ].tail(12).mean()
    )

    return {
        "week_start_date": input_date,
        "time_idx": new_time_idx,
        "series_id": weekly_input.tea_type,
        "year": year,
        "month": month,
        "week_of_year": week_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "grade": weekly_input.tea_type,
        "demand_kg": weekly_input.demand_kg,
        "production_kg": weekly_input.production_kg,
        "inventory_kg": weekly_input.inventory_kg,
        "colombo_auction_price_lkr":
            weekly_input.colombo_auction_price_lkr,
        "auction_price_lag1w":
            auction_price_lag1w,
        "auction_price_lag4w":
            auction_price_lag4w,
        "rainfall_index":
            weekly_input.rainfall_index,
        "usd_lkr_rate":
            weekly_input.usd_lkr_rate,
        "festival_flag":
            weekly_input.festival_flag,
        "global_commodity_shock":
            weekly_input.global_commodity_shock,
        "demand_rolling4w_avg":
            demand_rolling4w_avg,
        "demand_rolling12w_avg":
            demand_rolling12w_avg,
    }


def validate_weekly_batch(
    batch_input: WeeklyBatchInput,
):
    tea_types = [
        record.tea_type
        for record in batch_input.records
    ]

    if len(tea_types) != 5:
        raise ValueError(
            "Weekly batch must contain exactly "
            "5 tea-type records."
        )

    if len(set(tea_types)) != 5:
        raise ValueError(
            "Each tea type must appear exactly once."
        )

    if set(tea_types) != REQUIRED_TEA_TYPES:
        missing = (
            REQUIRED_TEA_TYPES
            - set(tea_types)
        )

        extra = (
            set(tea_types)
            - REQUIRED_TEA_TYPES
        )

        raise ValueError(
            f"Invalid weekly batch. "
            f"Missing: {sorted(missing)}. "
            f"Extra: {sorted(extra)}."
        )


def prepare_weekly_batch(
    batch_input: WeeklyBatchInput,
):
    validate_weekly_batch(
        batch_input
    )

    prepared_records = []

    for record in batch_input.records:
        prepared_record = (
            prepare_single_weekly_record(
                record,
                batch_input.week_start_date,
            )
        )

        prepared_records.append(
            prepared_record
        )

    return prepared_records


def save_weekly_batch(
    batch_input: WeeklyBatchInput,
):
    prepared_records = (
        prepare_weekly_batch(
            batch_input
        )
    )

    dataframe = load_dataset()

    new_rows = pd.DataFrame(
        prepared_records
    )

    updated_dataframe = pd.concat(
        [
            dataframe,
            new_rows,
        ],
        ignore_index=True,
    )

    updated_dataframe = (
        updated_dataframe.sort_values(
            by=[
                "series_id",
                "time_idx",
            ]
        ).reset_index(drop=True)
    )

    updated_dataframe.to_csv(
        DATA_FILE_PATH,
        index=False,
    )

    return prepared_records