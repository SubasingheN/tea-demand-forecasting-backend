import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


TARGET_COLUMN = "demand_kg"

GROUP_IDS = ["series_id"]

STATIC_CATEGORICALS = [
    "series_id",
]

TIME_VARYING_KNOWN_REALS = [
    "time_idx",
    "year",
    "month",
    "week_of_year",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "festival_flag",
]

TIME_VARYING_UNKNOWN_REALS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "colombo_auction_price_lkr",
    "auction_price_lag1w",
    "auction_price_lag4w",
    "rainfall_index",
    "usd_lkr_rate",
    "global_commodity_shock",
    "demand_rolling4w_avg",
    "demand_rolling12w_avg",
]

MAX_ENCODER_LENGTH = 52
MAX_PREDICTION_LENGTH = 4

BATCH_SIZE = 32


def prepare_tft_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    df["series_id"] = df["series_id"].astype(str)

    df = df.sort_values(
        by=["series_id", "time_idx"]
    ).reset_index(drop=True)

    return df


def create_tft_dataset(
    dataframe: pd.DataFrame,
) -> TimeSeriesDataSet:
    df = prepare_tft_dataframe(dataframe)

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=GROUP_IDS,

        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

        static_categoricals=STATIC_CATEGORICALS,

        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,

        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,

        target_normalizer=GroupNormalizer(
            groups=GROUP_IDS,
            transformation=None,
        ),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

        allow_missing_timesteps=False,
    )

    return dataset


def create_validation_dataset(
    training_dataset: TimeSeriesDataSet,
    full_dataframe: pd.DataFrame,
    validation_start_time_idx: int,
) -> TimeSeriesDataSet:
    validation_df = prepare_tft_dataframe(
        full_dataframe
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_df,
        min_prediction_idx=validation_start_time_idx,
        stop_randomization=True,
    )

    return validation_dataset


def create_test_dataset(
    training_dataset: TimeSeriesDataSet,
    full_dataframe: pd.DataFrame,
    test_start_time_idx: int,
) -> TimeSeriesDataSet:
    test_df = prepare_tft_dataframe(
        full_dataframe
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_df,
        min_prediction_idx=test_start_time_idx,
        stop_randomization=True,
    )

    return test_dataset


def create_dataloaders(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
):
    training_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    validation_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    return (
        training_dataloader,
        validation_dataloader,
    )