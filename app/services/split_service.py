import pandas as pd


TRAIN_END_TIME_IDX = 155
VALIDATION_END_TIME_IDX = 181


def split_dataset(
    dataframe: pd.DataFrame,
):
    df = dataframe.copy()

    training_data = df[
        df["time_idx"] <= TRAIN_END_TIME_IDX
    ].copy()

    validation_data = df[
        (df["time_idx"] > TRAIN_END_TIME_IDX)
        & (df["time_idx"] <= VALIDATION_END_TIME_IDX)
    ].copy()

    test_data = df[
        df["time_idx"] > VALIDATION_END_TIME_IDX
    ].copy()

    return (
        training_data,
        validation_data,
        test_data,
    )