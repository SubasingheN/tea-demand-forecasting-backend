import pandas as pd


def preprocess_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Work with a copy so the original dataset is not modified
    df = dataframe.copy()

    # Convert mixed date formats safely
    df["week_start_date"] = pd.to_datetime(
        df["week_start_date"],
        format="mixed",
    )

    # Sort each tea series chronologically
    df = df.sort_values(
        by=["series_id", "week_start_date"]
    ).reset_index(drop=True)

    # Recalculate the 4-week rolling demand average
    # using ONLY previous weeks.
    df["demand_rolling4w_avg"] = (
        df.groupby("series_id")["demand_kg"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window=4,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    # Recalculate the 12-week rolling demand average
    # using ONLY previous weeks.
    df["demand_rolling12w_avg"] = (
        df.groupby("series_id")["demand_kg"]
        .transform(
            lambda series: (
                series.shift(1)
                .rolling(
                    window=12,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    # The very first week of each tea series has no previous demand.
    # Fill only those initial values with that row's historical demand.
    df["demand_rolling4w_avg"] = (
        df["demand_rolling4w_avg"]
        .fillna(df["demand_kg"])
    )

    df["demand_rolling12w_avg"] = (
        df["demand_rolling12w_avg"]
        .fillna(df["demand_kg"])
    )

    return df