from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/finalized_tft_weekly_tea_demand_model_data.csv")


def load_dataset() -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_FILE)

    return dataframe