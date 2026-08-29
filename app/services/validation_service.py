REQUIRED_COLUMNS = [
    "week_start_date",
    "time_idx",
    "series_id",
    "year",
    "month",
    "week_of_year",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "grade",
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "colombo_auction_price_lkr",
    "auction_price_lag1w",
    "auction_price_lag4w",
    "rainfall_index",
    "usd_lkr_rate",
    "festival_flag",
    "global_commodity_shock",
    "demand_rolling4w_avg",
    "demand_rolling12w_avg",
]

ALLOWED_GRADES = [
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]

def validate_dataset(dataframe):
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        return {
            "valid": False,
            "message": "Dataset is missing required columns.",
            "missing_columns": missing_columns,
        }

    invalid_grades = sorted(
        set(dataframe["grade"].dropna().unique()) - set(ALLOWED_GRADES)
    )

    if invalid_grades:
        return {
            "valid": False,
            "message": "Dataset contains invalid tea grades.",
            "invalid_grades": invalid_grades,
        }

    return {
        "valid": True,
        "message": "Dataset structure is valid.",
    }