
"""
Historical data service for the six-input TFT.

Combines the protected factory CSV with newer
completed-week MongoDB observations.

Each tea grade is treated independently when
checking its 52-week forecasting history.

No missing observations are generated.
No model checkpoints or datasets are modified.
"""

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

from app.db.mongodb import get_database
from app.schemas.six_input_weekly import (
    SixInputWeeklyBatch,
)


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import build_tft_six_input_datasets as builder
import tft_six_input_config as config


EXTENDED_DATA_FILE = (
    ROOT
    / "data"
    / (
        "finalized_tft_weekly_tea_demand_model_data_"
        "extended_to_2026-10-05.csv"
    )
)

OBSERVATION_COLLECTION = (
    "six_input_factory_weekly_observations"
)

ENCODER_LENGTH = config.ENCODER_LENGTH
PREDICTION_LENGTH = config.PREDICTION_LENGTH

HISTORICAL_ONLY_REALS = list(
    config.HISTORICAL_ONLY_REALS
)

KNOWN_FUTURE_REALS = list(
    config.KNOWN_FUTURE_REALS
)

TEA_GRADES = list(config.TEA_GRADES)

INPUT_FIELDS = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_rate",
    "festival_flag",
    "colombo_auction_price_lkr",
]

MODEL_COLUMNS = list(
    dict.fromkeys(
        [
            "week_start_date",
            "grade",
            *KNOWN_FUTURE_REALS,
            *HISTORICAL_ONLY_REALS,
        ]
    )
)

NUMERIC_COLUMNS = list(
    dict.fromkeys(
        KNOWN_FUTURE_REALS
        + HISTORICAL_ONLY_REALS
    )
)


def load_original_factory_history():
    """
    Read the original protected CSV with the
    preprocessing used during TFT training.
    """

    if not EXTENDED_DATA_FILE.is_file():
        raise FileNotFoundError(
            "Final factory dataset not found: "
            f"{EXTENDED_DATA_FILE}"
        )

    with patch.object(
        config,
        "DATA_FILE",
        EXTENDED_DATA_FILE,
    ), patch.object(
        builder,
        "DATA_FILE",
        EXTENDED_DATA_FILE,
    ):
        history = builder.load_six_input_data().copy()

    history["week_start_date"] = (
        pd.to_datetime(
            history["week_start_date"],
            errors="raise",
        ).dt.normalize()
    )

    history["grade"] = history["grade"].astype(str)

    return history.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)


def load_mongodb_factory_observations():
    """
    Read all newly saved factory observations.

    Database access is read-only.
    """

    collection = get_database()[
        OBSERVATION_COLLECTION
    ]

    projection = {
        "_id": 0,
        "week_start_date": 1,
        "tea_type": 1,
        **{
            field: 1
            for field in INPUT_FIELDS
        },
    }

    documents = list(
        collection.find({}, projection)
    )

    columns = [
        "week_start_date",
        "grade",
        *INPUT_FIELDS,
    ]

    if not documents:
        return pd.DataFrame(columns=columns)

    records = pd.DataFrame(documents)

    required = [
        "week_start_date",
        "tea_type",
        *INPUT_FIELDS,
    ]

    missing = [
        field
        for field in required
        if field not in records.columns
    ]

    if missing:
        raise ValueError(
            "MongoDB observations are missing: "
            + ", ".join(missing)
        )

    records["week_start_date"] = (
        pd.to_datetime(
            records["week_start_date"],
            errors="raise",
        ).dt.normalize()
    )

    records["grade"] = (
        records["tea_type"].astype(str)
    )

    if not records["grade"].isin(
        TEA_GRADES
    ).all():
        raise ValueError(
            "Unsupported tea grade in MongoDB history."
        )

    if (
        records["week_start_date"]
        .dt.weekday.ne(0).any()
    ):
        raise ValueError(
            "MongoDB reporting weeks must start "
            "on Monday."
        )

    if records.duplicated(
        ["week_start_date", "grade"]
    ).any():
        raise ValueError(
            "Duplicate MongoDB grade/week records."
        )

    for field in INPUT_FIELDS:
        records[field] = pd.to_numeric(
            records[field],
            errors="raise",
        )

    if not np.isfinite(
        records[INPUT_FIELDS].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "MongoDB contains non-finite inputs."
        )

    for field in (
        "demand_kg",
        "usd_lkr_rate",
        "colombo_auction_price_lkr",
    ):
        if (records[field] <= 0).any():
            raise ValueError(
                f"Invalid MongoDB {field}: "
                "must be positive."
            )

    for field in (
        "production_kg",
        "inventory_kg",
    ):
        if (records[field] < 0).any():
            raise ValueError(
                f"Invalid MongoDB {field}: "
                "cannot be negative."
            )

    if not records[
        "festival_flag"
    ].isin([0, 1]).all():
        raise ValueError(
            "Invalid MongoDB festival_flag."
        )

    return records[columns].sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)


def build_calendar_features(dataframe, first_week):
    """
    Calculate calendar features using the
    frozen TFT's preprocessing conventions.
    """

    result = dataframe.copy()

    dates = pd.to_datetime(
        result["week_start_date"],
        errors="raise",
    )

    origin = pd.Timestamp(first_week)

    result["time_idx"] = (
        (dates - origin).dt.days // 7
    ).astype(int)

    weeks = (
        dates.dt.isocalendar()
        .week.astype(int)
    )

    months = dates.dt.month

    result["week_sin"] = np.sin(
        2 * np.pi * weeks / 52.1775
    )

    result["week_cos"] = np.cos(
        2 * np.pi * weeks / 52.1775
    )

    result["month_sin"] = np.sin(
        2 * np.pi * months / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * months / 12
    )

    return result


def verify_shared_weekly_features(dataframe):
    """
    Verify agreement among grades that have
    observations for the same reporting week.

    A week does not need all five grades.
    """

    for week, group in dataframe.groupby(
        "week_start_date"
    ):
        rates = group[
            "usd_lkr_rate"
        ].to_numpy(dtype=float)

        festivals = group[
            "festival_flag"
        ].to_numpy(dtype=float)

        if (
            rates.max() - rates.min()
        ) > 0.001:
            raise ValueError(
                f"Conflicting exchange rates "
                f"for week {week.date()}."
            )

        if len(set(festivals)) > 1:
            raise ValueError(
                f"Conflicting festival flags "
                f"for week {week.date()}."
            )


def validate_completed_mongodb_weeks(saved):
    """
    Reject observations whose reporting week
    has not yet finished.
    """

    today = date.today()

    for timestamp in saved[
        "week_start_date"
    ].drop_duplicates():
        week_start = pd.Timestamp(
            timestamp
        ).date()

        week_end = (
            week_start + timedelta(days=6)
        )

        if week_end >= today:
            raise ValueError(
                "MongoDB contains an unfinished "
                f"reporting week: {week_start}."
            )


def load_six_input_history():
    """
    Return combined CSV + MongoDB factory history.

    Individual grades may have different
    latest recorded weeks.

    Full continuity is validated later for
    the grades selected for forecasting.
    """

    original = load_original_factory_history()

    first_week = pd.Timestamp(
        original["week_start_date"].min()
    )

    csv_latest = pd.Timestamp(
        original["week_start_date"].max()
    )

    saved = load_mongodb_factory_observations()

    if saved.empty:
        return original

    if (
        saved["week_start_date"] <= csv_latest
    ).any():
        raise ValueError(
            "MongoDB records overlap the protected "
            "factory CSV history."
        )

    validate_completed_mongodb_weeks(saved)

    saved = build_calendar_features(
        saved,
        first_week,
    )

    if not set(MODEL_COLUMNS).issubset(
        original.columns
    ):
        raise ValueError(
            "CSV history is missing TFT model fields."
        )

    if not set(MODEL_COLUMNS).issubset(
        saved.columns
    ):
        raise ValueError(
            "MongoDB history is missing TFT fields."
        )

    combined = pd.concat(
        [
            original[MODEL_COLUMNS].copy(),
            saved[MODEL_COLUMNS].copy(),
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)

    if combined.duplicated(
        ["week_start_date", "grade"]
    ).any():
        raise ValueError(
            "Duplicate combined factory observations."
        )

    if not np.isfinite(
        combined[NUMERIC_COLUMNS].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Combined history has invalid model inputs."
        )

    verify_shared_weekly_features(combined)

    return combined


def verify_grade_encoder(history, grade):
    """
    Require exactly 52 consecutive observations
    ending on the selected forecast-origin week.

    Only the selected grade is checked.
    """

    history = history.sort_values(
        "week_start_date"
    ).reset_index(drop=True)

    if len(history) != ENCODER_LENGTH:
        raise ValueError(
            f"{grade}: {ENCODER_LENGTH} consecutive "
            "historical weeks are required."
        )

    actual_dates = pd.DatetimeIndex(
        history["week_start_date"]
    )

    expected_dates = pd.date_range(
        end=actual_dates[-1],
        periods=ENCODER_LENGTH,
        freq="W-MON",
    )

    if (
        actual_dates.has_duplicates
        or not actual_dates.equals(expected_dates)
    ):
        raise ValueError(
            f"{grade}: missing or irregular weekly "
            "observations in the 52-week encoder."
        )

    time_indices = history[
        "time_idx"
    ].to_numpy(dtype=np.int64)

    if not np.all(
        np.diff(time_indices) == 1
    ):
        raise ValueError(
            f"{grade}: nonconsecutive TFT time indices."
        )

    if not np.isfinite(
        history[NUMERIC_COLUMNS].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            f"{grade}: missing or invalid TFT inputs."
        )


def verify_record_against_history(
    record,
    historical_row,
):
    """
    Ensure submitted forecast inputs match
    previously recorded factory observations.
    """

    for field in HISTORICAL_ONLY_REALS:
        submitted = float(
            getattr(record, field)
        )

        recorded = float(
            historical_row[field]
        )

        if not np.isclose(
            submitted,
            recorded,
            rtol=0,
            atol=0.001,
        ):
            raise ValueError(
                f"{record.tea_type}: {field} "
                "does not match the recorded "
                "factory observation."
            )


def prepare_six_input_history(
    batch_input: SixInputWeeklyBatch,
):
    """
    Prepare one 52-week encoder for each
    selected tea grade.

    Other grades are not required to have
    observations for the selected origin.

    Returns the structure expected by the
    existing frozen TFT inference service.
    """

    historical = load_six_input_history()

    requested_date = pd.Timestamp(
        batch_input.week_start_date
    )

    earliest_date = pd.Timestamp(
        historical["week_start_date"].min()
    )

    if requested_date < earliest_date:
        raise ValueError(
            "Requested week precedes the factory "
            "dataset."
        )

    if (
        requested_date + timedelta(days=6)
    ).date() >= date.today():
        raise ValueError(
            "The selected reporting week "
            "has not yet been completed."
        )

    selected_grades = list(
        batch_input.selected_grades
    )

    prepared_groups = []
    latest_per_grade = {}

    for record in batch_input.records:
        grade = record.tea_type

        if grade not in TEA_GRADES:
            raise ValueError(
                f"Unsupported tea grade: {grade}"
            )

        all_grade_history = historical.loc[
            historical["grade"] == grade
        ].sort_values(
            "week_start_date"
        )

        if all_grade_history.empty:
            raise ValueError(
                f"No recorded observations for {grade}."
            )

        latest_per_grade[grade] = pd.Timestamp(
            all_grade_history[
                "week_start_date"
            ].max()
        )

        matching = all_grade_history.loc[
            all_grade_history[
                "week_start_date"
            ] == requested_date
        ]

        if len(matching) != 1:
            raise ValueError(
                f"{grade}: no recorded observation "
                f"for {requested_date.date()}."
            )

        verify_record_against_history(
            record,
            matching.iloc[0],
        )

        encoder = (
            all_grade_history.loc[
                all_grade_history[
                    "week_start_date"
                ] <= requested_date
            ]
            .tail(ENCODER_LENGTH)
            .copy()
        )

        verify_grade_encoder(
            encoder,
            grade,
        )

        prepared_groups.append(
            encoder[MODEL_COLUMNS]
        )

    prepared = pd.concat(
        prepared_groups,
        ignore_index=True,
    ).sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    expected_rows = (
        len(selected_grades)
        * ENCODER_LENGTH
    )

    if len(prepared) != expected_rows:
        raise ValueError(
            "Incorrect selected-grade encoder size."
        )

    forecast_start = (
        requested_date + timedelta(weeks=1)
    )

    forecast_end = (
        requested_date
        + timedelta(weeks=PREDICTION_LENGTH)
    )

    # The most recent observation must exist
    # for every selected grade at this origin.
    latest_for_all_selected = all(
        latest_per_grade[grade] == requested_date
        for grade in selected_grades
    )

    forecast_mode = (
        "latest_completed_week_forecast"
        if latest_for_all_selected
        else "historical_replay"
    )

    # Report latest recorded week among all grades.
    dataset_latest_date = pd.Timestamp(
        historical["week_start_date"].max()
    )

    return {
        "historical_dataframe": prepared,
        "selected_grades": selected_grades,
        "forecast_origin": (
            requested_date.strftime("%Y-%m-%d")
        ),
        "forecast_start_date": (
            forecast_start.strftime("%Y-%m-%d")
        ),
        "forecast_end_date": (
            forecast_end.strftime("%Y-%m-%d")
        ),
        "historical_input_weeks": ENCODER_LENGTH,
        "historical_rows": len(prepared),
        "forecast_horizon_weeks": (
            PREDICTION_LENGTH
        ),
        "forecast_mode": forecast_mode,
        "dataset_latest_week": (
            dataset_latest_date.strftime(
                "%Y-%m-%d"
            )
        ),
        "latest_week_per_selected_grade": {
            grade: latest_per_grade[
                grade
            ].strftime("%Y-%m-%d")
            for grade in selected_grades
        },
    }
