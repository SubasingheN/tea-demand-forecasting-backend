
"""
Historical input preparation for the production nine-input TFT.

Historical source:
    tft_eight_input_model_ready_ranges_2021_2026.csv

Supports:
- Historical forecast preparation.
- Latest completed reporting week.
- One to five tea grades.
- Continuous 52-week encoder windows.
- Read-only MongoDB observation extensions.
- Strict verification of saved historical observations.
- Preview forecasting using edited latest-week inputs.

The preview workflow:
- Accepts only the latest completed reporting week.
- Never changes protected historical data.
- Replaces the selected week's values in memory only.
- Validates nine inputs and shared weekly values.
- Produces a complete 52-week inference window.
- Does not write to MongoDB.

No model training or database writes occur here.
"""

import sys

from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.db.mongodb import get_database

from app.schemas.eight_input_weekly import (
    EightInputWeeklyBatch,
)


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from build_eight_input_dataset import (
    load_eight_input_data,
)

from tft_eight_input_config import (
    DATA_FILE,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    TEA_GRADES,
    HISTORICAL_ONLY_REALS,
    KNOWN_FUTURE_REALS,
)


OBSERVATION_COLLECTION = (
    "eight_input_factory_weekly_observations"
)

FACTORY_TIMEZONE = ZoneInfo("Asia/Colombo")

EXCHANGE_FIELDS = [
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
]

AUCTION_FIELDS = [
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]

SHARED_FIELDS = [
    *EXCHANGE_FIELDS,
    "festival_flag",
    *AUCTION_FIELDS,
]

INPUT_FIELDS = list(HISTORICAL_ONLY_REALS)

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
        [
            *KNOWN_FUTURE_REALS,
            *HISTORICAL_ONLY_REALS,
        ]
    )
)


def factory_today():
    """Return today's date in Sri Lanka."""

    return pd.Timestamp.now(
        tz=FACTORY_TIMEZONE
    ).date()


def latest_completed_week_start():
    """Return the Monday of the latest completed week."""

    today = factory_today()

    current_monday = (
        today - timedelta(days=today.weekday())
    )

    return current_monday - timedelta(weeks=1)


def validate_feature_configuration():
    """Check the active production feature definitions."""

    if ENCODER_LENGTH != 52:
        raise ValueError(
            "The production model requires "
            "52 historical weeks."
        )

    if PREDICTION_LENGTH != 4:
        raise ValueError(
            "The production model requires "
            "four forecast weeks."
        )

    if len(INPUT_FIELDS) != 9:
        raise ValueError(
            "The production model requires "
            "nine historical inputs."
        )

    if not set(EXCHANGE_FIELDS).issubset(
        set(INPUT_FIELDS)
    ):
        raise ValueError(
            "Both USD/LKR exchange-rate "
            "features are required."
        )

    if "usd_lkr_rate" in INPUT_FIELDS:
        raise ValueError(
            "The obsolete single exchange-rate "
            "feature is not supported."
        )

    if set(TEA_GRADES) != {
        "BOP",
        "BOPF",
        "Dust",
        "OP",
        "Pekoe",
    }:
        raise ValueError(
            "Unexpected tea-grade configuration."
        )


def load_original_factory_history():
    """Load the protected historical CSV."""

    validate_feature_configuration()

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            "Revised historical dataset missing: "
            f"{DATA_FILE}"
        )

    history = load_eight_input_data().copy()

    missing = sorted(
        set(MODEL_COLUMNS) - set(history.columns)
    )

    if missing:
        raise ValueError(
            "Historical dataset lacks model fields: "
            f"{missing}"
        )

    history["week_start_date"] = (
        pd.to_datetime(
            history["week_start_date"],
            errors="raise",
        ).dt.normalize()
    )

    history["grade"] = history[
        "grade"
    ].astype(str)

    return history.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)


def load_mongodb_factory_observations():
    """
    Read additional factory observations from MongoDB.

    Never silently infer missing nine-input values
    from an older input format.
    """

    collection = get_database()[
        OBSERVATION_COLLECTION
    ]

    projection = {
        "_id": 0,
        "week_start_date": 1,
        "tea_type": 1,
        "grade": 1,
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

    if "week_start_date" not in records.columns:
        raise ValueError(
            "MongoDB observations are missing "
            "week_start_date."
        )

    if "tea_type" in records.columns:
        if "grade" not in records.columns:
            records["grade"] = records["tea_type"]
        else:
            records["grade"] = records[
                "tea_type"
            ].fillna(records["grade"])

    if "grade" not in records.columns:
        raise ValueError(
            "MongoDB observations must contain "
            "tea_type or grade."
        )

    missing = sorted(
        set(INPUT_FIELDS) - set(records.columns)
    )

    if missing:
        raise ValueError(
            "Saved observations use an outdated "
            "input format. Missing fields: "
            f"{missing}"
        )

    if records[INPUT_FIELDS].isna().any().any():
        raise ValueError(
            "Saved observations contain missing "
            "nine-input values."
        )

    records["week_start_date"] = (
        pd.to_datetime(
            records["week_start_date"],
            errors="raise",
        ).dt.normalize()
    )

    records["grade"] = records[
        "grade"
    ].astype(str)

    if not records["grade"].isin(
        TEA_GRADES
    ).all():
        raise ValueError(
            "Unsupported tea grade in MongoDB."
        )

    if (
        records["week_start_date"].dt.weekday != 0
    ).any():
        raise ValueError(
            "MongoDB reporting weeks must "
            "begin on Monday."
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

    values = records[
        INPUT_FIELDS
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Saved observations contain "
            "invalid numerical values."
        )

    positive_fields = [
        "demand_kg",
        *EXCHANGE_FIELDS,
        *AUCTION_FIELDS,
    ]

    for field in positive_fields:
        if (records[field] <= 0).any():
            raise ValueError(
                f"MongoDB {field} must be positive."
            )

    for field in [
        "production_kg",
        "inventory_kg",
    ]:
        if (records[field] < 0).any():
            raise ValueError(
                f"MongoDB {field} cannot be negative."
            )

    if not records[
        "festival_flag"
    ].isin([0, 1]).all():
        raise ValueError(
            "MongoDB festival_flag must "
            "contain 0 or 1."
        )

    return records[columns].sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)


def validate_completed_weeks(dataframe):
    """Reject future or incomplete reporting weeks."""

    today = factory_today()

    for timestamp in dataframe[
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
                "MongoDB contains an incomplete "
                f"reporting week: {week_start}"
            )


def build_calendar_features(dataframe, first_week):
    """Reproduce the training calendar features."""

    result = dataframe.copy()

    dates = pd.to_datetime(
        result["week_start_date"],
        errors="raise",
    )

    origin = pd.Timestamp(first_week)

    offsets = (
        dates - origin
    ).dt.days

    if (offsets % 7 != 0).any():
        raise ValueError(
            "Reporting dates are not "
            "seven-day aligned."
        )

    result["time_idx"] = (
        offsets // 7
    ).astype("int64")

    iso_weeks = (
        dates.dt.isocalendar()
        .week.astype("int64")
    )

    months = dates.dt.month

    result["week_sin"] = np.sin(
        2.0 * np.pi * iso_weeks / 52.1775
    )

    result["week_cos"] = np.cos(
        2.0 * np.pi * iso_weeks / 52.1775
    )

    result["month_sin"] = np.sin(
        2.0 * np.pi * months / 12.0
    )

    result["month_cos"] = np.cos(
        2.0 * np.pi * months / 12.0
    )

    return result


def verify_shared_weekly_features(dataframe):
    """Check market values shared across grades."""

    for field in SHARED_FIELDS:
        grouped = dataframe.groupby(
            "week_start_date"
        )[field]

        minimum = grouped.min()
        maximum = grouped.max()

        tolerance = (
            0.0
            if field == "festival_flag"
            else 0.001
        )

        differences = (
            maximum - minimum
        )

        invalid = differences[
            differences > tolerance
        ]

        if not invalid.empty:
            invalid_week = pd.Timestamp(
                invalid.index[0]
            ).date()

            raise ValueError(
                f"Conflicting {field} values "
                f"for week {invalid_week}."
            )


def load_eight_input_history():
    """
    Combine protected CSV history and MongoDB
    observations without modifying either source.
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
        verify_shared_weekly_features(original)

        return original[
            MODEL_COLUMNS
        ].copy()

    if (
        saved["week_start_date"] <= csv_latest
    ).any():
        raise ValueError(
            "MongoDB observations overlap "
            "the protected historical dataset."
        )

    validate_completed_weeks(saved)

    saved = build_calendar_features(
        saved,
        first_week,
    )

    missing = sorted(
        set(MODEL_COLUMNS) - set(saved.columns)
    )

    if missing:
        raise ValueError(
            "Saved observations lack TFT "
            f"features: {missing}"
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
            "Duplicate combined historical "
            "factory observations."
        )

    numeric = combined[
        NUMERIC_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            "Combined history contains "
            "invalid numerical values."
        )

    verify_shared_weekly_features(combined)

    return combined


def verify_grade_encoder(history, grade):
    """Verify a complete 52-week encoder window."""

    history = history.sort_values(
        "week_start_date"
    ).reset_index(drop=True)

    if len(history) != ENCODER_LENGTH:
        raise ValueError(
            f"{grade}: expected exactly "
            f"{ENCODER_LENGTH} historical weeks."
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
        or not actual_dates.equals(
            expected_dates
        )
    ):
        raise ValueError(
            f"{grade}: incomplete or "
            "irregular 52-week history."
        )

    indices = history[
        "time_idx"
    ].to_numpy(dtype=np.int64)

    if not np.all(
        np.diff(indices) == 1
    ):
        raise ValueError(
            f"{grade}: nonconsecutive "
            "TFT time indices."
        )

    numeric = history[
        NUMERIC_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"{grade}: invalid historical "
            "model inputs."
        )


def verify_record_against_history(
    record,
    historical_row,
):
    """
    Keep strict matching for the existing
    saved-data forecasting workflow.
    """

    for field in INPUT_FIELDS:
        if not hasattr(record, field):
            raise ValueError(
                "Weekly request is missing "
                f"required model field: {field}"
            )

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


def validate_preview_batch(
    batch_input: EightInputWeeklyBatch,
):
    """
    Validate a latest-week forecast preview.

    The preview is not a permission to update
    the database. Storage permissions are
    enforced separately.
    """

    if not isinstance(
        batch_input,
        EightInputWeeklyBatch,
    ):
        raise ValueError(
            "A valid weekly forecast batch is required."
        )

    requested = batch_input.week_start_date

    if requested != latest_completed_week_start():
        raise ValueError(
            "Forecast previews may use only "
            "the latest completed reporting week."
        )

    submitted = [
        record.tea_type
        for record in batch_input.records
    ]

    if (
        len(submitted) != len(TEA_GRADES)
        or set(submitted) != set(TEA_GRADES)
        or len(set(submitted)) != len(submitted)
    ):
        raise ValueError(
            "Forecast preview requires all "
            "five tea grades exactly once."
        )

    reference = batch_input.records[0]

    for record in batch_input.records[1:]:
        for field in SHARED_FIELDS:
            left = float(
                getattr(reference, field)
            )

            right = float(
                getattr(record, field)
            )

            tolerance = (
                0.0
                if field == "festival_flag"
                else 0.001
            )

            if not np.isclose(
                left,
                right,
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"{field} must match "
                    "across all tea grades."
                )


def apply_preview_observations(
    historical,
    batch_input,
):
    """
    Overlay submitted latest-week values in memory.

    Existing MongoDB observations are not changed.
    """

    validate_preview_batch(batch_input)

    original = load_original_factory_history()

    protected_until = pd.Timestamp(
        original["week_start_date"].max()
    )

    requested = pd.Timestamp(
        batch_input.week_start_date
    ).normalize()

    if requested <= protected_until:
        raise ValueError(
            "Historical factory records "
            "cannot be previewed or replaced."
        )

    # Do not allow any existing observations
    # beyond the submitted reporting week.
    if (
        historical["week_start_date"] > requested
    ).any():
        raise ValueError(
            "Preview reporting week is not "
            "the latest available observation week."
        )

    first_week = pd.Timestamp(
        historical["week_start_date"].min()
    )

    # Preserve all previous weeks unchanged.
    earlier = historical.loc[
        historical["week_start_date"] < requested
    ].copy()

    previous_week = (
        requested - timedelta(weeks=1)
    )

    previous_records = earlier.loc[
        earlier["week_start_date"] == previous_week
    ]

    if set(previous_records["grade"]) != set(
        TEA_GRADES
    ):
        raise ValueError(
            "The preceding reporting week must "
            "contain all five tea-grade observations."
        )

    rows = []

    for record in batch_input.records:
        row = {
            "week_start_date": requested,
            "grade": record.tea_type,
        }

        for field in INPUT_FIELDS:
            row[field] = float(
                getattr(record, field)
            )

        rows.append(row)

    preview = pd.DataFrame(rows)

    preview = build_calendar_features(
        preview,
        first_week,
    )

    combined = pd.concat(
        [
            earlier[MODEL_COLUMNS].copy(),
            preview[MODEL_COLUMNS].copy(),
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)

    if combined.duplicated(
        ["grade", "week_start_date"]
    ).any():
        raise ValueError(
            "Preview contains duplicate "
            "grade/week observations."
        )

    numeric = combined[
        NUMERIC_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            "Preview contains invalid "
            "numerical values."
        )

    verify_shared_weekly_features(combined)

    return combined


def _prepare_history(
    batch_input: EightInputWeeklyBatch,
    *,
    preview_latest_week=False,
):
    """
    Shared encoder preparation.

    Normal mode strictly verifies submitted data
    against stored observations.

    Preview mode overlays latest-week values
    only in memory.
    """

    historical = load_eight_input_history()

    requested_date = pd.Timestamp(
        batch_input.week_start_date
    ).normalize()

    if requested_date.dayofweek != 0:
        raise ValueError(
            "Forecast origin must be Monday."
        )

    if (
        requested_date + timedelta(days=6)
    ).date() >= factory_today():
        raise ValueError(
            "The selected reporting week "
            "has not been completed."
        )

    if preview_latest_week:
        historical = apply_preview_observations(
            historical,
            batch_input,
        )

    earliest = pd.Timestamp(
        historical["week_start_date"].min()
    )

    if requested_date < earliest:
        raise ValueError(
            "Requested reporting date precedes "
            "historical observations."
        )

    selected_grades = list(
        batch_input.selected_grades
    )

    if not selected_grades:
        raise ValueError(
            "At least one tea grade is required."
        )

    if len(selected_grades) != len(
        set(selected_grades)
    ):
        raise ValueError(
            "Duplicate selected tea grades."
        )

    if not set(selected_grades).issubset(
        set(TEA_GRADES)
    ):
        raise ValueError(
            "Unsupported selected tea grade."
        )

    records_by_grade = {}

    for record in batch_input.records:
        grade = record.tea_type

        if grade in records_by_grade:
            raise ValueError(
                f"Duplicate request record: {grade}"
            )

        records_by_grade[grade] = record

    if set(records_by_grade) != set(
        selected_grades
    ):
        raise ValueError(
            "Submitted records must match "
            "the selected tea grades exactly."
        )

    prepared_groups = []
    latest_per_grade = {}

    for grade in selected_grades:
        record = records_by_grade[grade]

        grade_history = historical.loc[
            historical["grade"] == grade
        ].sort_values(
            "week_start_date"
        )

        if grade_history.empty:
            raise ValueError(
                f"No historical data for {grade}."
            )

        latest_per_grade[grade] = (
            pd.Timestamp(
                grade_history[
                    "week_start_date"
                ].max()
            )
        )

        matching = grade_history.loc[
            grade_history[
                "week_start_date"
            ] == requested_date
        ]

        if len(matching) != 1:
            raise ValueError(
                f"{grade}: no observation for "
                f"{requested_date.date()}."
            )

        if not preview_latest_week:
            verify_record_against_history(
                record,
                matching.iloc[0],
            )

        encoder = (
            grade_history.loc[
                grade_history[
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
            "Incorrect historical encoder "
            "record count."
        )

    forecast_start = (
        requested_date + timedelta(weeks=1)
    )

    forecast_end = (
        requested_date
        + timedelta(weeks=PREDICTION_LENGTH)
    )

    latest_for_all = all(
        latest_per_grade[grade] == requested_date
        for grade in selected_grades
    )

    forecast_mode = (
        "latest_completed_week_forecast"
        if latest_for_all
        else "historical_replay"
    )

    latest_dataset_week = pd.Timestamp(
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
        "forecast_horizon_weeks": PREDICTION_LENGTH,
        "forecast_mode": forecast_mode,
        "dataset_latest_week": (
            latest_dataset_week.strftime("%Y-%m-%d")
        ),
        "latest_week_per_selected_grade": {
            grade: latest_per_grade[
                grade
            ].strftime("%Y-%m-%d")
            for grade in selected_grades
        },
        "preview_latest_week": preview_latest_week,
    }


def prepare_eight_input_history(
    batch_input: EightInputWeeklyBatch,
):
    """
    Existing strict historical preparation.

    This function keeps its original behavior:
    submitted values must match recorded values.
    """

    return _prepare_history(
        batch_input,
        preview_latest_week=False,
    )


def prepare_eight_input_history_preview(
    batch_input: EightInputWeeklyBatch,
):
    """
    Prepare a latest-week forecast using edited
    observations in memory.

    No MongoDB observations are modified.

    This function is intended for the controlled
    Generate Forecasts backend workflow.
    """

    return _prepare_history(
        batch_input,
        preview_latest_week=True,
    )
