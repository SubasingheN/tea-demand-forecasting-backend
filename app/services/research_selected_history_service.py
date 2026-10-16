
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from app.schemas.research_weekly_input import (
    ResearchWeeklyBatchInput,
)


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"

if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from train_tft_no_price import (
    prepare_no_price_data,
    GRADES,
    ENCODER_LENGTH,
)

from forecast_next_four_weeks import (
    calculate_calendar_features,
)


MODEL_COLUMNS = [
    "week_start_date",
    "grade",
    "time_idx",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "demand_kg",
    "production_kg",
    "inventory_kg",
]


def load_research_history():
    """
    Load the same processed research data
    used by the selected no-price TFT.

    This function does not write to the
    research dataset.
    """
    dataframe = prepare_no_price_data().copy()

    dataframe["week_start_date"] = (
        pd.to_datetime(
            dataframe["week_start_date"],
            errors="raise",
        ).dt.normalize()
    )

    return dataframe.sort_values(
        ["grade", "week_start_date"]
    ).reset_index(drop=True)


def validate_consecutive_history(
    history: pd.DataFrame,
    grade: str,
):
    """
    Verify that the encoder has exactly
    52 consecutive Monday observations.
    """
    if len(history) != ENCODER_LENGTH:
        raise ValueError(
            f"{grade}: expected "
            f"{ENCODER_LENGTH} historical weeks, "
            f"found {len(history)}."
        )

    dates = pd.DatetimeIndex(
        history["week_start_date"]
    )

    if dates.has_duplicates:
        raise ValueError(
            f"{grade}: duplicate historical weeks."
        )

    expected_dates = pd.date_range(
        end=dates[-1],
        periods=ENCODER_LENGTH,
        freq="W-MON",
    )

    if not dates.equals(expected_dates):
        raise ValueError(
            f"{grade}: historical weeks are "
            "missing or not consecutive Mondays."
        )

    indices = history["time_idx"].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        np.diff(indices),
        np.ones(
            ENCODER_LENGTH - 1,
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            f"{grade}: historical time indices "
            "are not consecutive."
        )


def build_new_observation(
    record,
    week_start_date,
    next_time_idx,
):
    """
    Construct one genuine observed week
    using the research model's feature
    definitions.
    """
    calendar = calculate_calendar_features(
        week_start_date
    )

    return {
        "week_start_date": pd.Timestamp(
            week_start_date
        ),
        "grade": record.tea_type,
        "time_idx": int(next_time_idx),
        "week_sin": calendar["week_sin"],
        "week_cos": calendar["week_cos"],
        "month_sin": calendar["month_sin"],
        "month_cos": calendar["month_cos"],
        "demand_kg": float(record.demand_kg),
        "production_kg": float(
            record.production_kg
        ),
        "inventory_kg": float(
            record.inventory_kg
        ),
    }


def prepare_selected_history(
    batch_input: ResearchWeeklyBatchInput,
):
    """
    Prepare selected-grade encoder history.

    If the submitted completed week is one
    week after the research dataset endpoint,
    append the actual observation in memory.

    The original research dataset and model
    checkpoint remain unchanged.
    """
    historical = load_research_history()

    selected_grades = (
        batch_input.selected_tea_grades
    )

    requested_date = pd.Timestamp(
        batch_input.week_start_date
    )

    prepared_groups = []

    for record in batch_input.records:
        grade = record.tea_type

        if grade not in GRADES:
            raise ValueError(
                f"Unsupported grade: {grade}"
            )

        grade_history = historical[
            historical["grade"] == grade
        ].sort_values("week_start_date").copy()

        if grade_history.empty:
            raise ValueError(
                f"{grade}: no historical data."
            )

        latest_date = pd.Timestamp(
            grade_history[
                "week_start_date"
            ].iloc[-1]
        )

        latest_idx = int(
            grade_history[
                "time_idx"
            ].iloc[-1]
        )

        if requested_date == latest_date:
            existing = grade_history.iloc[-1]

            # Do not silently overwrite an
            # established research observation.
            observed_fields = (
                "demand_kg",
                "production_kg",
                "inventory_kg",
            )

            for field in observed_fields:
                if not np.isclose(
                    float(getattr(record, field)),
                    float(existing[field]),
                    rtol=0,
                    atol=0.0001,
                ):
                    raise ValueError(
                        f"{grade}: submitted {field} "
                        "differs from the existing "
                        "research dataset observation. "
                        "Historical corrections require "
                        "a separate audited workflow."
                    )

            encoder = grade_history.tail(
                ENCODER_LENGTH
            ).copy()

        elif requested_date == (
            latest_date + timedelta(weeks=1)
        ):
            new_row = build_new_observation(
                record,
                requested_date,
                latest_idx + 1,
            )

            encoder = pd.concat(
                [
                    grade_history.tail(
                        ENCODER_LENGTH - 1
                    ),
                    pd.DataFrame([new_row]),
                ],
                ignore_index=True,
            )

        else:
            raise ValueError(
                f"{grade}: submitted week must be "
                f"{latest_date.date()} or "
                f"{(latest_date + timedelta(weeks=1)).date()}. "
                "Additional operational weeks "
                "require a persisted, validated "
                "history-extension workflow."
            )

        encoder = encoder.sort_values(
            "week_start_date"
        ).reset_index(drop=True)

        validate_consecutive_history(
            encoder,
            grade,
        )

        prepared_groups.append(
            encoder[MODEL_COLUMNS].copy()
        )

    prepared = pd.concat(
        prepared_groups,
        ignore_index=True,
    )

    prepared = prepared.sort_values(
        ["grade", "time_idx"]
    ).reset_index(drop=True)

    expected_rows = (
        len(selected_grades) * ENCODER_LENGTH
    )

    if len(prepared) != expected_rows:
        raise ValueError(
            "Unexpected selected history size."
        )

    return {
        "historical_dataframe": prepared,
        "selected_grades": selected_grades,
        "forecast_origin": (
            requested_date.strftime("%Y-%m-%d")
        ),
        "forecast_start_date": (
            requested_date
            + timedelta(weeks=1)
        ).strftime("%Y-%m-%d"),
        "forecast_end_date": (
            requested_date
            + timedelta(weeks=4)
        ).strftime("%Y-%m-%d"),
        "encoder_weeks": ENCODER_LENGTH,
        "historical_rows": len(prepared),
    }
