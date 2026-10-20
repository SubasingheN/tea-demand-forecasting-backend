
"""
Input validation for the production nine-input
Tea Demand Forecasting TFT.

Supported tea grades:
    BOP
    BOPF
    Dust
    OP
    Pekoe

Nine historical model inputs:
    demand_kg
    production_kg
    inventory_kg
    usd_lkr_monday_rate
    usd_lkr_sunday_rate
    festival_flag
    auction_price_high_lkr
    auction_price_medium_lkr
    auction_price_low_lkr

The two USD/LKR rates represent the Monday
and Sunday observations for a completed
Monday-to-Sunday reporting week.

Shared weekly inputs must be consistent
across all selected tea grades.

This schema:
    - Validates submitted forecasting data
    - Does not train the model
    - Does not write to MongoDB
    - Does not change factory entry permissions

The latest-completed-week restriction is
enforced by the factory observation API.
"""

from datetime import date, timedelta
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


TeaGrade = Literal[
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]


HISTORICAL_INPUT_FEATURES = [
    "demand_kg",
    "production_kg",
    "inventory_kg",
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
    "festival_flag",
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]


SHARED_WEEKLY_FIELDS = [
    "usd_lkr_monday_rate",
    "usd_lkr_sunday_rate",
    "festival_flag",
    "auction_price_high_lkr",
    "auction_price_medium_lkr",
    "auction_price_low_lkr",
]


class EightInputTeaRecord(BaseModel):
    """
    One tea grade's observed data for
    a completed reporting week.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    tea_type: TeaGrade

    demand_kg: float = Field(
        gt=0,
        le=100000,
    )

    production_kg: float = Field(
        ge=0,
        le=100000,
    )

    inventory_kg: float = Field(
        ge=0,
        le=100000,
    )

    usd_lkr_monday_rate: float = Field(
        gt=0,
        le=10000,
    )

    usd_lkr_sunday_rate: float = Field(
        gt=0,
        le=10000,
    )

    festival_flag: Literal[0, 1]

    auction_price_high_lkr: float = Field(
        gt=0,
        le=100000,
    )

    auction_price_medium_lkr: float = Field(
        gt=0,
        le=100000,
    )

    auction_price_low_lkr: float = Field(
        gt=0,
        le=100000,
    )

    @property
    def usd_lkr_exchange_rate_range(self) -> str:
        """
        Return the exchange-rate range
        in the same format as the dataset.

        This is a display/reference value,
        not a separate model input.
        """

        return (
            f"{self.usd_lkr_monday_rate:.4f}"
            "-"
            f"{self.usd_lkr_sunday_rate:.4f}"
        )


class EightInputWeeklyBatch(BaseModel):
    """
    One completed reporting week containing
    between one and five tea-grade records.

    The factory save endpoint separately
    requires all five grades.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    week_start_date: date

    records: list[EightInputTeaRecord] = Field(
        min_length=1,
        max_length=5,
    )

    @field_validator("week_start_date")
    @classmethod
    def validate_week_date(cls, value: date):
        if value.weekday() != 0:
            raise ValueError(
                "Week start date must be a Monday."
            )

        return value

    @model_validator(mode="after")
    def validate_records(self):
        """
        Reject duplicate tea grades and
        inconsistent shared weekly inputs.
        """

        grades = [
            record.tea_type
            for record in self.records
        ]

        if len(grades) != len(set(grades)):
            raise ValueError(
                "Duplicate tea grades are not allowed."
            )

        for field_name in SHARED_WEEKLY_FIELDS:
            values = [
                getattr(record, field_name)
                for record in self.records
            ]

            if field_name == "festival_flag":
                consistent = (
                    len(set(values)) == 1
                )
            else:
                # Small tolerance for decimal
                # representation and serialization.
                consistent = all(
                    abs(value - values[0]) <= 0.0001
                    for value in values[1:]
                )

            if not consistent:
                raise ValueError(
                    f"{field_name} must be consistent "
                    "across selected tea grades "
                    "for the same reporting week."
                )

        return self

    @property
    def selected_grades(self) -> list[str]:
        return [
            record.tea_type
            for record in self.records
        ]

    @property
    def forecast_start_date(self) -> date:
        return (
            self.week_start_date
            + timedelta(weeks=1)
        )

    @property
    def forecast_end_date(self) -> date:
        return (
            self.week_start_date
            + timedelta(weeks=4)
        )

    @property
    def historical_input_features(self) -> list[str]:
        return list(
            HISTORICAL_INPUT_FEATURES
        )

    @property
    def usd_lkr_exchange_rate_range(self) -> str:
        """
        Return the common weekly USD/LKR range.
        """

        return (
            self.records[0]
            .usd_lkr_exchange_rate_range
        )
