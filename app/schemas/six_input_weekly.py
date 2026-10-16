
"""
Input validation for the six-input TFT model.

This schema is separate from the existing
three-input research forecasting schema.
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


class SixInputTeaRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
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

    usd_lkr_rate: float = Field(
        gt=0,
        le=10000,
    )

    festival_flag: Literal[0, 1]

    colombo_auction_price_lkr: float = Field(
        gt=0,
        le=100000,
    )


class SixInputWeeklyBatch(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    week_start_date: date

    records: list[SixInputTeaRecord] = Field(
        min_length=1,
        max_length=5,
    )

    @field_validator("week_start_date")
    @classmethod
    def validate_week_date(cls, value):
        if value.weekday() != 0:
            raise ValueError(
                "Week start date must be a Monday."
            )

        return value

    @model_validator(mode="after")
    def validate_records(self):
        grades = [
            record.tea_type
            for record in self.records
        ]

        if len(grades) != len(set(grades)):
            raise ValueError(
                "Duplicate tea grades are not allowed."
            )

        exchange_rates = {
            record.usd_lkr_rate
            for record in self.records
        }

        if len(exchange_rates) != 1:
            raise ValueError(
                "USD/LKR exchange rate must be "
                "consistent across selected grades "
                "for the same week."
            )

        festival_values = {
            record.festival_flag
            for record in self.records
        }

        if len(festival_values) != 1:
            raise ValueError(
                "Festival indicator must be "
                "consistent across selected grades "
                "for the same week."
            )

        return self

    @property
    def selected_grades(self):
        return [
            record.tea_type
            for record in self.records
        ]

    @property
    def forecast_start_date(self):
        return (
            self.week_start_date
            + timedelta(weeks=1)
        )

    @property
    def forecast_end_date(self):
        return (
            self.week_start_date
            + timedelta(weeks=4)
        )
