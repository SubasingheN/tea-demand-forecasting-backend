
from datetime import date, timedelta
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


TeaGrade = Literal[
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]


class ResearchWeeklyTeaInput(BaseModel):
    """
    Historical observations required by the
    selected no-price TFT research model.

    Quantities are expressed in kilograms.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    tea_type: TeaGrade

    demand_kg: float = Field(
        ...,
        gt=0,
        le=100000,
        description=(
            "Observed customer demand during "
            "the completed week, in kg."
        ),
    )

    production_kg: float = Field(
        ...,
        ge=0,
        le=100000,
        description=(
            "Tea production during the "
            "completed week, in kg."
        ),
    )

    inventory_kg: float = Field(
        ...,
        ge=0,
        le=100000,
        description=(
            "Closing tea inventory at the "
            "end of the completed week, in kg."
        ),
    )


class ResearchWeeklyBatchInput(BaseModel):
    """
    Request containing between one and five
    selected tea-grade observations.

    Each grade can appear only once.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    week_start_date: date

    records: list[ResearchWeeklyTeaInput] = Field(
        ...,
        min_length=1,
        max_length=5,
    )

    @field_validator("week_start_date")
    @classmethod
    def validate_monday(cls, value: date):
        if value.weekday() != 0:
            raise ValueError(
                "The completed week must start "
                "on a Monday."
            )

        return value

    @field_validator("records")
    @classmethod
    def validate_unique_tea_grades(cls, records):
        grades = [
            record.tea_type
            for record in records
        ]

        if len(grades) != len(set(grades)):
            raise ValueError(
                "Duplicate tea grades are not allowed."
            )

        return records

    @property
    def selected_tea_grades(self) -> list[str]:
        return [
            record.tea_type
            for record in self.records
        ]

    @property
    def record_count(self) -> int:
        return len(self.records)

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
