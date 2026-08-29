from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


TeaType = Literal[
    "BOP",
    "BOPF",
    "Dust",
    "OP",
    "Pekoe",
]


class WeeklyTeaInput(BaseModel):
    tea_type: TeaType

    demand_kg: float = Field(
        ge=0,
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

    colombo_auction_price_lkr: float = Field(
        gt=0,
        le=100000,
    )

    rainfall_index: float = Field(
        ge=0,
        le=10000,
    )

    usd_lkr_rate: float = Field(
        gt=0,
        le=10000,
    )

    festival_flag: int = Field(
        ge=0,
        le=1,
    )

    global_commodity_shock: float = Field(
        ge=-10,
        le=10,
    )


class WeeklyBatchInput(BaseModel):
    week_start_date: date

    records: list[WeeklyTeaInput]

    @field_validator("records")
    @classmethod
    def validate_record_count(
        cls,
        records,
    ):
        if len(records) != 5:
            raise ValueError(
                "Weekly batch must contain "
                "exactly 5 tea-type records."
            )

        return records

    @field_validator("week_start_date")
    @classmethod
    def validate_monday(
        cls,
        week_start_date,
    ):
        if week_start_date.weekday() != 0:
            raise ValueError(
                "week_start_date must be a Monday."
            )

        return week_start_date