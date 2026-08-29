from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
)
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.services.weekly_forecast_service import (
    format_future_forecasts,
)


REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def _format_number(
    value,
) -> str:
    if (
        value is None
    ):
        return "-"

    try:
        return (
            f"{float(value):,.2f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def _get_demand_range(
    forecast: dict,
) -> str:
    demand_range = (
        forecast.get(
            "demand_range"
        )
    )

    if demand_range:
        return str(
            demand_range
        )

    lower = forecast.get(
        "lower_demand_kg"
    )

    upper = forecast.get(
        "upper_demand_kg"
    )

    if (
        lower is None or
        upper is None
    ):
        return "-"

    return (
        f"{_format_number(lower)} - "
        f"{_format_number(upper)} kg"
    )


def generate_forecast_pdf(
    forecasts: list,
    input_week_start_date=None,
    source: str = "research",
) -> Path:
    """
    Generate a PDF report from already
    available TFT forecast results.

    This function does not retrain or
    regenerate the TFT model.
    """

    if not forecasts:
        raise ValueError(
            "Forecast data is required "
            "to generate the report."
        )


    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )


    if source == "operational":
        filename_prefix = (
            "operational_tea_demand_forecast"
        )
    else:
        filename_prefix = (
            "weekly_tea_demand_forecast"
        )


    report_path = (
        REPORT_DIR /
        (
            f"{filename_prefix}_"
            f"{timestamp}.pdf"
        )
    )


    document = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )


    styles = (
        getSampleStyleSheet()
    )


    story = []


    # ==================================================
    # TITLE
    # ==================================================

    title = Paragraph(
        (
            "Tea Demand Forecast Report"
        ),
        styles["Title"],
    )

    story.append(title)

    story.append(
        Spacer(
            1,
            0.4 * cm,
        )
    )


    # ==================================================
    # REPORT INFORMATION
    # ==================================================

    generated_text = Paragraph(
        (
            "Generated at: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ),
        styles["Normal"],
    )

    story.append(
        generated_text
    )


    story.append(
        Spacer(
            1,
            0.2 * cm,
        )
    )


    source_text = (
        "Operational Forecast"
        if source == "operational"
        else "Research Forecast"
    )


    report_source = Paragraph(
        (
            "Forecast Source: "
            f"{source_text}"
        ),
        styles["Normal"],
    )

    story.append(
        report_source
    )


    if input_week_start_date:
        story.append(
            Spacer(
                1,
                0.2 * cm,
            )
        )

        input_week_text = Paragraph(
            (
                "Input Week: "
                f"{input_week_start_date}"
            ),
            styles["Normal"],
        )

        story.append(
            input_week_text
        )


    story.append(
        Spacer(
            1,
            0.5 * cm,
        )
    )


    # ==================================================
    # DESCRIPTION
    # ==================================================

    description = Paragraph(
        (
            "This report contains the "
            "4-week Temporal Fusion "
            "Transformer (TFT) tea demand "
            "forecast for BOP, BOPF, Dust, "
            "OP and Pekoe. Each weekly "
            "forecast includes lower, "
            "median and upper demand "
            "estimates together with the "
            "predicted demand range and "
            "overall demand trend."
        ),
        styles["Normal"],
    )

    story.append(
        description
    )


    story.append(
        Spacer(
            1,
            0.6 * cm,
        )
    )


    # ==================================================
    # FORECAST TABLES
    # ==================================================

    for tea_result in forecasts:
        tea_type = (
            tea_result.get(
                "tea_type",
                "Unknown",
            )
        )

        trend = (
            tea_result.get(
                "trend",
                "Not Available",
            )
        )


        section_title = Paragraph(
            (
                f"{tea_type} - "
                f"Trend: {trend}"
            ),
            styles["Heading2"],
        )


        story.append(
            section_title
        )

        story.append(
            Spacer(
                1,
                0.2 * cm,
            )
        )


        table_data = [
            [
                "Week",
                "Date",
                "Lower (kg)",
                "Median (kg)",
                "Upper (kg)",
                "Demand Range",
            ]
        ]


        tea_forecasts = (
            tea_result.get(
                "forecasts",
                [],
            )
        )


        for forecast in tea_forecasts:
            table_data.append(
                [
                    forecast.get(
                        "week",
                        "-"
                    ),

                    forecast.get(
                        "week_start_date",
                        "-"
                    ),

                    _format_number(
                        forecast.get(
                            "lower_demand_kg"
                        )
                    ),

                    _format_number(
                        forecast.get(
                            "median_demand_kg"
                        )
                    ),

                    _format_number(
                        forecast.get(
                            "upper_demand_kg"
                        )
                    ),

                    _get_demand_range(
                        forecast
                    ),
                ]
            )


        table = Table(
            table_data,
            colWidths=[
                1.3 * cm,
                2.5 * cm,
                2.5 * cm,
                2.5 * cm,
                2.5 * cm,
                4.0 * cm,
            ],
            repeatRows=1,
        )


        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.black,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, 0),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, 0),
                        6,
                    ),
                ]
            )
        )


        story.append(
            table
        )

        story.append(
            Spacer(
                1,
                0.6 * cm,
            )
        )


    # ==================================================
    # FOOTER NOTE
    # ==================================================

    note = Paragraph(
        (
            "Note: Forecast ranges represent "
            "model prediction uncertainty. "
            "The median value is used as the "
            "main demand estimate."
        ),
        styles["Normal"],
    )

    story.append(
        Spacer(
            1,
            0.3 * cm,
        )
    )

    story.append(
        note
    )


    document.build(
        story
    )


    return report_path


def generate_weekly_forecast_pdf() -> Path:
    """
    Preserve the original research
    report behaviour.
    """

    forecasts = (
        format_future_forecasts()
    )

    return generate_forecast_pdf(
        forecasts=forecasts,
        source="research",
    )


def generate_operational_forecast_pdf(
    forecasts: list,
    input_week_start_date=None,
) -> Path:
    """
    Generate a PDF directly from an
    operational forecast that has already
    been produced from Weekly Data Entry.
    """

    return generate_forecast_pdf(
        forecasts=forecasts,
        input_week_start_date=
            input_week_start_date,
        source="operational",
    )