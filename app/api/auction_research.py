"""Read-only auction study results, ablation and dated forecast explanations."""

import csv
import json
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/research/auction", tags=["auction research"])
OUTPUTS = Path(__file__).resolve().parents[2] / "research" / "outputs"
TARGETS = {"high_1000kg", "medium_1000kg", "low_1000kg"}
HORIZONS = {1, 2, 3, 4}
EXPECTED = {(target, horizon) for target in TARGETS for horizon in HORIZONS}


def read_json(filename):
    path = OUTPUTS / filename
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Missing research output: {filename}",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unreadable research output: {filename}",
        ) from exc


def forecast_baselines(origin):
    """Use only the eight observed calendar weeks ending at the forecast origin."""
    path = OUTPUTS / "auction_calendar_weeks.csv"
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            weeks = list(csv.DictReader(handle))

        origins = [
            index
            for index, row in enumerate(weeks)
            if row["week_start"] == origin.isoformat()
        ]
        if len(origins) != 1 or origins[0] < 7:
            raise ValueError("Forecast origin absent from calendar")

        history = weeks[origins[0] - 7:origins[0] + 1]
        if any(
            date.fromisoformat(row["week_start"])
            != origin - timedelta(weeks=7 - index)
            or row["sale_count"] != "1"
            for index, row in enumerate(history)
        ):
            raise ValueError("Incomplete eight-week forecast history")

        return {
            target: {
                "last_1000kg": float(history[-1][target]),
                "mean4_1000kg": round(
                    sum(float(row[target]) for row in history[-4:]) / 4,
                    2,
                ),
            }
            for target in TARGETS
        }
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Auction baseline history is unavailable or stale",
        ) from exc


@router.get("/results")
def auction_research_results():
    evaluation = read_json("tft_comparison.json")
    intervals = read_json("tft_interval_summary.json")

    if evaluation.get("target_kind") != "observed_auction_sales_by_elevation":
        raise HTTPException(
            status_code=503,
            detail="Unexpected auction evaluation target",
        )

    origin_count = evaluation.get("origin_count_per_target")
    if type(origin_count) is not int or origin_count < 1:
        raise HTTPException(
            status_code=503,
            detail="Invalid test-origin count",
        )

    interval_rows = intervals.get("results", [])
    if (
        len(interval_rows) != len(EXPECTED)
        or {
            (row.get("target"), row.get("horizon_week"))
            for row in interval_rows
        } != EXPECTED
        or any(row.get("n") != origin_count for row in interval_rows)
    ):
        raise HTTPException(
            status_code=503,
            detail="Rerun research/check_auction_intervals.py",
        )

    interpretation = {
        "status": "unavailable",
        "scope": "Run research/explain_auction_tft.py.",
    }
    if (OUTPUTS / "tft_explanation.json").is_file():
        item = read_json("tft_explanation.json")
        findings = item.get("results", {})
        if (
            isinstance(findings, dict)
            and set(findings) == TARGETS
            and all(
                findings[target].get("test_origins") == origin_count
                for target in TARGETS
            )
        ):
            item.pop("checkpoint", None)
            interpretation = item
        else:
            interpretation = {
                "status": "outdated",
                "scope": "Rerun research/explain_auction_tft.py.",
            }

    ablation = None
    if (OUTPUTS / "price_ablation.json").is_file():
        item = read_json("price_ablation.json")
        metrics = item.get("metrics", [])
        methods = {"last", "mean4", "volume_only_tft", "price_aware_tft"}
        keys = {
            (row.get("target"), row.get("method"), row.get("horizon_week"))
            for row in metrics
        }
        expected_metrics = {
            (target, method, horizon)
            for target in TARGETS
            for method in methods
            for horizon in HORIZONS
        }
        if (
            item.get("target_kind") == "observed_auction_sales_by_elevation"
            and item.get("test_origins_per_elevation") == origin_count
            and len(metrics) == len(expected_metrics)
            and keys == expected_metrics
            and all(
                row.get("n_origins") == origin_count
                for row in metrics
            )
        ):
            item.pop("checkpoint", None)
            ablation = item

    limitations = [
        f"{origin_count} overlapping 2026 test origins per elevation.",
        "Medium-grown TFT errors exceed the simple baselines in this experiment.",
        "Nominal 80% intervals are not reliably calibrated for medium-grown sales.",
        "Aggregate model importance is not a causal or individual forecast explanation.",
    ]
    if interpretation.get("status") in {"outdated", "unavailable"}:
        limitations.append(
            "Model explanation must be regenerated for this evaluation."
        )
    if ablation is not None:
        limitations.append(
            "Price ablation is exploratory because 2026 test results were already inspected."
        )

    return {
        "status": "exploratory",
        "target": "Observed weekly tea auction sales by elevation, in thousand kg",
        "scope_note": "Separate from the generated-data five-grade demand prototype.",
        "limitations": limitations,
        "evaluation": evaluation,
        "intervals": intervals,
        "interpretation": interpretation,
        "ablation": ablation,
    }


@router.get("/forecast")
def auction_research_forecast():
    forecast = read_json("latest_tft_forecast.json")
    try:
        origin = date.fromisoformat(forecast["origin_week_start"])
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Auction forecast origin is invalid",
        ) from exc

    rows = forecast.get("forecasts", [])
    if (
        forecast.get("target_kind")
        != "observed_auction_sales_by_elevation"
        or len(rows) != len(EXPECTED)
        or {
            (row.get("target"), row.get("horizon_week"))
            for row in rows
        } != EXPECTED
    ):
        raise HTTPException(
            status_code=503,
            detail="Unexpected auction forecast scope",
        )

    baselines = forecast_baselines(origin)
    for row in rows:
        expected_week = (
            origin + timedelta(weeks=row["horizon_week"])
        ).isoformat()
        if row.get("week_start") != expected_week:
            raise HTTPException(
                status_code=503,
                detail="Auction forecast dates do not match origin",
            )

    explanation = None
    if (OUTPUTS / "local_forecast_sensitivity.json").is_file():
        item = read_json("local_forecast_sensitivity.json")
        details = item.get("results", {})
        if (
            item.get("source_week_start") == origin.isoformat()
            and isinstance(details, dict)
            and set(details) == TARGETS
            and all(
                len(details[target].get("input_sensitivity", [])) == 8
                for target in TARGETS
            )
        ):
            explanation = item

    current_week = origin <= date.today() < origin + timedelta(weeks=1)
    return {
        **forecast,
        "baselines_by_elevation": baselines,
        "is_current_source_week": current_week,
        "source_status": (
            "Current source week"
            if current_week
            else "Historical source week; refresh published auction data before new decisions"
        ),
        "local_explanation": explanation,
    }


@router.get("/prospective")
def auction_research_prospective():
    """Expose the committed forecast snapshot; never substitute a new forecast."""
    snapshot = read_json(
        "prospective/auction_forecast_20260917T222937Z.json"
    )
    if (
        snapshot.get("target_kind")
        != "observed_auction_sales_by_elevation"
        or snapshot.get("issued_at_utc")
        != "2026-09-17T22:29:37.909548+00:00"
        or len(snapshot.get("forecasts", [])) != 6
        or {
            (row.get("target"), row.get("target_week_start"))
            for row in snapshot["forecasts"]
        }
        != {
            (target, week)
            for target in TARGETS
            for week in ("2026-09-21", "2026-09-28")
        }
    ):
        raise HTTPException(
            status_code=503,
            detail="Frozen forecast snapshot is invalid",
        )

    calendar_path = OUTPUTS / "auction_calendar_weeks.csv"
    try:
        with calendar_path.open(newline="", encoding="utf-8-sig") as handle:
            calendar = {
                row["week_start"]: row
                for row in csv.DictReader(handle)
            }
    except (OSError, KeyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Auction calendar unavailable",
        ) from exc

    results = []
    for prediction in snapshot["forecasts"]:
        row = calendar.get(prediction["target_week_start"])
        result = {
            **prediction,
            "outcome_status": "awaiting published sale",
        }
        if (
            row
            and row.get("sale_count") == "1"
            and row.get(prediction["target"])
        ):
            try:
                sale_date = date.fromisoformat(row["source_sale_dates"])
                if sale_date <= date.fromisoformat("2026-09-17"):
                    raise ValueError("Sale date precedes snapshot")
                result["outcome_status"] = (
                    "observed in local auction dataset"
                )
                result["actual_1000kg"] = float(
                    row[prediction["target"]]
                )
                result["sale_date"] = sale_date.isoformat()
            except (ValueError, TypeError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Invalid prospective outcome date",
                ) from exc
        results.append(result)

    return {
        "issued_at_utc": snapshot["issued_at_utc"],
        "last_source_sale_date": snapshot["last_source_sale_date"],
        "unit": snapshot["unit"],
        "target_kind": snapshot["target_kind"],
        "scope_note": (
            "Frozen before the target weeks; only two future weeks "
            "were eligible on issuance."
        ),
        "outcome_note": (
            "Local observations require source verification and scoring "
            "before an accuracy claim."
        ),
        "forecasts": results,
    }