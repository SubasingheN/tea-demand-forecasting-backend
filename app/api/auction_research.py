"""Read-only results for the separately evaluated auction-sales study."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/research/auction", tags=["auction research"])
OUTPUTS = Path(__file__).resolve().parents[2] / "research" / "outputs"
FILES = {
    "evaluation": "tft_comparison.json",
    "interpretation": "tft_explanation.json",
    "intervals": "tft_interval_summary.json",
}


@router.get("/results")
def auction_research_results():
    missing = [filename for filename in FILES.values() if not (OUTPUTS / filename).is_file()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"message": "Run the auction research evaluation scripts first.",
                    "missing_files": missing},
        )
    try:
        results = {name: json.loads((OUTPUTS / filename).read_text(encoding="utf-8"))
                   for name, filename in FILES.items()}
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Auction research output is unreadable") from exc
    if (results["evaluation"].get("target_kind") != "observed_auction_sales_by_elevation"
            or results["evaluation"].get("origin_count_per_target") != 9):
        raise HTTPException(status_code=503, detail="Unexpected auction evaluation scope")
    results["interpretation"].pop("checkpoint", None)  # Local filesystem path is not an API field.
    return {
        "status": "exploratory",
        "target": "Observed weekly tea auction sales by elevation, in thousand kg",
        "scope_note": "Separate from the generated-data five-grade demand prototype.",
        "limitations": [
            "Nine overlapping 2026 test origins per elevation.",
            "Medium-grown TFT errors exceed the baselines in this experiment.",
            "Nominal 80% intervals are not reliably calibrated for medium-grown sales.",
            "Aggregate model importance is not a causal or individual forecast explanation.",
        ],
        **results,
    }
