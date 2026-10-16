
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import matplotlib

# Suitable for generating figures without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# STEP 18 - TFT EXPLAINABILITY VISUALIZATIONS
#
# Source: Step 17 interpretation CSV files
# Model: Selected no-price TFT
# Period: 2024 validation
#
# Generate:
#   1. Static feature importance
#   2. Historical encoder importance
#   3. Known-future decoder importance
#   4. Historical attention (52 weeks)
#
# Files: PNG (300 DPI) and SVG
# No model retraining or test-set evaluation.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_explainability"
)

OUTPUT_DIR = (
    ROOT
    / "research"
    / "outputs"
    / "tft_explainability"
    / "figures"
)

SOURCE_FILES = {
    "static": (
        INPUT_DIR / "static_feature_importance.csv"
    ),
    "encoder": (
        INPUT_DIR / "encoder_feature_importance.csv"
    ),
    "decoder": (
        INPUT_DIR / "decoder_feature_importance.csv"
    ),
    "attention": (
        INPUT_DIR / "historical_attention.csv"
    ),
}

REPORT_FILE = OUTPUT_DIR / "visualization_report.json"

DPI = 300

# Consistent visual identity across all figures.
NAVY = "#17324D"
BLUE = "#367DA8"
TEAL = "#268780"
LIGHT_BLUE = "#DCEAF2"
GRID = "#E1E8ED"
TEXT = "#263746"

FEATURE_LABELS = {
    "grade": "Tea grade",
    "demand_kg_scale": "Demand scale",
    "demand_kg_center": "Demand center",
    "relative_time_idx": "Relative time index",
    "time_idx": "Time index",
    "week_sin": "Week sine",
    "week_cos": "Week cosine",
    "month_sin": "Month sine",
    "month_cos": "Month cosine",
    "production_kg": "Production quantity",
    "demand_kg": "Historical demand",
    "inventory_kg": "Inventory quantity",
}

FIGURES = {
    "static": {
        "stem": "figure_01_static_feature_importance",
        "title": "Static Feature Importance",
        "subtitle": (
            "Selected no-price TFT | "
            "2024 validation"
        ),
    },
    "encoder": {
        "stem": "figure_02_historical_feature_importance",
        "title": "Historical Encoder Feature Importance",
        "subtitle": (
            "52-week historical input | "
            "2024 validation"
        ),
    },
    "decoder": {
        "stem": "figure_03_future_feature_importance",
        "title": "Known-Future Decoder Feature Importance",
        "subtitle": (
            "Four-week forecast horizon | "
            "2024 validation"
        ),
    },
    "attention": {
        "stem": "figure_04_historical_attention_52_weeks",
        "title": "Historical Temporal Attention",
        "subtitle": (
            "Aggregated TFT attention | "
            "2024 validation"
        ),
    },
}


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GRID,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def load_importance(path, category):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "feature",
        "relative_importance_percent",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{category}: missing columns {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            f"{category}: empty importance file"
        )

    if df["feature"].isna().any():
        raise ValueError(
            f"{category}: missing feature names"
        )

    df["feature"] = df["feature"].astype(str)

    if df["feature"].duplicated().any():
        raise ValueError(
            f"{category}: duplicate feature names"
        )

    df["relative_importance_percent"] = (
        pd.to_numeric(
            df["relative_importance_percent"],
            errors="raise",
        )
    )

    values = df[
        "relative_importance_percent"
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{category}: non-finite importance"
        )

    if (values < 0).any():
        raise ValueError(
            f"{category}: negative importance"
        )

    if not np.isclose(
        values.sum(),
        100.0,
        atol=0.05,
        rtol=0.0,
    ):
        raise ValueError(
            f"{category}: importance totals "
            f"{values.sum():.5f}%, not 100%"
        )

    df = df.sort_values(
        "relative_importance_percent",
        ascending=False,
    ).reset_index(drop=True)

    return df


def load_attention(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "encoder_position",
        "weeks_before_latest_observation",
        "relative_attention_percent",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Attention: missing columns "
            f"{sorted(missing)}"
        )

    if len(df) != 52:
        raise ValueError(
            f"Expected 52 attention positions, "
            f"found {len(df)}"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = df.sort_values(
        "weeks_before_latest_observation"
    ).reset_index(drop=True)

    ages = df[
        "weeks_before_latest_observation"
    ].to_numpy()

    if not np.array_equal(
        ages,
        np.arange(52),
    ):
        raise ValueError(
            "Historical attention ages must "
            "contain each integer from 0 to 51"
        )

    weights = df[
        "relative_attention_percent"
    ].to_numpy(dtype=float)

    if not np.isfinite(weights).all():
        raise ValueError(
            "Non-finite attention values"
        )

    if (weights < 0).any():
        raise ValueError(
            "Negative attention values"
        )

    if not np.isclose(
        weights.sum(),
        100.0,
        atol=0.05,
        rtol=0.0,
    ):
        raise ValueError(
            "Historical attention does not "
            "sum to approximately 100%"
        )

    return df


def add_figure_note(fig, note):
    fig.text(
        0.07,
        0.035,
        note,
        fontsize=8,
        color="#5E6F7A",
        ha="left",
        va="bottom",
        wrap=True,
    )


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {}

    for extension in ("png", "svg"):
        path = OUTPUT_DIR / (
            f"{stem}.{extension}"
        )

        options = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }

        if extension == "png":
            options["dpi"] = DPI

        fig.savefig(
            path,
            **options,
        )

        paths[extension] = str(path)

    plt.close(fig)

    return paths


def plot_importance(df, category):
    config = FIGURES[category]

    labels = [
        FEATURE_LABELS.get(
            feature,
            feature.replace("_", " ").title(),
        )
        for feature in df["feature"]
    ]

    values = df[
        "relative_importance_percent"
    ].to_numpy(dtype=float)

    height = max(
        4.2,
        1.15 + 0.56 * len(df),
    )

    fig, ax = plt.subplots(
        figsize=(10.5, height)
    )

    fig.subplots_adjust(
        left=0.27,
        right=0.95,
        top=0.76,
        bottom=0.19,
    )

    positions = np.arange(len(values))

    # Preserve descending rank from top to bottom.
    bars = ax.barh(
        positions,
        values,
        height=0.62,
        color=BLUE,
        edgecolor="none",
    )

    if len(bars):
        bars[0].set_color(TEAL)

    ax.set_yticks(
        positions,
        labels=labels,
    )

    ax.invert_yaxis()

    upper_bound = max(
        10.0,
        np.ceil(values.max() / 10) * 10 + 10,
    )

    ax.set_xlim(
        0,
        upper_bound,
    )

    ax.set_xlabel(
        "Relative importance within feature group (%)",
        labelpad=9,
    )

    ax.xaxis.grid(
        True,
        color=GRID,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        length=0,
        pad=8,
    )

    for bar, value in zip(bars, values):
        ax.text(
            value + upper_bound * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=NAVY,
        )

    fig.text(
        0.07,
        0.94,
        config["title"],
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )

    fig.text(
        0.07,
        0.875,
        config["subtitle"],
        fontsize=10,
        color="#5E6F7A",
    )

    add_figure_note(
        fig,
        "Source: TFT variable-selection interpretation. "
        "Percentages are normalized within this feature "
        "group and are not causal effects.",
    )

    return save_figure(
        fig,
        config["stem"],
    )


def plot_attention(df):
    config = FIGURES["attention"]

    ages = df[
        "weeks_before_latest_observation"
    ].to_numpy(dtype=int)

    values = df[
        "relative_attention_percent"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(12, 5.8)
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.96,
        top=0.76,
        bottom=0.25,
    )

    ax.bar(
        ages,
        values,
        color=BLUE,
        width=0.82,
        edgecolor="none",
        label="Aggregated attention",
    )

    # Distinguish a uniform allocation as
    # a visual reference, not a model baseline.
    uniform_level = 100.0 / 52.0

    ax.axhline(
        uniform_level,
        color=TEAL,
        linewidth=1.5,
        linestyle="--",
        label="Uniform reference (100% / 52)",
    )

    top_indices = np.argsort(values)[-5:]

    for idx in top_indices:
        ax.annotate(
            f"{values[idx]:.2f}%",
            (ages[idx], values[idx]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=NAVY,
        )

    ax.set_xlim(-1, 52)

    ax.set_ylim(
        0,
        max(values.max() * 1.23, 3.0),
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(4)
    )

    ax.set_xlabel(
        "Weeks before latest observed demand "
        "(0 = most recent week)",
        labelpad=10,
    )

    ax.set_ylabel(
        "Relative aggregated attention (%)",
        labelpad=10,
    )

    ax.yaxis.grid(
        True,
        linewidth=0.8,
        color=GRID,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        length=0,
        pad=7,
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
    )

    fig.text(
        0.07,
        0.94,
        config["title"],
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )

    fig.text(
        0.07,
        0.88,
        config["subtitle"],
        fontsize=10,
        color="#5E6F7A",
    )

    add_figure_note(
        fig,
        "Source: native TFT attention interpretation "
        "aggregated over 250 validation windows. "
        "Attention indicates internal model weighting, "
        "not causality.",
    )

    return save_figure(
        fig,
        config["stem"],
    )


def write_report(inputs, outputs):
    report = {
        "status": "PASS",
        "step": 18,
        "model": "selected_no_price_tft",
        "interpretation_partition": "2024_validation",
        "validation_windows": 250,
        "encoder_weeks": 52,
        "forecast_horizon_weeks": 4,
        "figure_count": 4,
        "formats": ["png", "svg"],
        "png_dpi": DPI,
        "source_files": {
            name: str(path)
            for name, path in SOURCE_FILES.items()
        },
        "figures": outputs,
        "input_checks": inputs,
        "methodological_notes": [
            "All numerical values are read from Step 17.",
            "Variable-selection percentages are "
            "normalized within each feature group.",
            "Percentages across static, encoder and "
            "decoder groups cannot be compared directly.",
            "Temporal attention is aggregated across "
            "2024 validation forecast windows.",
            "Attention and variable selection are "
            "not causal explanations.",
            "No model training or test prediction "
            "was performed.",
        ],
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main():
    print(
        "PHASE 3 - STEP 18: "
        "TFT EXPLAINABILITY VISUALIZATIONS"
    )
    print("=" * 68)

    configure_plot_style()

    print("\nLOADING STEP 17 RESULTS")
    print("-" * 68)

    data = {}

    for category in (
        "static",
        "encoder",
        "decoder",
    ):
        data[category] = load_importance(
            SOURCE_FILES[category],
            category,
        )

        print(
            f"{category}: "
            f"{len(data[category])} variables - PASS"
        )

    data["attention"] = load_attention(
        SOURCE_FILES["attention"]
    )

    print(
        "attention: 52 historical weeks - PASS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {}

    print("\nGENERATING RESEARCH FIGURES")
    print("-" * 68)

    for category in (
        "static",
        "encoder",
        "decoder",
    ):
        outputs[category] = plot_importance(
            data[category],
            category,
        )

        print(
            f"{category}: PNG + SVG - PASS"
        )

    outputs["attention"] = plot_attention(
        data["attention"]
    )

    print("attention: PNG + SVG - PASS")

    checks = {
        "static_importance_total_percent": float(
            data["static"][
                "relative_importance_percent"
            ].sum()
        ),
        "encoder_importance_total_percent": float(
            data["encoder"][
                "relative_importance_percent"
            ].sum()
        ),
        "decoder_importance_total_percent": float(
            data["decoder"][
                "relative_importance_percent"
            ].sum()
        ),
        "attention_total_percent": float(
            data["attention"][
                "relative_attention_percent"
            ].sum()
        ),
    }

    write_report(
        checks,
        outputs,
    )

    print("\nGENERATED FIGURES")
    print("-" * 68)

    for category, paths in outputs.items():
        print(f"\n{category.upper()}")
        print("PNG:", paths["png"])
        print("SVG:", paths["svg"])

    print("\nVisualization report:")
    print(REPORT_FILE)

    print("\nMODEL RETRAINING: NONE")
    print("TEST EVALUATION: NOT REPEATED")
    print("TFT EXPLAINABILITY VISUALIZATIONS: PASS")


if __name__ == "__main__":
    main()
