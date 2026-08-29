from app.services.evaluation_service import (
    load_trained_model,
)
from app.services.weekly_forecast_service import (
    create_future_prediction_dataset,
)


def generate_tft_interpretation_from_dataset(
    future_dataset,
):
    """
    Generate TFT interpretation for any
    compatible prediction dataset.

    This allows the same explainability
    logic to be reused for both:
    1. Original research forecasts
    2. Latest operational forecasts
    """

    model = load_trained_model()

    dataloader = future_dataset.to_dataloader(
        train=False,
        batch_size=5,
        num_workers=0,
    )

    raw_predictions = model.predict(
        dataloader,
        mode="raw",
        return_x=True,
    )

    interpretation = model.interpret_output(
        raw_predictions.output,
        reduction="sum",
    )

    return {
        "interpretation": interpretation,
        "raw_predictions": raw_predictions,
    }


def generate_tft_interpretation():
    """
    Existing research forecast
    interpretation.

    This function is kept so the current
    /explainability/summary endpoint
    continues working.
    """

    future_dataset = (
        create_future_prediction_dataset()
    )

    return (
        generate_tft_interpretation_from_dataset(
            future_dataset
        )
    )


def _format_variable_importance(
    variable_names,
    importance_tensor,
):
    values = (
        importance_tensor
        .detach()
        .cpu()
        .flatten()
        .tolist()
    )

    results = []

    for name, value in zip(
        variable_names,
        values,
    ):
        results.append(
            {
                "variable": name,
                "importance": round(
                    float(value),
                    6,
                ),
            }
        )

    results = sorted(
        results,
        key=lambda item:
            item["importance"],
        reverse=True,
    )

    total_importance = sum(
        item["importance"]
        for item in results
    )

    if total_importance > 0:
        for item in results:
            item[
                "importance_percentage"
            ] = round(
                (
                    item["importance"]
                    / total_importance
                )
                * 100,
                2,
            )

    return results


def format_explainability_summary(
    interpretation,
):
    """
    Convert raw TFT interpretation
    tensors into frontend-friendly
    values.
    """

    model = load_trained_model()

    encoder_importance = (
        _format_variable_importance(
            model.encoder_variables,
            interpretation[
                "encoder_variables"
            ],
        )
    )

    decoder_importance = (
        _format_variable_importance(
            model.decoder_variables,
            interpretation[
                "decoder_variables"
            ],
        )
    )

    static_importance = (
        _format_variable_importance(
            model.static_variables,
            interpretation[
                "static_variables"
            ],
        )
    )

    attention = interpretation[
        "attention"
    ]

    attention_values = (
        attention
        .detach()
        .cpu()
        .flatten()
        .tolist()
    )

    return {
        "encoder_variable_importance":
            encoder_importance,

        "decoder_variable_importance":
            decoder_importance,

        "static_variable_importance":
            static_importance,

        "attention_values": [
            round(
                float(value),
                6,
            )
            for value
            in attention_values
        ],
    }


def get_explainability_summary():
    """
    Existing research explainability
    summary.
    """

    result = (
        generate_tft_interpretation()
    )

    return format_explainability_summary(
        result["interpretation"]
    )


def get_explainability_summary_from_dataset(
    future_dataset,
):
    """
    Generate explainability for a
    provided TFT prediction dataset.

    This will be used by the latest
    operational forecast flow.
    """

    result = (
        generate_tft_interpretation_from_dataset(
            future_dataset
        )
    )

    return format_explainability_summary(
        result["interpretation"]
    )