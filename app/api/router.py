
"""
Main API router for Tea Demand Forecasting.

Preserves the existing application, research,
six-input TFT and eight-input TFT endpoints.

Registers each child APIRoute directly without
duplicating its existing path prefix.

No database or model operations occur here.
"""

from fastapi import APIRouter
from fastapi.routing import APIRoute

# Main application endpoints
from app.api.health import router as health_router
from app.api.data import router as data_router
from app.api.training import router as training_router
from app.api.weekly_input import router as weekly_input_router
from app.api.forecast import router as forecast_router
from app.api.explainability import router as explainability_router
from app.api.alerts import router as alerts_router
from app.api.recommendations import router as recommendations_router
from app.api.demo import router as demo_router

# Existing research endpoints
from app.api.research_forecast import (
    router as research_forecast_router,
)

from app.api.research_selected_forecast import (
    router as research_selected_forecast_router,
)

from app.api.research_forecast_comparison import (
    router as research_forecast_comparison_router,
)

# Existing six-input TFT endpoints
from app.api.six_input_forecast import (
    router as six_input_forecast_router,
)

from app.api.six_input_observations import (
    router as six_input_observations_router,
)

from app.api.six_input_history import (
    router as six_input_history_router,
)

# Final eight-input TFT forecasting
from app.api.eight_input_forecast import (
    router as eight_input_forecast_router,
)

# Eight-input factory observations
from app.api.eight_input_observations import (
    router as eight_input_observations_router,
)

# Eight-input historical data retrieval
from app.api.eight_input_history import (
    router as eight_input_history_router,
)

# Eight-input final forecast history
from app.api.eight_input_forecast_history import (
    router as eight_input_forecast_history_router,
)

# Genuine eight-input TFT explainability
from app.api.eight_input_explainability import (
    router as eight_input_explainability_router,
)

# Factory observation audit API
from app.api.eight_input_observation_audit import (
    router as eight_input_observation_audit_router,
)


api_router = APIRouter()


ALL_CHILD_ROUTERS = (
    health_router,
    data_router,
    training_router,
    weekly_input_router,
    forecast_router,
    explainability_router,
    alerts_router,
    recommendations_router,
    demo_router,
    research_forecast_router,
    research_selected_forecast_router,
    research_forecast_comparison_router,
    six_input_forecast_router,
    six_input_observations_router,
    six_input_history_router,
    eight_input_forecast_router,
    eight_input_observations_router,
    eight_input_history_router,
    eight_input_forecast_history_router,
    eight_input_explainability_router,
    eight_input_observation_audit_router,
)


def register_child_routes(parent, children):
    """
    Register child API routes directly.

    Route paths already contain the prefixes
    configured by their original APIRouters.
    """

    registered = set()

    for child in children:
        for route in child.routes:

            if not isinstance(route, APIRoute):
                raise RuntimeError(
                    "Unexpected nested route type: "
                    f"{type(route).__name__}"
                )

            # Do not add child.prefix here.
            # route.path already contains it.
            full_path = route.path

            route_key = (
                full_path,
                tuple(sorted(route.methods or [])),
            )

            if route_key in registered:
                raise RuntimeError(
                    "Duplicate API route registration: "
                    f"{route_key}"
                )

            registered.add(route_key)

            parent.add_api_route(
                path=full_path,
                endpoint=route.endpoint,
                methods=list(route.methods),
                response_model=route.response_model,
                status_code=route.status_code,
                tags=[
                    *child.tags,
                    *(route.tags or []),
                ],
                dependencies=[
                    *child.dependencies,
                    *route.dependencies,
                ],
                summary=route.summary,
                description=route.description,
                response_description=(
                    route.response_description
                ),
                responses={
                    **child.responses,
                    **route.responses,
                },
                deprecated=(
                    child.deprecated
                    or route.deprecated
                ),
                operation_id=route.operation_id,
                response_model_include=(
                    route.response_model_include
                ),
                response_model_exclude=(
                    route.response_model_exclude
                ),
                response_model_by_alias=(
                    route.response_model_by_alias
                ),
                response_model_exclude_unset=(
                    route.response_model_exclude_unset
                ),
                response_model_exclude_defaults=(
                    route.response_model_exclude_defaults
                ),
                response_model_exclude_none=(
                    route.response_model_exclude_none
                ),
                include_in_schema=(
                    child.include_in_schema
                    and route.include_in_schema
                ),
                response_class=route.response_class,
                name=route.name,
                callbacks=route.callbacks,
                openapi_extra=route.openapi_extra,
                generate_unique_id_function=(
                    route.generate_unique_id_function
                ),
            )


register_child_routes(
    api_router,
    ALL_CHILD_ROUTERS,
)
