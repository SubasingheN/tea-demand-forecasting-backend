
from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.data import router as data_router
from app.api.training import router as training_router
from app.api.weekly_input import router as weekly_input_router
from app.api.forecast import router as forecast_router
from app.api.explainability import router as explainability_router
from app.api.alerts import router as alerts_router
from app.api.recommendations import router as recommendations_router
from app.api.demo import router as demo_router

from app.api.research_forecast import (
    router as research_forecast_router,
)

from app.api.research_selected_forecast import (
    router as research_selected_forecast_router,
)

from app.api.research_forecast_comparison import (
    router as research_forecast_comparison_router,
)

from app.api.six_input_forecast import (
    router as six_input_forecast_router,
)

from app.api.six_input_observations import (
    router as six_input_observations_router,
)

from app.api.six_input_history import (
    router as six_input_history_router,
)


api_router = APIRouter()

# Existing application endpoints
api_router.include_router(health_router)
api_router.include_router(data_router)
api_router.include_router(training_router)
api_router.include_router(weekly_input_router)
api_router.include_router(forecast_router)
api_router.include_router(explainability_router)
api_router.include_router(alerts_router)
api_router.include_router(recommendations_router)
api_router.include_router(demo_router)

# Original research forecasting
api_router.include_router(research_forecast_router)

# Existing selective three-input TFT
api_router.include_router(
    research_selected_forecast_router
)

# Existing forecast comparison
api_router.include_router(
    research_forecast_comparison_router
)

# New six-input TFT forecasting
api_router.include_router(
    six_input_forecast_router
)

api_router.include_router(
    six_input_observations_router
)

api_router.include_router(
    six_input_history_router
)