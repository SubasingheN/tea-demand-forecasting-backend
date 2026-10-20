
"""
Main FastAPI application for Tea Demand Forecasting.

Registers all existing API endpoints directly while
preserving their request validation, response models,
dependencies, and OpenAPI configuration.

No model training or database writes occur here.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from app.api.router import api_router
from app.api.error_handlers import (
    validation_exception_handler,
)


app = FastAPI(
    title="Tea Demand Forecasting API",
    description=(
        "Backend API for the TFT-based weekly "
        "tea demand forecasting system"
    ),
    version="1.0.0",
)


# Frontend applications allowed to access the API.
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)


def register_application_routes():
    """
    Register existing API route objects directly.

    This avoids the nested-router inclusion issue
    detected in the current FastAPI environment.
    """

    registered = set()

    for route in api_router.routes:
        if not isinstance(route, APIRoute):
            raise RuntimeError(
                "Unexpected route type in main API router: "
                f"{type(route).__name__}"
            )

        route_key = (
            route.path,
            tuple(sorted(route.methods or [])),
        )

        if route_key in registered:
            raise RuntimeError(
                f"Duplicate application route: {route_key}"
            )

        registered.add(route_key)

        app.router.routes.append(route)


register_application_routes()


@app.get("/")
def home():
    return {
        "message": (
            "Tea Demand Forecasting API is running"
        ),
        "status": "success",
    }
