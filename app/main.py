from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

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


# Frontend applications allowed to access the API
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


app.include_router(api_router)


@app.get("/")
def home():
    return {
        "message": (
            "Tea Demand Forecasting API is running"
        ),
        "status": "success",
    }