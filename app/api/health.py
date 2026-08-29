from fastapi import (
    APIRouter,
    HTTPException,
)

from app.db.mongodb import (
    check_database_connection,
    MONGODB_DATABASE,
)


router = APIRouter()


@router.get(
    "/health"
)
def health_check():
    return {
        "status": "healthy",
        "service":
            "Tea Demand Forecasting API",
    }


@router.get(
    "/health/database"
)
def database_health_check():
    try:
        check_database_connection()

        return {
            "status": "healthy",
            "database":
                "MongoDB Atlas",
            "database_name":
                MONGODB_DATABASE,
            "connection":
                "connected",
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "MongoDB Atlas connection "
                "failed: "
                f"{str(error)}"
            ),
        )