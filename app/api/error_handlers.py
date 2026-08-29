from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    errors = []

    for error in exc.errors():
        field_path = ".".join(
            str(item)
            for item in error.get(
                "loc",
                []
            )
            if item != "body"
        )

        errors.append(
            {
                "field": field_path,
                "message": error.get(
                    "msg",
                    "Invalid value",
                ),
                "type": error.get(
                    "type",
                    "validation_error",
                ),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": (
                "Input validation failed."
            ),
            "errors": errors,
        },
    )