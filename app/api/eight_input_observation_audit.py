
"""
Read-only API for eight-input TFT factory observation audits.

Endpoints:
    GET /factory/eight-input/audit/status
    GET /factory/eight-input/audit/events

Security:
    Correction writes remain disabled because
    authentication and authorization have not
    been implemented.

    Audit event history must not be publicly
    exposed once sensitive correction records
    or operator identities are stored.

No POST, PUT, PATCH or DELETE correction
endpoints are registered here.
"""

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)
from pymongo.errors import PyMongoError

from app.services.eight_input_observation_audit_service import (
    get_correction_security_status,
)


router = APIRouter(
    prefix="/factory/eight-input/audit",
    tags=["Eight-Input Factory Observation Audit"],
    include_in_schema=False,
)


@router.get("/status")
def get_observation_audit_status():
    """
    Return the correction security configuration.

    This endpoint performs no database writes.
    """

    return {
        **get_correction_security_status(),
        "model": "eight_input_tft",
        "audit_history_access": "disabled_until_authentication",
        "write_performed": False,
    }


@router.get("/events")
def get_observation_audit_events(
    grade: str | None = Query(default=None),
    week_start_date: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    Deny audit history access until authentication
    and authorization are implemented.

    Query parameters are reserved for the future
    protected audit-history implementation.
    """

    raise HTTPException(
        status_code=403,
        detail=(
            "Factory observation audit history "
            "requires authenticated and authorized access. "
            "No audit records were disclosed."
        ),
    )
