
"""
Read-only audit support for eight-input TFT
factory observation corrections.

Security:
    Correction writes are disabled until
    authentication and authorization exist.

This module does not change:
    - The original factory training CSV
    - The frozen TFT model checkpoint
    - Existing MongoDB factory observations
    - Existing finalized forecasts

Future authorized correction transactions must
write the observation change and corresponding
audit event atomically.
"""

from datetime import datetime, timezone

from pymongo import DESCENDING

from app.db.mongodb import get_database

from app.services.eight_input_observation_storage_service import (
    TEA_GRADES,
)


AUDIT_COLLECTION_NAME = (
    "eight_input_factory_observation_audit"
)

CORRECTION_WRITES_ENABLED = False


def get_audit_collection():
    """Return the observation audit collection."""

    return get_database()[AUDIT_COLLECTION_NAME]


def get_correction_security_status():
    """
    Explain why observation corrections
    are currently unavailable.
    """

    return {
        "status": "success",
        "authentication_configured": False,
        "correction_writes_enabled": (
            CORRECTION_WRITES_ENABLED
        ),
        "correction_policy": (
            "authenticated_authorized_only"
        ),
        "message": (
            "Factory observation corrections are "
            "disabled until authentication and "
            "role-based authorization are implemented."
        ),
    }


def normalize_audit_document(document):
    """Convert MongoDB values to JSON-safe values."""

    item = dict(document)

    item.pop("_id", None)

    for key, value in list(item.items()):
        if isinstance(value, datetime):
            item[key] = value.isoformat()

    return item


def list_observation_audit_events(
    grade=None,
    week_start_date=None,
    limit=100,
):
    """
    Retrieve audit events without modifying data.

    Empty results indicate that no audit events
    match the supplied filters.
    """

    if grade is not None:
        if grade not in TEA_GRADES:
            raise ValueError(
                "Invalid tea grade."
            )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 500
    ):
        raise ValueError(
            "limit must be between 1 and 500."
        )

    if week_start_date is not None:
        from app.services.eight_input_observation_storage_service import (
            normalize_week_date,
        )

        week_start_date = (
            normalize_week_date(
                week_start_date
            ).isoformat()
        )

    query = {}

    if grade is not None:
        query["tea_type"] = grade

    if week_start_date is not None:
        query["week_start_date"] = (
            week_start_date
        )

    cursor = (
        get_audit_collection()
        .find(
            query,
            {"_id": 0},
        )
        .sort(
            [("created_at", DESCENDING)]
        )
        .limit(limit)
    )

    return [
        normalize_audit_document(document)
        for document in cursor
    ]


def request_observation_correction(
    *args,
    **kwargs,
):
    """
    Explicitly reject correction requests until
    authenticated authorization is implemented.
    """

    raise PermissionError(
        "Factory observation correction writes "
        "are disabled. Authentication and "
        "authorization must be implemented first."
    )


def apply_observation_correction(
    *args,
    **kwargs,
):
    """
    Never apply an unauthenticated correction.

    A future implementation must:
    - Verify the authenticated operator.
    - Verify permission to correct observations.
    - Validate the requested changes.
    - Preserve before/after values.
    - Insert an audit event.
    - Use an atomic MongoDB transaction.
    """

    raise PermissionError(
        "Factory observation correction writes "
        "are disabled. No observation was modified."
    )
