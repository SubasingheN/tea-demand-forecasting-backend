
"""
Compatibility imports for forecast revision storage.

Provides the module name expected by the
Forecast History API while retaining the
existing revision storage implementation.

No database writes or forecast generation.
"""

from app.services.eight_input_forecast_revision_storage_service import (
    REVISION_COLLECTION_NAME,
    get_revision_collection,
    get_current_forecasts,
    get_complete_original_set,
    list_forecast_revisions,
)
