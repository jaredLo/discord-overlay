"""API key authentication for WebSocket upgrade."""

from starlette.requests import Request
from server.config import API_KEY


def verify_api_key(request: Request) -> bool:
    """Check API key from query param or Authorization header.

    Returns True if auth is valid or no key is configured (dev mode).
    """
    if not API_KEY:
        return True

    # Check query param first (convenient for WebSocket clients)
    key = request.query_params.get("api_key", "")
    if key == API_KEY:
        return True

    # Check Authorization: Bearer <key>
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and auth[7:].strip() == API_KEY:
        return True

    return False
