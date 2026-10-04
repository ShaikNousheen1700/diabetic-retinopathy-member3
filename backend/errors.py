"""API error type and the JSON shape every error response uses.

Every error looks like:
    {"success": false, "error": {"code": "UNSUPPORTED_FILE_TYPE", "message": "..."}}
so the frontend can always read `error.code` (for logic) and `error.message` (to show the user).
"""


class APIError(Exception):
    """Raised anywhere in the backend; main.py turns it into a JSON response."""

    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_body(code, message):
    return {"success": False, "error": {"code": code, "message": message}}
