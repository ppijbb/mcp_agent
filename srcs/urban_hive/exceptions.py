"""Custom exceptions for the urban hive module."""

class ExternalDataUnavailableError(Exception):
    """Raised when all external data sources fail or return no data."""

    def __init__(self, message: str):
        super().__init__(message)
