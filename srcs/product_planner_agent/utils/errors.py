class AgentError(Exception):
    """Base class for all custom agent exceptions."""


class MissingEnvError(AgentError):
    """Raised when required environment variables are missing."""


class ExternalServiceError(AgentError):
    """Raised when calls to external services (e.g., Figma, Notion APIs) fail.""" 