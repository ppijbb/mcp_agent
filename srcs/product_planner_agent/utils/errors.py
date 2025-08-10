class AgentError(Exception):
    """Base class for all custom agent exceptions."""


class MissingEnvError(AgentError):
    """Raised when required environment variables are missing."""


class ExternalServiceError(AgentError):
    """Raised when calls to external services (e.g., Figma, Notion APIs) fail.""" 


class WorkflowError(AgentError):
    """Raised when an agent workflow fails due to invalid inputs or processing errors."""