from warnings import warn

# Primary, fully-supported agent
from .urban_hive_agent import UrbanHiveMCPAgent

# ---------------------------------------------------------------------------
# Legacy agents (deprecated)
# These remain importable for backward compatibility, but will be removed in a
# future release.  Importing them raises a DeprecationWarning at runtime so that
# downstream code can migrate to UrbanHiveMCPAgent.
# ---------------------------------------------------------------------------

try:
    from .resource_matcher_agent import ResourceMatcherAgent  # type: ignore
    warn(
        "ResourceMatcherAgent is deprecated and will be removed in a future "
        "version. Please migrate to UrbanHiveMCPAgent.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:  # pragma: no cover
    # The source file may be removed in a later phase
    ResourceMatcherAgent = None  # type: ignore

try:
    from .social_connector_agent import SocialConnectorAgent  # type: ignore
    warn(
        "SocialConnectorAgent is deprecated and will be removed in a future "
        "version. Please migrate to UrbanHiveMCPAgent.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:  # pragma: no cover
    SocialConnectorAgent = None  # type: ignore

try:
    from .urban_analyst_agent import UrbanAnalystAgent  # type: ignore
    warn(
        "UrbanAnalystAgent is deprecated and will be removed in a future "
        "version. Please migrate to UrbanHiveMCPAgent.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError:  # pragma: no cover
    UrbanAnalystAgent = None  # type: ignore

__all__ = [
    "UrbanHiveMCPAgent",
    # Deprecated exports kept temporarily for compatibility
    "ResourceMatcherAgent",
    "SocialConnectorAgent",
    "UrbanAnalystAgent",
] 