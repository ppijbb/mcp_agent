"""Structured logging configuration.

If `AGENT_JSON_LOG=1`, logs are emitted in JSON. If `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
OpenTelemetry handler is configured for trace & log export (best-effort, no external deps if not installed).
"""
from __future__ import annotations

import os, sys, logging, json
from logging import LogRecord

_JSON = bool(int(os.getenv("AGENT_JSON_LOG", "0")))


class _JsonFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        base = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    handler = logging.StreamHandler(sys.stdout)
    if _JSON:
        handler.setFormatter(_JsonFormatter())
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # --- optional OTEL (opt-in) ---
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") if os.getenv("AGENT_OTEL", "0") == "1" else None
    if endpoint:
        try:
            from opentelemetry.sdk._logs import LoggingHandler, LoggerProvider
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, OTLPLogExporter
            provider = LoggerProvider()
            logging.getLogger(__name__).info("OpenTelemetry logging enabled → %s", endpoint)
            exporter = OTLPLogExporter(endpoint=endpoint)
            provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
            otel_handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
            root.addHandler(otel_handler)
        except ImportError:
            logging.getLogger(__name__).warning("opentelemetry-sdk not installed, OTEL logging disabled.")


# auto-initialize
setup_logging()
