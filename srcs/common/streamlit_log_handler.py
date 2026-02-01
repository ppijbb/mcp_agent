import logging
from collections import deque


class StreamlitLogHandler(logging.Handler):
    """
    A custom logging handler that redirects logs to a Streamlit container.
    """
    def __init__(self, container, max_lines=100):
        super().__init__()
        self.container = container
        # Use a deque to automatically manage the max number of log lines
        self.log_messages = deque(maxlen=max_lines)

    def emit(self, record):
        """
        Formats the log record and writes it to the Streamlit container.
        """
        msg = self.format(record)
        # Add new message to the top
        self.log_messages.appendleft(msg)
        # Update the container with all messages, newest first
        # Use st.code for better formatting of log messages
        self.container.code("\n".join(self.log_messages))


def setup_streamlit_logging(logger_names: list[str], container):
    """
    Configures a logger (or multiple loggers) to send its output to a Streamlit container.

    This function gets the specified loggers, creates a StreamlitLogHandler,
    and adds it to them, ensuring no duplicate handlers are added
    on script reruns.

    Args:
        logger_names: A list of logger names to capture.
        container: The Streamlit container widget to write logs to.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%H:%M:%S"
    )
    handler = StreamlitLogHandler(container)
    handler.setFormatter(formatter)

    for name in logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Check if a handler of our type is already attached to avoid duplicates
        if not any(isinstance(h, StreamlitLogHandler) and h.container == container for h in logger.handlers):
            logger.addHandler(handler)
