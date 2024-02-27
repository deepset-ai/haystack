import logging
import os
import sys
import typing
from typing import List, Optional
import builtins

import haystack.tracing.tracer

if typing.TYPE_CHECKING:
    from structlog.typing import Processor, WrappedLogger, EventDict

HAYSTACK_LOGGING_USE_JSON_ENV_VAR = "HAYSTACK_LOGGING_USE_JSON"
HAYSTACK_LOGGING_IGNORE_STRUCTLOG_ENV_VAR = "HAYSTACK_LOGGING_IGNORE_STRUCTLOG"


def correlate_logs_with_traces(_: "WrappedLogger", __: str, event_dict: "EventDict") -> "EventDict":
    """Add correlation data for logs.

    This is useful if you want to correlate logs with traces.
    """
    if not haystack.tracing.is_tracing_enabled():
        return event_dict

    current_span = haystack.tracing.tracer.current_span()
    if current_span:
        event_dict.update(current_span.get_correlation_data_for_logs())

    return event_dict


def configure_logging(use_json: Optional[bool] = None) -> None:
    """Configure logging for Haystack.

    - If `structlog` is not installed, we keep everything as it is. The user is responsible for configuring logging
      themselves.
    - If `structlog` is installed, we configure it to format log entries including its key-value data. To disable this
      behavior set the environment variable `HAYSTACK_LOGGING_IGNORE_STRUCTLOG` to `true`.
    - If `structlog` is installed, you can JSON format all logs. Enable this by
        - setting the `use_json` parameter to `True` when calling this function
        - setting the environment variable `HAYSTACK_LOGGING_USE_JSON` to `true`
    """
    try:
        import structlog
        from structlog.processors import ExceptionRenderer
        from structlog.tracebacks import ExceptionDictTransformer

    except ImportError:
        # structlog is not installed - fall back to standard logging
        return

    if os.getenv(HAYSTACK_LOGGING_IGNORE_STRUCTLOG_ENV_VAR, "false").lower() == "true":
        # If the user wants to ignore structlog, we don't configure it and fall back to standard logging
        return

    # We roughly follow the structlog documentation here:
    # https://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging
    # This means that we use structlog to format the log entries for entries emitted via `logging` and `structlog`.

    if use_json is None:  # explicit parameter takes precedence over everything else
        use_json_env_var = os.getenv(HAYSTACK_LOGGING_USE_JSON_ENV_VAR)
        if use_json_env_var is None:
            # Automatically enable JSON logging if stderr is not a TTY
            use_json = not (sys.stderr.isatty() or hasattr(builtins, "__IPYTHON__"))
        else:
            # User gave us an explicit value via environment variable
            use_json = use_json_env_var.lower() == "true"

    shared_processors: List[Processor] = [
        # Add the log level to the event_dict for structlog to use
        structlog.stdlib.add_log_level,
        # Adds the current timestamp in ISO format to logs
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if use_json:
        # We only need that in sophisticated production setups where we want to correlate logs with traces
        shared_processors.append(correlate_logs_with_traces)

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
        # This is a filter that will filter out log entries that are below the log level of the root logger.
        wrapper_class=structlog.make_filtering_bound_logger(min_level=logging.root.getEffectiveLevel()),
    )

    renderers: List[Processor]
    if use_json:
        renderers = [
            ExceptionRenderer(
                # don't show locals in production logs - this can be quite sensitive information
                ExceptionDictTransformer(show_locals=False)
            ),
            structlog.processors.JSONRenderer(),
        ]
    else:
        renderers = [structlog.dev.ConsoleRenderer()]

    formatter = structlog.stdlib.ProcessorFormatter(
        # These run ONLY on `logging` entries that do NOT originate within
        # structlog.
        foreign_pre_chain=shared_processors
        + [
            # Add the information from the `logging` `extras` to the event dictionary
            structlog.stdlib.ExtraAdder()
        ],
        # These run on ALL entries after the pre_chain is done.
        processors=[
            # Remove _record & _from_structlog. to avoid that this metadata is added to the final log record
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *renderers,
        ],
    )

    handler = logging.StreamHandler()
    handler.name = "HaystackLoggingHandler"
    # Use OUR `ProcessorFormatter` to format all `logging` entries.
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # avoid adding our handler twice
    old_handlers = [
        h
        for h in root_logger.handlers
        if not (isinstance(h, logging.StreamHandler) and h.name == "HaystackLoggingHandler")
    ]
    new_handlers = [handler, *old_handlers]
    root_logger.handlers = new_handlers
