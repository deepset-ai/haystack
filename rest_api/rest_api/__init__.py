import logging
import re
import sys
from types import TracebackType
from typing import Any, List, Optional, Type

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import ExceptionDictTransformer, ExceptionRenderer, JSONRenderer
from structlog.types import EventDict, WrappedLogger

from rest_api.config import LOG_FORMAT, LOG_LEVEL, LOGGING_LOCALS_MAX_STRING, LogFormatEnum


_FAST_API_MIDDLEWARE_REGEX = re.compile(r"(\/(starlette(_context)?|fastapi|anyio)\/|/rest_api/application_utils.py$)")
_ANYIO_EXC_NEGATIVE_LIST = ["EndOfStream", "WouldBlock"]


def _clean_fastapi_excs(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """
    Custom structlog processor that:
    - removes exceptions that are thrown by FastAPI and AnyIO as a consequence of async exception processing
    - removes locals of fastAPI middleware stack frames

    Note that when an exception is thrown in an FastAPI app, the exception is bubbled through the async FastAPI/AnyIO
    stack.
    In order to communicate between async tasks streams are used under the hood.
    To communicate task failures between tasks, special exceptions are thrown internally.
    These internal exceptions are handled by the FastAPI/AnyIO stack and exchanged for the original exception.
    They are not relevant to the user and can be safely removed from the logs to avoid cluttering.
    """
    exceptions = event_dict.get("exception", [])

    excs_to_drop = [exc for exc in exceptions if exc["exc_type"] in _ANYIO_EXC_NEGATIVE_LIST]
    for exc in excs_to_drop:
        exceptions.remove(exc)

    for exc in exceptions:
        for frame in exc.get("frames", []):
            filename = frame.get("filename", "")
            if _FAST_API_MIDDLEWARE_REGEX.search(filename):
                frame["locals"] = {}
    return event_dict


# Timestamper preprocessor that to add unified timestamps to each log
_timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

# Shared processors (structlog + python-logging) see e.g.: https://www.structlog.org/en/stable/standard-library.html
_shared_preprocessors: List[Any] = [structlog.stdlib.add_log_level, _timestamper]


def _build_renderers(
    log_format: LogFormatEnum, log_capture: Optional[Any] = None, locals_max_string: int = LOGGING_LOCALS_MAX_STRING
) -> List[Any]:
    """
    Build renderers based on the environment settings.
    Inserts a log_capture processor as the second last processor if one is provided.
    """
    renderers: List[Any]
    if log_format == LogFormatEnum.RAW:
        renderers = [ConsoleRenderer()]
    else:
        # ConsoleRenderer automatically obtains the exc and renders it on logger.exception()
        # JSONRenderer instead does not do this, so we need this processor
        renderers = [
            ExceptionRenderer(ExceptionDictTransformer(locals_max_string=locals_max_string)),
            _clean_fastapi_excs,
            JSONRenderer(),
        ]
    if log_capture:
        renderers.insert(-1, log_capture)
    return renderers


def build_structlog_processors(renderers: List[Any]) -> List[Any]:
    """
    Build structlog processors with the given renderers.
    """
    structlog_only_preprocessors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    processors = _shared_preprocessors + structlog_only_preprocessors + renderers
    return processors


def _configure_structlog(renderers: List[Any], log_level: int) -> None:
    """
    Configure structlog to use the processors and renderers defined in this module
    and to process logs on its own without using stdlib's logging.
    """
    structlog_processors = build_structlog_processors(renderers=renderers)
    structlog.configure(
        processors=structlog_processors,
        wrapper_class=structlog.make_filtering_bound_logger(min_level=log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def _build_stdlib_processors(renderers: List[Any]) -> List[Any]:
    """
    Build python logging processors with the given renderers.
    """
    stdlib_only_preprocessors: List[Any] = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]
    return _shared_preprocessors + stdlib_only_preprocessors + renderers


def _configure_stdlib_logging(renderers: List[Any], log_level: int) -> None:
    """
    Configure python logging to use structlog's processors with the given renderers.

    Note that structlog's plugin for python logging (ProcessorFormatter) typically is capable of processing both
    structlog and logging logs.
    structlog however is configured to not use stdlib's logging (see `_configure_structlog`).
    Hence the distinction between `foreign_pre_chain` (logging-originated logs only) and `processors` (both) is not
    required here.
    """
    stdlib_processors = _build_stdlib_processors(renderers=renderers)
    formatter = structlog.stdlib.ProcessorFormatter(processors=stdlib_processors)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove previously set handler to avoid logging twice and especially to avoid logging in the wrong format
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def _log_unhandled_exception(
    exc_type: Type[BaseException], exc_value: BaseException, exc_traceback: Optional[TracebackType]
) -> Any:
    """
    Exception handler to wrap exceptions into log messages and report
    them according to the logging configuration. This way it is possible to also report them as jsons.

    :param exc_type: Exception type
    :param exc_value: Exception value
    :param exc_traceback: Traceback
    """
    log = structlog.get_logger()
    if issubclass(exc_type, KeyboardInterrupt) and exc_traceback:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


def configure_logging(log_format: LogFormatEnum, log_capture: Optional[Any] = None, log_level: int = LOG_LEVEL) -> None:
    """
    Configures logging for the rest_api apps.

    :param log_capture: In order to capture and inspect logs as your final renderer receives them (e.g. for testing),
        you can pass a custom processor here.
        The value is expected to be a structlog processor that does not modify the event_dict.
    `configure_logging` places the value in the second last position of the processor chain right before the final
        renderer.
    """
    renderers = _build_renderers(log_format=log_format, log_capture=log_capture)
    _configure_structlog(renderers=renderers, log_level=log_level)
    _configure_stdlib_logging(renderers=renderers, log_level=log_level)
    sys.excepthook = _log_unhandled_exception


configure_logging(log_format=LOG_FORMAT)
