# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import functools
import logging
import os
import sys
import typing
from typing import Any, List, Optional

if typing.TYPE_CHECKING:
    from structlog.typing import EventDict, Processor, WrappedLogger

HAYSTACK_LOGGING_USE_JSON_ENV_VAR = "HAYSTACK_LOGGING_USE_JSON"
HAYSTACK_LOGGING_IGNORE_STRUCTLOG_ENV_VAR = "HAYSTACK_LOGGING_IGNORE_STRUCTLOG"


class PatchedLogger(typing.Protocol):
    """Class which enables using type checkers to find wrong logger usage."""

    def debug(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a debug message."""

    def info(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log an info message."""

    def warn(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a warning message."""

    def warning(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a warning message."""

    def error(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log an error message."""

    def critical(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a critical message."""

    def exception(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log an exception message."""

    def fatal(
        self,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a fatal message."""

    def log(
        self,
        level: int,
        msg: str,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a message."""

    def setLevel(self, level: int) -> None:
        """Set the logging level."""


def patch_log_method_to_kwargs_only(func: typing.Callable) -> typing.Callable:
    """A decorator to make sure that a function is only called with keyword arguments."""

    @functools.wraps(func)
    def _log_only_with_kwargs(
        msg, *, _: Any = None, exc_info: Any = None, stack_info: Any = False, stacklevel: int = 1, **kwargs: Any
    ) -> Any:  # we need the `_` to avoid a syntax error
        existing_extra = kwargs.pop("extra", {})
        return func(
            # we need to increase the stacklevel by 1 to point to the correct caller
            # (otherwise it points to this function)
            msg,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
            extra={**existing_extra, **kwargs},
        )

    return _log_only_with_kwargs


def patch_log_with_level_method_to_kwargs_only(func: typing.Callable) -> typing.Callable:
    """A decorator to make sure that a function is only called with keyword arguments."""

    @functools.wraps(func)
    def _log_only_with_kwargs(
        level,
        msg,
        *,
        _: Any = None,
        exc_info: Any = None,
        stack_info: Any = False,
        stacklevel: int = 1,
        **kwargs: Any,  # we need the `_` to avoid a syntax error
    ) -> Any:
        existing_extra = kwargs.pop("extra", {})

        return func(
            level,
            msg,
            exc_info=exc_info,
            stack_info=stack_info,
            # we need to increase the stacklevel by 1 to point to the correct caller
            # (otherwise it points to this function)
            stacklevel=stacklevel + 1,
            extra={**existing_extra, **kwargs},
        )

    return _log_only_with_kwargs


def patch_make_records_to_use_kwarg_string_interpolation(original_make_records: typing.Callable) -> typing.Callable:
    """A decorator to ensure string interpolation is used."""

    @functools.wraps(original_make_records)
    def _wrapper(name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None) -> Any:
        safe_extra = extra or {}
        try:
            interpolated_msg = msg.format(**safe_extra)
        except (KeyError, ValueError):
            interpolated_msg = msg
        return original_make_records(name, level, fn, lno, interpolated_msg, (), exc_info, func, extra, sinfo)

    return _wrapper


def _patch_structlog_call_information(logger: logging.Logger) -> None:
    # structlog patches the findCaller to hide itself from the traceback.
    # We need to patch their patch to hide `haystack.logging` from the traceback.
    try:
        from structlog._frames import _find_first_app_frame_and_name, _format_stack
        from structlog.stdlib import _FixedFindCallerLogger

        if not isinstance(logger, _FixedFindCallerLogger):
            return

        # completely copied from structlog. We only add `haystack.logging` to the list of ignored frames
        # pylint: disable=unused-variable
        def findCaller(stack_info: bool = False, stacklevel: int = 1) -> typing.Tuple[str, int, str, Optional[str]]:
            try:
                sinfo: Optional[str]
                # we need to exclude `haystack.logging` from the stack
                f, name = _find_first_app_frame_and_name(["logging", "haystack.logging"])
                sinfo = _format_stack(f) if stack_info else None
            except Exception as error:
                print(f"Error in findCaller: {error}")

            return f.f_code.co_filename, f.f_lineno, f.f_code.co_name, sinfo

        logger.findCaller = findCaller  # type: ignore
    except ImportError:
        pass


def getLogger(name: str) -> PatchedLogger:
    """
    Get the Haystack logger, a patched version of the one from the standard library.

    We patch the default logger methods to make sure that they are only called with keyword arguments.
    We enforce keyword-arguments because
        - it brings in consistency
        - it makes structure logging effective, not just an available feature
    """
    logger = logging.getLogger(name)
    logger.debug = patch_log_method_to_kwargs_only(logger.debug)  # type: ignore
    logger.info = patch_log_method_to_kwargs_only(logger.info)  # type: ignore
    logger.warn = patch_log_method_to_kwargs_only(logger.warn)  # type: ignore
    logger.warning = patch_log_method_to_kwargs_only(logger.warning)  # type: ignore
    logger.error = patch_log_method_to_kwargs_only(logger.error)  # type: ignore
    logger.critical = patch_log_method_to_kwargs_only(logger.critical)  # type: ignore
    logger.exception = patch_log_method_to_kwargs_only(logger.exception)  # type: ignore
    logger.fatal = patch_log_method_to_kwargs_only(logger.fatal)  # type: ignore
    logger.log = patch_log_with_level_method_to_kwargs_only(logger.log)  # type: ignore

    _patch_structlog_call_information(logger)

    # We also patch the `makeRecord` method to use keyword string interpolation
    logger.makeRecord = patch_make_records_to_use_kwarg_string_interpolation(logger.makeRecord)  # type: ignore

    return typing.cast(PatchedLogger, logger)


def add_line_and_file(_: "WrappedLogger", __: str, event_dict: "EventDict") -> "EventDict":
    """Add line and file to log entries."""
    stdlib_record = event_dict.get("_record")
    if not stdlib_record:
        return event_dict

    event_dict["lineno"] = stdlib_record.lineno
    event_dict["module"] = stdlib_record.name

    return event_dict


def correlate_logs_with_traces(_: "WrappedLogger", __: str, event_dict: "EventDict") -> "EventDict":
    """
    Add correlation data for logs.

    This is useful if you want to correlate logs with traces.
    """
    import haystack.tracing.tracer  # to avoid circular imports

    if not haystack.tracing.is_tracing_enabled():
        return event_dict

    current_span = haystack.tracing.tracer.current_span()
    if current_span:
        event_dict.update(current_span.get_correlation_data_for_logs())

    return event_dict


def configure_logging(use_json: Optional[bool] = None) -> None:
    """
    Configure logging for Haystack.

    - If `structlog` is not installed, we keep everything as it is. The user is responsible for configuring logging
      themselves.
    - If `structlog` is installed, we configure it to format log entries including its key-value data. To disable this
      behavior set the environment variable `HAYSTACK_LOGGING_IGNORE_STRUCTLOG` to `true`.
    - If `structlog` is installed, you can JSON format all logs. Enable this by
        - setting the `use_json` parameter to `True` when calling this function
        - setting the environment variable `HAYSTACK_LOGGING_USE_JSON` to `true`
    """
    import haystack.utils.jupyter  # to avoid circular imports

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
            # We try to guess if we are in an interactive terminal or not
            interactive_terminal = (
                sys.stderr.isatty() or hasattr(builtins, "__IPYTHON__") or haystack.utils.jupyter.is_in_jupyter()
            )
            use_json = not interactive_terminal
        else:
            # User gave us an explicit value via environment variable
            use_json = use_json_env_var.lower() == "true"

    shared_processors: List[Processor] = [
        # Add the log level to the event_dict for structlog to use
        structlog.stdlib.add_log_level,
        # Adds the current timestamp in ISO format to logs
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.contextvars.merge_contextvars,
        add_line_and_file,
    ]

    if use_json:
        # We only need that in sophisticated production setups where we want to correlate logs with traces
        shared_processors.append(correlate_logs_with_traces)

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(ignore_frame_names=["haystack.logging"]),
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
