import builtins
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, Mock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from haystack import logging as haystack_logging
from test.tracing.utils import SpyingTracer
import haystack.utils.jupyter


@pytest.fixture(autouse=True)
def reset_logging_config() -> None:
    old_handlers = logging.root.handlers.copy()
    yield
    # Reset the logging configuration after each test to avoid impacting other tests
    logging.root.handlers = old_handlers


class TestSkipLoggingConfiguration:
    def test_skip_logging_configuration(
        self, monkeypatch: MonkeyPatch, capfd: CaptureFixture, caplog: LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_IGNORE_STRUCTLOG", "true")
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # the pytest fixture caplog only captures logs being rendered from the stdlib logging module
        assert caplog.messages == ["Hello, structured logging!"]

        # Nothing should be captured by capfd since structlog is not configured
        assert capfd.readouterr().err == ""

    def test_skip_logging_if_structlog_not_installed(
        self, monkeypatch: MonkeyPatch, capfd: CaptureFixture, caplog: LogCaptureFixture
    ) -> None:
        monkeypatch.delitem(sys.modules, "structlog", raising=False)
        monkeypatch.setattr(builtins, "__import__", Mock(side_effect=ImportError))

        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # the pytest fixture caplog only captures logs being rendered from the stdlib logging module
        assert caplog.messages == ["Hello, structured logging!"]

        # Nothing should be captured by capfd since structlog is not configured
        assert capfd.readouterr().err == ""


class TestStructuredLoggingConsoleRendering:
    def test_log_filtering_when_using_debug(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.debug("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        assert output == ""

    def test_log_filtering_when_using_debug_and_log_level_is_debug(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        logger.debug("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        assert "Hello, structured logging" in output
        assert "{" not in output, "Seems JSON rendering is enabled when it should not be"

    def test_console_rendered_structured_log_even_if_no_tty_but_python_config(
        self, capfd: CaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)

        haystack_logging.configure_logging(use_json=False)

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert "Hello, structured logging!" in output
        assert "{" not in output, "Seems JSON rendering is enabled when it should not be"

    def test_console_rendered_structured_log_if_in_ipython(
        self, capfd: CaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(builtins, "__IPYTHON__", "true", raising=False)

        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert "Hello, structured logging!" in output
        assert "{" not in output, "Seems JSON rendering is enabled when it should not be"

    def test_console_rendered_structured_log_even_in_jupyter(
        self, capfd: CaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(haystack.utils.jupyter, haystack.utils.jupyter.is_in_jupyter.__name__, lambda: True)

        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert "Hello, structured logging!" in output
        assert "{" not in output, "Seems JSON rendering is enabled when it should not be"

    def test_console_rendered_structured_log_even_if_no_tty_but_forced_through_env(
        self, capfd: CaptureFixture, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_USE_JSON", "false")

        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert "Hello, structured logging!" in output
        assert "{" not in output, "Seems JSON rendering is enabled when it should not be"

    def test_console_rendered_structured_log(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        # Only check for the minute to be a bit more robust
        today = datetime.now(tz=timezone.utc).isoformat(timespec="minutes").replace("+00:00", "")
        assert today in output

        log_level = "warning"
        assert log_level in output

        assert "Hello, structured logging!" in output

        assert "key1" in output
        assert "value1" in output

    def test_logging_exceptions(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)

        def function_that_raises_and_adds_to_stack_trace():
            raise ValueError("This is an error")

        try:
            function_that_raises_and_adds_to_stack_trace()
        except ValueError:
            logger.exception("An error happened")

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert "An error happened" in output


class TestStructuredLoggingJSONRendering:
    def test_logging_as_json_if_not_atty(self, capfd: CaptureFixture, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        parsed_output = json.loads(output)  # should not raise an error

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key1": "value1",
            "key2": "value2",
            "level": "warning",
            "timestamp": ANY,
        }

    def test_logging_as_json(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        parsed_output = json.loads(output)  # should not raise an error

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key1": "value1",
            "key2": "value2",
            "level": "warning",
            "timestamp": ANY,
        }

    def test_logging_as_json_enabling_via_env(self, capfd: CaptureFixture, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_USE_JSON", "true")
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        parsed_output = json.loads(output)  # should not raise an error

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key1": "value1",
            "key2": "value2",
            "level": "warning",
            "timestamp": ANY,
        }

    def test_logging_exceptions_json(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger(__name__)

        def function_that_raises_and_adds_to_stack_trace():
            my_local_variable = "my_local_variable"  # noqa: F841
            raise ValueError("This is an error")

        try:
            function_that_raises_and_adds_to_stack_trace()
        except ValueError:
            logger.exception("An error happened ")

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        parsed_output = json.loads(output)
        assert parsed_output == {
            "event": "An error happened ",
            "level": "error",
            "timestamp": ANY,
            "exception": [
                {
                    "exc_type": "ValueError",
                    "exc_value": "This is an error",
                    "syntax_error": None,
                    "is_cause": False,
                    "frames": [
                        {
                            "filename": str(Path.cwd() / "test" / "test_logging.py"),
                            "lineno": ANY,  # otherwise the test breaks if you add a line :-)
                            "name": "test_logging_exceptions_json",
                            "line": "",
                            "locals": None,
                        },
                        {
                            "filename": str(Path.cwd() / "test" / "test_logging.py"),
                            "lineno": ANY,  # otherwise the test breaks if you add a line :-)
                            "name": "function_that_raises_and_adds_to_stack_trace",
                            "line": "",
                            "locals": None,
                        },
                    ],
                }
            ],
        }


class TestLogTraceCorrelation:
    def test_trace_log_correlation_python_logs_with_console_rendering(
        self, spying_tracer: SpyingTracer, capfd: CaptureFixture
    ) -> None:
        haystack_logging.configure_logging(use_json=False)

        with spying_tracer.trace("test-operation"):
            logger = logging.getLogger(__name__)
            logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        assert "trace_id" not in output

    def test_trace_log_correlation_python_logs(self, spying_tracer: SpyingTracer, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        with spying_tracer.trace("test-operation") as span:
            logger = logging.getLogger(__name__)
            logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        parsed_output = json.loads(output)

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key1": "value1",
            "key2": "value2",
            "level": "warning",
            "timestamp": ANY,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
        }

    def test_trace_log_correlation_no_span(self, spying_tracer: SpyingTracer, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger(__name__)

        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        parsed_output = json.loads(output)

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key1": "value1",
            "key2": "value2",
            "level": "warning",
            "timestamp": ANY,
        }

    def test_trace_log_correlation_no_tracer(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger(__name__)

        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        parsed_output = json.loads(output)

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key1": "value1",
            "key2": "value2",
            "level": "warning",
            "timestamp": ANY,
        }
