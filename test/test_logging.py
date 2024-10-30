# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import builtins
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import structlog

from test.tracing.utils import SpyingTracer
from unittest.mock import ANY

import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import haystack.utils.jupyter
from haystack import logging as haystack_logging


@pytest.fixture(autouse=True)
def reset_logging_config() -> None:
    old_handlers = logging.root.handlers.copy()
    yield
    # Reset the logging configuration after each test to avoid impacting other tests
    logging.root.handlers = old_handlers


@pytest.fixture()
def set_context_var_key() -> Generator[str, None, None]:
    structlog.contextvars.bind_contextvars(context_var="value")

    yield "context_var"

    structlog.contextvars.unbind_contextvars("context_var")


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


class TestStructuredLoggingConsoleRendering:
    def test_log_filtering_when_using_debug(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=False)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.debug("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        assert output == ""

    def test_log_filtering_when_using_debug_and_log_level_is_debug(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=False)

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

    def test_logging_of_contextvars(self, capfd: CaptureFixture, set_context_var_key: str) -> None:
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert set_context_var_key in output


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
            "lineno": ANY,
            "module": "test.test_logging",
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
            "lineno": ANY,
            "module": "test.test_logging",
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
            "lineno": ANY,
            "module": "test.test_logging",
        }

    def test_logging_of_contextvars(
        self, capfd: CaptureFixture, monkeypatch: MonkeyPatch, set_context_var_key: str
    ) -> None:
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
            set_context_var_key: "value",
            "level": "warning",
            "timestamp": ANY,
            "lineno": ANY,
            "module": "test.test_logging",
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
            "lineno": ANY,
            "module": "test.test_logging",
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
                        },
                        {
                            "filename": str(Path.cwd() / "test" / "test_logging.py"),
                            "lineno": ANY,  # otherwise the test breaks if you add a line :-)
                            "name": "function_that_raises_and_adds_to_stack_trace",
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
            "lineno": ANY,
            "module": "test.test_logging",
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
            "lineno": ANY,
            "module": "test.test_logging",
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
            "lineno": ANY,
            "module": "test.test_logging",
        }


class TestCompositeLogger:
    def test_correct_stack_level_with_stdlib_rendering(
        self, monkeypatch: MonkeyPatch, capfd: CaptureFixture, caplog: LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_IGNORE_STRUCTLOG", "true")
        haystack_logging.configure_logging()

        logger = logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # the pytest fixture caplog only captures logs being rendered from the stdlib logging module
        assert caplog.messages == ["Hello, structured logging!"]
        assert caplog.records[0].name == "test.test_logging"

        # Nothing should be captured by capfd since structlog is not configured
        assert capfd.readouterr().err == ""

    def test_correct_stack_level_with_consoler_rendering(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=False)

        logger = haystack_logging.getLogger(__name__)
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        assert "test.test_logging" in output

    @pytest.mark.parametrize(
        "method, expected_level",
        [
            ("debug", "debug"),
            ("info", "info"),
            ("warning", "warning"),
            ("error", "error"),
            ("fatal", "critical"),
            ("exception", "error"),
            ("critical", "critical"),
        ],
    )
    def test_various_levels(self, capfd: LogCaptureFixture, method: str, expected_level: str) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger(__name__)

        logger.setLevel(logging.DEBUG)

        getattr(logger, method)("Hello, structured {key}!", key="logging", key1="value1", key2="value2")

        output = capfd.readouterr().err
        parsed_output = json.loads(output)  # should not raise an error

        assert parsed_output == {
            "event": "Hello, structured logging!",
            "key": "logging",
            "key1": "value1",
            "key2": "value2",
            "level": expected_level,
            "timestamp": ANY,
            "lineno": ANY,
            "module": "test.test_logging",
        }

    def test_log(self, capfd: LogCaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        logger.log(logging.DEBUG, "Hello, structured '{key}'!", key="logging", key1="value1", key2="value2")

        output = capfd.readouterr().err
        parsed_output = json.loads(output)

        assert parsed_output == {
            "event": "Hello, structured 'logging'!",
            "key": "logging",
            "key1": "value1",
            "key2": "value2",
            "level": "debug",
            "timestamp": ANY,
            "lineno": ANY,
            "module": "test.test_logging",
        }

    def test_log_json_content(self, capfd: LogCaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        logger.log(logging.DEBUG, 'Hello, structured: {"key": "value"}', key="logging", key1="value1", key2="value2")

        output = capfd.readouterr().err
        parsed_output = json.loads(output)

        assert parsed_output == {
            "event": 'Hello, structured: {"key": "value"}',
            "key": "logging",
            "key1": "value1",
            "key2": "value2",
            "level": "debug",
            "timestamp": ANY,
            "lineno": ANY,
            "module": "test.test_logging",
        }

    def test_log_with_string_cast(self, capfd: LogCaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        logger.log(logging.DEBUG, "Hello, structured '{key}'!", key=LogCaptureFixture, key1="value1", key2="value2")

        output = capfd.readouterr().err
        parsed_output = json.loads(output)

        assert parsed_output == {
            "event": "Hello, structured '<class '_pytest.logging.LogCaptureFixture'>'!",
            "key": "<class '_pytest.logging.LogCaptureFixture'>",
            "key1": "value1",
            "key2": "value2",
            "level": "debug",
            "timestamp": ANY,
            "lineno": ANY,
            "module": "test.test_logging",
        }

    @pytest.mark.parametrize(
        "method, expected_level",
        [
            ("debug", "debug"),
            ("info", "info"),
            ("warning", "warning"),
            ("error", "error"),
            ("fatal", "critical"),
            ("exception", "exception"),
            ("critical", "critical"),
        ],
    )
    def test_haystack_logger_with_positional_args(self, method: str, expected_level: str) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        with pytest.raises(TypeError):
            getattr(logger, method)("Hello, structured logging %s!", "logging")

    @pytest.mark.parametrize(
        "method, expected_level",
        [
            ("debug", "debug"),
            ("info", "info"),
            ("warning", "warning"),
            ("error", "error"),
            ("fatal", "critical"),
            ("exception", "exception"),
            ("critical", "critical"),
        ],
    )
    def test_haystack_logger_with_old_interpolation(self, method: str, expected_level: str) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # does not raise - hence we need to check this separately
        getattr(logger, method)("Hello, structured logging %s!", key="logging")

    def test_that_haystack_logger_is_used(self) -> None:
        """Forces the usage of the Haystack logger instead of the standard library logger."""
        allowed_list = [Path("haystack") / "logging.py"]
        for root, dirs, files in os.walk("haystack"):
            for file in files:
                path = Path(root) / file

                if not path.suffix.endswith(".py"):
                    continue

                if path in allowed_list:
                    continue

                content = path.read_text(encoding="utf-8")

                # that looks like somebody is using our standard logger
                if " logging.getLogger" in content:
                    haystack_logger_in_content = " haystack import logging" in content or ", logging" in content
                    assert haystack_logger_in_content, (
                        f"{path} doesn't use the Haystack logger. Please use the Haystack logger instead of the "
                        f"standard library logger and add plenty of keyword arguments."
                    )
