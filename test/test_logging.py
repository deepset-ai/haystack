# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import logging
import os
import sys
from collections.abc import Callable, Generator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY

import pytest
import structlog
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import haystack.utils.jupyter
from haystack import logging as haystack_logging
from test.tracing.utils import SpyingTracer


@pytest.fixture(autouse=True)
def reset_logging_config() -> None:
    # `configure_logging` attaches its handler to Haystack's own namespaces (and may flip `propagate`), so we snapshot
    # and restore the root logger plus those namespaces to keep tests isolated.
    names = ["haystack", "haystack_integrations", "haystack_experimental"]
    root_handlers = logging.root.handlers.copy()
    snapshots = {name: _snapshot_logger(name) for name in names}
    yield
    logging.root.handlers = root_handlers
    for name, (handlers, propagate, level) in snapshots.items():
        logger = logging.getLogger(name)
        logger.handlers = handlers
        logger.propagate = propagate
        logger.setLevel(level)


def _snapshot_logger(name: str) -> tuple[list[logging.Handler], bool, int]:
    logger = logging.getLogger(name)
    return logger.handlers.copy(), logger.propagate, logger.level


@pytest.fixture()
def restore_named_loggers() -> Generator[Callable[[str], logging.Logger], None, None]:
    """Snapshot a named logger's handlers/propagate/level and restore them after the test."""
    snapshots: dict[str, tuple[list[logging.Handler], bool, int]] = {}

    def _snapshot(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        snapshots[name] = (logger.handlers.copy(), logger.propagate, logger.level)
        return logger

    yield _snapshot

    for name, (handlers, propagate, level) in snapshots.items():
        logger = logging.getLogger(name)
        logger.handlers = handlers
        logger.propagate = propagate
        logger.setLevel(level)


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

        logger = logging.getLogger("haystack.test_logging")
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # the pytest fixture caplog only captures logs being rendered from the stdlib logging module
        assert caplog.messages == ["Hello, structured logging!"]

        # Nothing should be captured by capfd since structlog is not configured
        assert capfd.readouterr().err == ""


class TestStructuredLoggingConsoleRendering:
    def test_log_filtering_when_using_debug(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=False)

        logger = logging.getLogger("haystack.test_logging")
        logger.setLevel(logging.INFO)
        logger.debug("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err
        assert output == ""

    def test_log_filtering_when_using_debug_and_log_level_is_debug(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=False)

        logger = logging.getLogger("haystack.test_logging")
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

        logger = logging.getLogger("haystack.test_logging")
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

        logger = logging.getLogger("haystack.test_logging")
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

        logger = logging.getLogger("haystack.test_logging")
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

        logger = logging.getLogger("haystack.test_logging")
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert "Hello, structured logging!" in output
        assert "{" not in output, "Seems JSON rendering is enabled when it should not be"

    def test_console_rendered_structured_log(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging()

        logger = logging.getLogger("haystack.test_logging")
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

        logger = logging.getLogger("haystack.test_logging")

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

        logger = logging.getLogger("haystack.test_logging")
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # Use `capfd` to capture the output of the final structlog rendering result
        output = capfd.readouterr().err

        assert set_context_var_key in output


class TestStructuredLoggingJSONRendering:
    def test_logging_as_json_if_not_atty(self, capfd: CaptureFixture, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
        haystack_logging.configure_logging()

        logger = logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_logging_as_json(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_logging_as_json_enabling_via_env(self, capfd: CaptureFixture, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_USE_JSON", "true")
        haystack_logging.configure_logging()

        logger = logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_logging_of_contextvars(
        self, capfd: CaptureFixture, monkeypatch: MonkeyPatch, set_context_var_key: str
    ) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_USE_JSON", "true")
        haystack_logging.configure_logging()

        logger = logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_logging_exceptions_json(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger("haystack.test_logging")

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
            "module": "haystack.test_logging",
            "exception": [
                {
                    "exc_notes": [],
                    "exc_type": "ValueError",
                    "exc_value": "This is an error",
                    "exceptions": [],
                    "syntax_error": None,
                    "is_cause": False,
                    "is_group": False,
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
            logger = logging.getLogger("haystack.test_logging")
            logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        assert "trace_id" not in output

    def test_trace_log_correlation_python_logs(self, spying_tracer: SpyingTracer, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        with spying_tracer.trace("test-operation") as span:
            logger = logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_trace_log_correlation_no_span(self, spying_tracer: SpyingTracer, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger("haystack.test_logging")

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
            "module": "haystack.test_logging",
        }

    def test_trace_log_correlation_no_tracer(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = logging.getLogger("haystack.test_logging")

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
            "module": "haystack.test_logging",
        }


class TestCompositeLogger:
    def test_correct_stack_level_with_stdlib_rendering(
        self, monkeypatch: MonkeyPatch, capfd: CaptureFixture, caplog: LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("HAYSTACK_LOGGING_IGNORE_STRUCTLOG", "true")
        haystack_logging.configure_logging()

        logger = logging.getLogger("haystack.test_logging")
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        # the pytest fixture caplog only captures logs being rendered from the stdlib logging module
        assert caplog.messages == ["Hello, structured logging!"]
        assert caplog.records[0].name == "haystack.test_logging"

        # Nothing should be captured by capfd since structlog is not configured
        assert capfd.readouterr().err == ""

    def test_correct_stack_level_with_consoler_rendering(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=False)

        logger = haystack_logging.getLogger("haystack.test_logging")
        logger.warning("Hello, structured logging!", extra={"key1": "value1", "key2": "value2"})

        output = capfd.readouterr().err
        assert "haystack.test_logging" in output

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

        logger = haystack_logging.getLogger("haystack.test_logging")

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
            "module": "haystack.test_logging",
        }

    def test_log(self, capfd: LogCaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_log_json_content(self, capfd: LogCaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
        }

    def test_log_with_string_cast(self, capfd: LogCaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logger = haystack_logging.getLogger("haystack.test_logging")
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
            "module": "haystack.test_logging",
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

        logger = haystack_logging.getLogger("haystack.test_logging")
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

        logger = haystack_logging.getLogger("haystack.test_logging")
        logger.setLevel(logging.DEBUG)

        # does not raise - hence we need to check this separately
        getattr(logger, method)("Hello, structured logging %s!", key="logging")

    def test_that_haystack_logger_is_used(self) -> None:
        """Forces the usage of the Haystack logger instead of the standard library logger."""
        allowed_list = [Path("haystack") / "logging.py"]
        for root, _, files in os.walk("haystack"):
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


class TestLoggingScope:
    """
    Haystack is a library that usually runs next to other services in the same process (e.g. a web server, or another
    app's logging setup). These tests pin down that `configure_logging` only touches Haystack's own loggers and does
    not reformat or hijack the logs of everything else in the process.
    """

    def test_handler_is_attached_to_haystack_namespaces_and_not_root(self) -> None:
        haystack_logging.configure_logging(use_json=True)

        # Haystack's own namespace and the ones used by integration and experimental packages get the handler.
        for name in ["haystack", "haystack_integrations", "haystack_experimental"]:
            logger = logging.getLogger(name)
            assert any(getattr(h, "name", None) == "HaystackLoggingHandler" for h in logger.handlers)

        # The root logger - shared by every other library/service in the process - is left untouched.
        assert not any(getattr(h, "name", None) == "HaystackLoggingHandler" for h in logging.root.handlers)

    def test_integrations_logs_are_formatted_by_haystack(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True)

        logging.getLogger("haystack_integrations.components.demo").warning("a log line from an integration")

        output = capfd.readouterr().err
        assert json.loads(output)["event"] == "a log line from an integration"

    def test_other_services_logs_are_not_reformatted_by_haystack(
        self, capfd: CaptureFixture, restore_named_loggers: Callable[[str], logging.Logger]
    ) -> None:
        # Stand-in for another service that configured its own plain-text logging (e.g. uvicorn) before Haystack runs.
        other_service = restore_named_loggers("some_other_service")
        other_service.handlers = [logging.StreamHandler()]
        other_service.setLevel(logging.INFO)
        other_service.propagate = False

        haystack_logging.configure_logging(use_json=True)

        other_service.info("a log line from another service")
        logging.getLogger("haystack.test_logging").warning("a log line from haystack")

        lines = [line for line in capfd.readouterr().err.splitlines() if line.strip()]

        # Haystack formats its own record as JSON ...
        haystack_json = [json.loads(line) for line in lines if line.startswith("{") and "from haystack" in line]
        assert any(record["event"] == "a log line from haystack" for record in haystack_json)

        # ... but the other service's record stays exactly as that service rendered it - plain text, not Haystack JSON.
        plain_lines = [line for line in lines if "from another service" in line]
        assert plain_lines
        for line in plain_lines:
            with pytest.raises(json.JSONDecodeError):
                json.loads(line)

    def test_legacy_root_behavior_still_available_via_empty_logger_name(self, capfd: CaptureFixture) -> None:
        # Opt back into the old behavior of formatting *every* record in the process.
        haystack_logging.configure_logging(use_json=True, logger_name="")

        assert any(getattr(h, "name", None) == "HaystackLoggingHandler" for h in logging.root.handlers)

        # A non-Haystack logger is now formatted by Haystack because the handler sits on the root logger.
        logging.getLogger("some_other_service").warning("formatted by haystack")
        output = capfd.readouterr().err
        assert json.loads(output)["event"] == "formatted by haystack"

    def test_propagate_is_true_by_default(self, caplog: LogCaptureFixture) -> None:
        # The default keeps records flowing to ancestor loggers, so tooling that captures via the root logger (such as
        # pytest's `caplog`) keeps working.
        haystack_logging.configure_logging(use_json=True)
        assert logging.getLogger("haystack").propagate is True

        with caplog.at_level(logging.WARNING, logger="haystack.test_logging"):
            logging.getLogger("haystack.test_logging").warning("captured via propagation")
        assert "captured via propagation" in caplog.text

    def test_propagate_false_stops_records_from_reaching_root(self, capfd: CaptureFixture) -> None:
        haystack_logging.configure_logging(use_json=True, propagate=False)
        assert logging.getLogger("haystack").propagate is False

        # A plain handler on the root logger should NOT see Haystack's records when propagation is disabled.
        root_handler = logging.StreamHandler()
        root_handler.setFormatter(logging.Formatter("ROOT | %(message)s"))
        logging.root.addHandler(root_handler)
        try:
            logging.getLogger("haystack.test_logging").warning("haystack owns this line")
        finally:
            logging.root.removeHandler(root_handler)

        output = capfd.readouterr().err
        # Formatted once by Haystack (JSON), never by the root handler.
        assert "ROOT |" not in output
        assert json.loads(output)["event"] == "haystack owns this line"


class TestDynamicLogLevel:
    """
    `configure_logging` runs at import time (in `haystack/__init__.py`), long before an application sets its desired
    log level. These tests pin down that a log level set *after* `configure_logging` is still respected, instead of
    being frozen to whatever the root level happened to be at import time.
    """

    def test_structlog_native_logger_respects_level_lowered_after_configure(self, capfd: CaptureFixture) -> None:
        # Mirror the real ordering: configure while the level is high ...
        logging.getLogger("haystack").setLevel(logging.WARNING)
        haystack_logging.configure_logging(use_json=True)

        # ... then have the application lower the level afterwards.
        logging.getLogger("haystack").setLevel(logging.DEBUG)

        structlog.get_logger("haystack.native_dynamic_level").debug("debug emitted after lowering the level")

        assert "debug emitted after lowering the level" in capfd.readouterr().err

    def test_structlog_native_logger_still_filters_below_level(self, capfd: CaptureFixture) -> None:
        logging.getLogger("haystack").setLevel(logging.INFO)
        haystack_logging.configure_logging(use_json=True)

        structlog.get_logger("haystack.native_filtered_level").debug("debug below the configured level")

        assert "debug below the configured level" not in capfd.readouterr().err
