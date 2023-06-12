import logging
import random
import sys
from typing import Generator, List
from unittest.mock import Mock

from structlog.testing import capture_logs
import ddtrace
import freezegun
import pytest
import structlog
from _pytest.monkeypatch import MonkeyPatch
from ddtrace.settings import Config
from structlog.types import EventDict, WrappedLogger

from consumer_indexing.src import _log_unhandled_exception, configure_logging
from consumer_indexing.src.config import LOGGING_LOCALS_MAX_STRING


class PluginLogCapture:
    """
    Plugin structlog processor for capturing log messages in its entries list.
    This is a lightweight version of structlog's LogCapture class with the following differences:
    - It does not raise DropEvent (which causes problems with our stdlib logging setup).
    - It passes the event_dict through to the next processor as is.
    - It is intended to be used as a plugin to be placed directly before the actual renderer (e.g. JSONRenderer) and not as a replacement.
      See `configure_logging`'s `log_capture` parameter.
    """

    entries: List[EventDict]

    def __init__(self) -> None:
        self.entries = []

    def __call__(self, _: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
        self.entries.append(event_dict)
        return event_dict


@pytest.fixture
def log_capture() -> PluginLogCapture:
    return PluginLogCapture()


@pytest.fixture
def configure_logging_for_prod_with_log_capture(log_capture: PluginLogCapture) -> Generator[None, None, None]:
    try:
        configure_logging(dev_logs=False, log_capture=log_capture)
        yield
    finally:
        configure_logging()

@pytest.fixture
def raw_logs() -> Generator[List[EventDict], None, None]:
    with capture_logs() as logs:
        yield logs


class TestUnhandledExceptionLogging:
    def test_exception_handler(self, raw_logs: List[EventDict]) -> None:
        try:
            raise Exception("Test exception")
        except Exception as ex:
            tb = sys.exc_info()[2]
            _log_unhandled_exception(type(ex), ex, tb)

        assert raw_logs[0].get("event") == "Unhandled exception"
        assert len(raw_logs[0]["exc_info"]) == 3
        assert raw_logs[0]["exc_info"][0] == Exception
        assert isinstance(raw_logs[0]["exc_info"][1], Exception)
        assert isinstance(raw_logs[0]["exc_info"][2], type(tb))

    @pytest.mark.usefixtures(configure_logging_for_prod_with_log_capture.__name__)
    def test_exception_handler_with_configured_logs(self, log_capture: PluginLogCapture) -> None:
        try:
            raise Exception("Test exception")
        except Exception as ex:
            tb = sys.exc_info()[2]
            _log_unhandled_exception(type(ex), ex, tb)

        assert log_capture.entries[0].get("event") == "Unhandled exception"
        assert log_capture.entries[0].get("level") == "critical"
        assert isinstance(log_capture.entries[0].get("exception"), list)
        assert len(log_capture.entries[0]["exception"]) == 1
        assert log_capture.entries[0]["exception"][0]["exc_type"] == "Exception"
        assert log_capture.entries[0]["exception"][0]["exc_value"] == "Test exception"
        assert log_capture.entries[0]["exception"][0]["frames"][0]["filename"] == __file__
        assert (
            log_capture.entries[0]["exception"][0]["frames"][0]["name"]
            == self.test_exception_handler_with_configured_logs.__name__
        )

    @pytest.mark.usefixtures(configure_logging_for_prod_with_log_capture.__name__)
    def test_log_exception_structlog(self, log_capture: PluginLogCapture) -> None:
        logger = structlog.get_logger()
        try:
            raise Exception("Test exception")
        except Exception:
            logger.exception("This is an exception")

        assert log_capture.entries[0].get("event") == "This is an exception"
        assert log_capture.entries[0].get("level") == "error"
        assert isinstance(log_capture.entries[0].get("exception"), list)
        assert len(log_capture.entries[0]["exception"]) == 1
        assert log_capture.entries[0]["exception"][0]["exc_type"] == "Exception"
        assert log_capture.entries[0]["exception"][0]["exc_value"] == "Test exception"
        assert log_capture.entries[0]["exception"][0]["frames"][0]["filename"] == __file__
        assert log_capture.entries[0]["exception"][0]["frames"][0]["name"] == self.test_log_exception_structlog.__name__

    @pytest.mark.usefixtures(configure_logging_for_prod_with_log_capture.__name__)
    def test_log_exception_stdlib(self, log_capture: PluginLogCapture) -> None:
        logger = logging.getLogger()
        try:
            raise Exception("Test exception")
        except Exception:
            logger.exception("This is an exception")

        assert log_capture.entries[0].get("event") == "This is an exception"
        assert log_capture.entries[0].get("level") == "error"
        assert isinstance(log_capture.entries[0].get("exception"), list)
        assert len(log_capture.entries[0]["exception"]) == 1
        assert log_capture.entries[0]["exception"][0]["exc_type"] == "Exception"
        assert log_capture.entries[0]["exception"][0]["exc_value"] == "Test exception"
        assert log_capture.entries[0]["exception"][0]["frames"][0]["filename"] == __file__
        assert log_capture.entries[0]["exception"][0]["frames"][0]["name"] == self.test_log_exception_stdlib.__name__

    @pytest.mark.usefixtures(configure_logging_for_prod_with_log_capture.__name__)
    def test_log_exception_stdlib_with_less_than_max_string_length_chars(self, log_capture: PluginLogCapture) -> None:
        """ """
        logger = logging.getLogger()
        message = "".join((random.choice("asdf") for i in range(LOGGING_LOCALS_MAX_STRING - 1)))
        try:
            raise Exception(message)
        except Exception:
            logger.exception(message)

        assert log_capture.entries[0].get("event") == message
        assert log_capture.entries[0].get("level") == "error"

    @pytest.mark.usefixtures(configure_logging_for_prod_with_log_capture.__name__)
    def test_log_exception_stdlib_with_more_than_max_string_length_chars(self, log_capture: PluginLogCapture) -> None:
        """ """
        logger = logging.getLogger()
        message = "".join((random.choice("asdf") for i in range(LOGGING_LOCALS_MAX_STRING + 1)))
        try:
            raise Exception(message)
        except Exception:
            logger.exception(message)

        locals_message = log_capture.entries[0].get("exception")[0].get("frames")[0].get("locals").get("message")  # type: ignore
        assert locals_message.endswith("+1")

    @pytest.mark.usefixtures(configure_logging_for_prod_with_log_capture.__name__)
    def test_log_exception_stdlib_with_exactly_max_string_length_chars(self, log_capture: PluginLogCapture) -> None:
        logger = logging.getLogger()
        message = "".join((random.choice("asdf") for i in range(LOGGING_LOCALS_MAX_STRING)))
        try:
            raise Exception(message)
        except Exception:
            logger.exception(message)

        locals_message = log_capture.entries[0].get("exception")[0].get("frames")[0].get("locals").get("message")  # type: ignore
        assert not locals_message.endswith("+1")

