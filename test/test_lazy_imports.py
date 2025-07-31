import pytest

from haystack.lazy_imports import DEFAULT_IMPORT_ERROR_MSG, LazyImport


class TestLazyImport:
    def test_import_error_is_suppressed_and_deferred(self):
        with LazyImport() as lazy_import:
            import a_module

        assert lazy_import._deferred is not None
        exc_value, message = lazy_import._deferred
        assert isinstance(exc_value, ImportError)
        expected_message = (
            "Haystack failed to import the optional dependency 'a_module'. Try 'pip install a_module'. "
            "Original error: No module named 'a_module'"
        )
        assert expected_message in message
