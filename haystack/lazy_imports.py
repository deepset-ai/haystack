from typing import Optional, Type
from types import TracebackType
from lazy_imports.try_import import _DeferredImportExceptionContextManager


DEFAULT_IMPORT_ERROR_MSG = (
    "Tried to import '{module}' but failed. Make sure that the package is installed correctly to use this feature. "
    "Actual error: {exception}."
)
DEFAULT_SYNTAX_ERROR_MSG = (
    "Tried to import a package but failed due to a syntax error in {module}. Actual error: {exception}."
)


class LazyImport(_DeferredImportExceptionContextManager):
    """
    Wrapper on top of lazy_import's _DeferredImportExceptionContextManager that adds the possibility to customize the
    error messages.
    """

    def __init__(
        self, import_error_msg: str = DEFAULT_IMPORT_ERROR_MSG, syntax_error_msg: str = DEFAULT_SYNTAX_ERROR_MSG
    ) -> None:
        super().__init__()
        self.import_error_msg = import_error_msg
        self.syntax_error_msg = syntax_error_msg

    def __exit__(
        self, exc_type: Optional[Type[Exception]], exc_value: Optional[Exception], traceback: Optional[TracebackType]
    ) -> Optional[bool]:
        """Exit the context manager.

        Args:
            exc_type:
                Raised exception type. :obj:`None` if nothing is raised.
            exc_value:
                Raised exception object. :obj:`None` if nothing is raised.
            traceback:
                Associated traceback. :obj:`None` if nothing is raised.

        Returns:
            :obj:`None` if nothing is deferred, otherwise :obj:`True`.
            :obj:`True` will suppress any exceptions avoiding them from propagating.

        """
        if isinstance(exc_value, (ImportError, SyntaxError)):
            if isinstance(exc_value, ImportError):
                message = self.import_error_msg.format(module=exc_value.name, exception=exc_value)
            elif isinstance(exc_value, SyntaxError):
                message = self.syntax_error_msg.format(module=exc_value.filename, exception=exc_value)
            else:
                assert False

            self._deferred = (exc_value, message)
            return True
        return None
