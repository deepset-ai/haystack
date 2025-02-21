# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from types import TracebackType
from typing import Optional, Type

from lazy_imports.try_import import _DeferredImportExceptionContextManager

DEFAULT_IMPORT_ERROR_MSG = "Try 'pip install {}'"


class LazyImport(_DeferredImportExceptionContextManager):
    """
    A context manager that provides controlled handling of import errors.

    It adds the possibility to customize the error messages.

    NOTE: Despite its name, this class does not delay the actual import operation.
    For installed modules: executes the import immediately.
    For uninstalled modules: captures the error and defers it until check() is called.
    """

    def __init__(self, message: str = DEFAULT_IMPORT_ERROR_MSG) -> None:
        super().__init__()
        self.import_error_msg = message

    def __exit__(
        self, exc_type: Optional[Type[Exception]], exc_value: Optional[Exception], traceback: Optional[TracebackType]
    ) -> Optional[bool]:
        """
        Exit the context manager.

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
        if isinstance(exc_value, ImportError):
            message = (
                f"Failed to import '{exc_value.name}'. {self.import_error_msg.format(exc_value.name)}. "
                f"Original error: {exc_value}"
            )
            self._deferred = (exc_value, message)
            return True
        return None
