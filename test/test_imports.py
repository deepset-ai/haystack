# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch

from haystack.lazy_imports import LazyImport


def test_lazy_importer_avoids_importing_unused_modules():
    # Save the original state of the module if it exists
    original_imported = sys.modules.get("haystack.components.generators.chat.azure")

    with patch.dict(sys.modules):
        # Remove the module from sys.modules if it was already imported
        if original_imported:
            del sys.modules["haystack.components.generators.chat.azure"]

        assert "haystack.components.generators.chat.openai" in sys.modules.keys()
        assert "haystack.components.generators.chat.azure" not in sys.modules.keys()

    # Restore the module if it was previously imported (preserves test isolation)
    if original_imported:
        sys.modules["haystack.components.generators.chat.azure"] = original_imported


def test_import_error_is_suppressed_and_deferred():
    with LazyImport() as lazy_import:
        pass

    assert lazy_import._deferred is not None
    exc_value, message = lazy_import._deferred
    assert isinstance(exc_value, ImportError)
    expected_message = (
        "Haystack failed to import the optional dependency 'a_module'. Try 'pip install a_module'. "
        "Original error: No module named 'a_module'"
    )
    assert expected_message in message
