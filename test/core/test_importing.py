# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import sys
from unittest.mock import patch


def test_lazy_import():
    # Save the original state of the module if it exists
    original_imported = sys.modules.get("haystack.components.generators.chat.azure")

    with patch.dict(sys.modules):
        # Remove the module from sys.modules if it was already imported
        if original_imported:
            del sys.modules["haystack.components.generators.chat.azure"]

    from haystack.components.generators.chat import OpenAIChatGenerator  # Import the intended class

    assert "haystack.components.generators.chat.openai" in list(sys.modules.keys())
    assert "haystack.components.generators.chat.azure" not in list(sys.modules.keys())

    # Restore the module if it was previously imported (preserves test isolation)
    if original_imported:
        sys.modules["haystack.components.generators.chat.azure"] = original_imported
