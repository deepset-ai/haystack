# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.testing.sample_components import Greet
from haystack.core.serialization import component_to_dict, component_from_dict


def test_greet_message(caplog):
    caplog.set_level(logging.WARNING)
    component = Greet()
    results = component.run(value=10, message="Hello, that's {value}", log_level="WARNING")
    assert results == {"value": 10}
    assert "Hello, that's 10" in caplog.text
