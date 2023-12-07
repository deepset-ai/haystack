# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from haystack.core.component import component
from haystack.core.pipeline import Pipeline

import logging

logging.basicConfig(level=logging.DEBUG)


@component
class WithDefault:
    @component.output_types(b=int)
    def run(self, a: int, b: int = 2):
        return {"c": a + b}


def test_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_component("with_defaults", WithDefault())
    pipeline.draw(tmp_path / "default_value.png")

    # Pass all the inputs
    results = pipeline.run({"with_defaults": {"a": 40, "b": 30}})
    assert results == {"with_defaults": {"c": 70}}

    # Rely on default value for 'b'
    results = pipeline.run({"with_defaults": {"a": 40}})
    assert results == {"with_defaults": {"c": 42}}
