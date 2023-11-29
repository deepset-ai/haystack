# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from pprint import pprint

from haystack.core.component import component
from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Sum

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
    pprint(results)
    assert results == {"with_defaults": {"c": 70}}

    # Rely on default value for 'b'
    results = pipeline.run({"with_defaults": {"a": 40}})
    pprint(results)
    assert results == {"with_defaults": {"c": 42}}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
