# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Double

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline()
    first_addition = AddFixedValue(add=2)
    second_addition = AddFixedValue()
    double = Double()
    pipeline.add_component("first_addition", first_addition)
    pipeline.add_component("second_addition", second_addition)
    pipeline.add_component("double", double)
    pipeline.connect(first_addition.outputs.result, double.inputs.value)
    pipeline.connect(double.outputs.value, second_addition.inputs.value)

    results = pipeline.run({"first_addition": {"value": 1}})
    assert results == {"second_addition": {"result": 7}}
