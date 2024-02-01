# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

from haystack.core.pipeline import Pipeline
from haystack.testing.sample_components import AddFixedValue, Double, Parity

logging.basicConfig(level=logging.DEBUG)


def test_pipeline():
    pipeline = Pipeline()
    add_one = AddFixedValue(add=1)
    parity = Parity()
    add_ten = AddFixedValue(add=10)
    double = Double()
    add_three = AddFixedValue(add=3)
    pipeline.add_component("add_one", add_one)
    pipeline.add_component("parity", parity)
    pipeline.add_component("add_ten", add_ten)
    pipeline.add_component("double", double)
    pipeline.add_component("add_three", add_three)

    pipeline.connect(add_one.outputs.result, parity.inputs.value)
    pipeline.connect(parity.outputs.even, add_ten.inputs.value)
    pipeline.connect(parity.outputs.odd, double.inputs.value)
    pipeline.connect(add_ten.outputs.result, add_three.inputs.value)

    results = pipeline.run({"add_one": {"value": 1}})
    assert results == {"add_three": {"result": 15}}

    results = pipeline.run({"add_one": {"value": 2}})
    assert results == {"double": {"value": 6}}
