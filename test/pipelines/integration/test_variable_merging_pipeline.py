# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from pprint import pprint

from canals.pipeline import Pipeline
from test.sample_components import AddFixedValue, Sum

import logging

logging.basicConfig(level=logging.DEBUG)


def test_pipeline(tmp_path):
    add_two = AddFixedValue(add=2)
    make_the_sum = Sum(inputs=["in_1", "in_2", "in_3"])

    pipeline = Pipeline()
    pipeline.add_component("first_addition", add_two)
    pipeline.add_component("second_addition", add_two)
    pipeline.add_component("third_addition", add_two)
    pipeline.add_component("sum", make_the_sum)
    pipeline.add_component("fourth_addition", AddFixedValue(add=1))

    pipeline.connect("first_addition", "second_addition")
    pipeline.connect("first_addition", "sum.in_1")
    pipeline.connect("second_addition", "sum.in_2")
    pipeline.connect("third_addition", "sum.in_3")
    pipeline.connect("sum", "fourth_addition.value")

    pipeline.draw(tmp_path / "variable_merging_pipeline.png")

    results = pipeline.run(
        {
            "first_addition": AddFixedValue().input(value=1),
            "third_addition": AddFixedValue().input(value=1),
        }
    )
    pprint(results)

    assert results == {"fourth_addition": AddFixedValue().output(value=12)}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
