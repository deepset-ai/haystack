# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from haystack.core.pipeline import Pipeline
from haystack.core.component import component
from haystack.testing.sample_components import StringListJoiner


@component
class InputMangler:
    @component.output_types(mangled_list=List[str])
    def run(self, input_list: List[str]):
        input_list.append("extra_item")
        return {"mangled_list": input_list}


def test_mutable_inputs():
    pipe = Pipeline()
    pipe.add_component("mangler1", InputMangler())
    pipe.add_component("mangler2", InputMangler())
    pipe.add_component("concat1", StringListJoiner())
    pipe.add_component("concat2", StringListJoiner())
    pipe.connect("mangler1", "concat1")
    pipe.connect("mangler2", "concat2")

    mylist = ["foo", "bar"]

    result = pipe.run(data={"mangler1": {"input_list": mylist}, "mangler2": {"input_list": mylist}})
    assert result["concat1"]["output"] == result["concat2"]["output"] == ["foo", "bar", "extra_item"]
