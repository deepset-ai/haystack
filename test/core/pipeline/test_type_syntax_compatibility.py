# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import pytest

from haystack.core.component import component
from haystack.core.pipeline import Pipeline


@component
class NonPep604Producer:
    @component.output_types(value=Optional[str], items=Union[str, list[str]])
    def run(self, text: str) -> dict[str, Union[Optional[str], Union[str, list[str]]]]:
        return {"value": text, "items": ["a", "b", "c"]}


@component
class NonPep604Consumer:
    @component.output_types(result=str)
    def run(self, value: Optional[str], items: Union[str, list[str]]) -> dict[str, str]:
        return {"result": f"{value}: {items}"}


@component
class Pep604Producer:
    @component.output_types(value=str | None, items=str | list[str])
    def run(self, text: str) -> dict[str, str | None | str | list[str]]:
        return {"value": text, "items": ["a", "b", "c"]}


@component
class Pep604Consumer:
    @component.output_types(result=str)
    def run(self, value: str | None, items: str | list[str]) -> dict[str, str]:
        return {"result": f"{value}: {items}"}


class TestTypeSyntaxCompatibility:
    """Tests for type syntax compatibility between non-PEP 604 and PEP 604."""

    @pytest.mark.parametrize(
        "producer, consumer", [(NonPep604Producer(), Pep604Consumer()), (Pep604Producer(), NonPep604Consumer())]
    )
    def test_type_syntax_compatibility(self, producer, consumer):
        pipe = Pipeline()
        pipe.add_component("producer", producer)
        pipe.add_component("consumer", consumer)
        pipe.connect("producer.value", "consumer.value")
        pipe.connect("producer.items", "consumer.items")
        result = pipe.run({"producer": {"text": "hello"}})
        assert result["consumer"]["result"] == "hello: ['a', 'b', 'c']"

    def test_non_pep604_pipeline(self):
        yaml_content = """
components:
  producer:
    init_parameters: {}
    type: test.core.pipeline.test_type_syntax_compatibility.NonPep604Producer
  consumer:
    init_parameters: {}
    type: test.core.pipeline.test_type_syntax_compatibility.NonPep604Consumer
connection_type_validation: true
connections:
- receiver: consumer.value
  sender: producer.value
- receiver: consumer.items
  sender: producer.items
max_runs_per_component: 100
metadata: {}
"""

        pipe = Pipeline.loads(yaml_content)

        result = pipe.run({"producer": {"text": "test"}})
        assert result["consumer"]["result"] == "test: ['a', 'b', 'c']"
