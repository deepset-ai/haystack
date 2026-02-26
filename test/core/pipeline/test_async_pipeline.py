# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from haystack import AsyncPipeline, Document, component


def test_async_pipeline_reentrance(waiting_component, spying_tracer):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    run_data = [{"wait_for": 0.001}, {"wait_for": 0.002}]

    async def run_all():
        # Create concurrent tasks for each pipeline run
        tasks = [pp.run_async(data) for data in run_data]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())
    component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run_async"]
    for span in component_spans:
        assert span.tags["haystack.component.visits"] == 1


def test_run_in_sync_context(waiting_component):
    pp = AsyncPipeline()
    pp.add_component("wait", waiting_component())

    result = pp.run({"wait_for": 0.001})

    assert result == {"wait": {"waited_for": 0.001}}


def test_run_in_async_context_raises_runtime_error():
    pp = AsyncPipeline()

    async def call_run():
        pp.run({})

    with pytest.raises(RuntimeError, match="Cannot call run\\(\\) from within an async context"):
        asyncio.run(call_run())


def test_component_with_empty_dict_as_output_appears_in_results():
    """Test that components that return an empty dict as output appear in results as an empty dict"""

    @component
    class Producer:
        def __init__(self, prefix: str):
            self.prefix = prefix

        @component.output_types(value=str | None)
        def run(self, text: str | None) -> dict[str, str | None]:
            return {"value": f"{self.prefix}: {text}"}

        @component.output_types(value=str | None)
        async def run_async(self, text: str | None) -> dict[str, str | None]:
            return {"value": f"{self.prefix}: {text}"}

    @component
    class EmptyProcessor:
        @component.output_types()
        def run(self, sources: list[str]) -> dict:
            # Returns empty dict when sources is empty
            return {}

        @component.output_types()
        async def run_async(self, sources: list[str]) -> dict:
            # Returns empty dict when sources is empty
            return {}

    @component
    class Combiner:
        @component.output_types(combined=str)
        def run(self, input_a: str | None, input_b: str | None) -> dict[str, str]:
            if input_a is None:
                input_a = ""
            if input_b is None:
                input_b = ""
            return {"combined": f"{input_a} | {input_b}"}

        @component.output_types(combined=str)
        async def run_async(self, input_a: str | None, input_b: str | None) -> dict[str, str]:
            if input_a is None:
                input_a = ""
            if input_b is None:
                input_b = ""
            return {"combined": f"{input_a} | {input_b}"}

    pp = AsyncPipeline()
    pp.add_component("producer_a", Producer("A"))
    pp.add_component("producer_b", Producer("B"))
    pp.add_component("empty_processor", EmptyProcessor())
    pp.add_component("combiner", Combiner())

    pp.connect("producer_a.value", "combiner.input_a")
    pp.connect("producer_b.value", "combiner.input_b")

    result = pp.run(
        {"producer_a": {"text": "hello"}, "producer_b": {"text": "world"}, "empty_processor": {"sources": []}},
        include_outputs_from={"producer_a", "empty_processor", "combiner"},
    )

    # Producer A should appear in results because it's in include_outputs_from
    assert "producer_a" in result
    assert result["producer_a"] == {"value": "A: hello"}
    # Producer B should NOT appear since it's not in include_outputs_from
    assert "producer_b" not in result
    # Combiner should appear in results
    assert "combiner" in result
    assert result["combiner"] == {"combined": "A: hello | B: world"}
    # Empty processor should appear in results even though it returns an empty dict
    # because it's in include_outputs_from
    assert "empty_processor" in result
    assert result["empty_processor"] == {}


def test_async_pipeline_ensure_inputs_are_deep_copied():
    """
    Test to ensure that async pipeline deep copies the inputs before passing them to components.

    This is important to prevent unintended side effects when components modify their inputs especially when
    the output from one component is passed to multiple other components.

    Some other notes about how this situation can arise in practice:
    - When a component returns a mutable object (like a Document) and that output is passed to multiple other
      components.
    - This doesn't happen when using output types like strings or integers, because they are not shared by
      reference so we will only commonly see this for objects like our dataclasses.
    """

    @component
    class SimpleComponent:
        @component.output_types(output=Document)
        def run(self, document: Document) -> dict[str, Document]:
            # Creates a new document to avoid modifying in place
            new_document = Document(content=document.content)
            return {"output": new_document}

    @component
    class ModifyingComponent:
        @component.output_types(output=Document)
        def run(self, document: Document) -> dict[str, Document]:
            # Modifies the incoming document inplace
            document.content = "modified"
            return {"output": document}

    pp = AsyncPipeline()
    pp.add_component("first", SimpleComponent())
    pp.add_component("modifier", ModifyingComponent())
    # It's important that the following component has a name lower down the alphabetical order than "modifier",
    # since the pipeline runs components in a first-in-first-out manner based on ordered_component_names which is
    # sorted alphabetically.
    pp.add_component("second", SimpleComponent())

    pp.connect("first.output", "modifier.document")
    pp.connect("first.output", "second.document")

    result = pp.run({"first": {"document": Document(content="original")}})

    assert result["modifier"]["output"].content == "modified"
    # Without deep copying the inputs, the second component would also see the modified document and produce
    # "modified" instead of "original"
    assert result["second"]["output"].content == "original"


def test_async_pipeline_does_not_corrupt_outputs():
    """
    Test that a component's output collected via include_outputs_from is not corrupted when a downstream
    component receives and mutates the same data in-place.
    """

    @component
    class Producer:
        @component.output_types(doc=Document)
        def run(self) -> dict:
            return {"doc": Document(content="original")}

    @component
    class Mutator:
        @component.output_types(doc=Document)
        def run(self, doc: Document) -> dict:
            doc.content = "mutated"
            return {"doc": doc}

    pipe = AsyncPipeline()
    pipe.add_component("producer", Producer())
    pipe.add_component("mutator", Mutator())
    pipe.connect("producer.doc", "mutator.doc")

    result = pipe.run({}, include_outputs_from={"producer"})

    assert result["producer"]["doc"].content == "original"
    assert result["mutator"]["doc"].content == "mutated"
