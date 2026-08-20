# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
`from_dict` must leave the dictionary it receives untouched.

Deserialization replaces serialized values with live objects, for example a callable path with the callable itself
or a nested component dictionary with the component. Doing that on the caller's dictionary leaves it holding objects
that are no longer serializable and that a second `from_dict` call cannot read.
"""

import inspect
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest

from haystack import Document, Pipeline, SuperComponent
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.joiners import BranchJoiner
from haystack.components.query import QueryExpander
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.routers import MetadataRouter
from haystack.components.writers import DocumentWriter
from haystack.core.component.component import component
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Tool
from haystack.utils.misc import expand_page_range


def _tool() -> Tool:
    return Tool(
        name="page_range",
        description="Expand a page range",
        parameters={"type": "object", "properties": {"page_range": {"type": "string"}}},
        function=expand_page_range,
    )


def _super_component() -> SuperComponent:
    pipeline = Pipeline()
    pipeline.add_component("writer", DocumentWriter(document_store=InMemoryDocumentStore()))
    return SuperComponent(pipeline=pipeline)


# One factory per deserialization style that replaces a serialized value with a live object: a callable, a nested
# component, a tool, a type and an enum.
FACTORIES: dict[str, Callable[[], Any]] = {
    "callable": lambda: OpenAIChatGenerator(streaming_callback=print_streaming_chunk),
    "tools": lambda: OpenAIChatGenerator(tools=[_tool()]),
    "chat_generator": lambda: QueryExpander(chat_generator=OpenAIChatGenerator()),
    "llm_ranker": lambda: LLMRanker(chat_generator=OpenAIChatGenerator()),
    "chat_messages": lambda: ChatPromptBuilder(template=[ChatMessage.from_user("{{ query }}")]),
    "output_type": lambda: OutputAdapter(template="{{ documents[0].content }}", output_type=str),
    "type_": lambda: BranchJoiner(type_=list[Document]),
    "enum": lambda: DocumentWriter(document_store=InMemoryDocumentStore()),
    "filter_policy": lambda: InMemoryBM25Retriever(document_store=InMemoryDocumentStore()),
    "router_output_type": lambda: MetadataRouter(
        rules={"edge": {"field": "meta.year", "operator": "==", "value": 2025}}, output_type=list[Document]
    ),
    "tool": _tool,
    "super_component": _super_component,
}


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=list(FACTORIES))
def test_from_dict_does_not_mutate_input(factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    instance = factory()
    data = instance.to_dict()
    expected = deepcopy(data)

    type(instance).from_dict(data)

    assert data == expected


@pytest.mark.parametrize("factory", FACTORIES.values(), ids=list(FACTORIES))
def test_from_dict_can_be_called_twice_on_the_same_data(factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    instance = factory()
    data = instance.to_dict()

    first = type(instance).from_dict(data)
    second = type(instance).from_dict(data)

    # The data is still serializable, so it can be written back to disk after being deserialized.
    assert json.loads(json.dumps(data)) == data
    assert first.to_dict() == second.to_dict()


def test_registered_components_do_not_mutate_input(monkeypatch):
    """Sweep every component that can be built without arguments, so new components are covered automatically."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    offenders = []
    for component_class in component.registry.values():
        parameters = list(inspect.signature(component_class.__init__).parameters.values())[1:]
        required = [
            parameter
            for parameter in parameters
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        if required:
            continue
        try:
            data = component_class().to_dict()
        except Exception:  # components that need optional dependencies or credentials to be built
            continue
        expected = deepcopy(data)
        try:
            component_class.from_dict(data)
        except Exception:
            continue
        if data != expected:
            offenders.append(component_class.__name__)

    assert offenders == []
