# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Optional, Union

import pytest
from haystack import Document, SuperComponent, Pipeline, AsyncPipeline, component, super_component
from haystack.core.pipeline.base import component_to_dict, component_from_dict
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import GeneratedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.core.super_component.super_component import InvalidMappingTypeError, InvalidMappingValueError


@pytest.fixture
def mock_openai_generator(monkeypatch):
    """Create a mock OpenAI Generator for testing."""

    def mock_run(self, prompt: str, **kwargs):
        return {"replies": ["This is a test response about capitals."]}

    monkeypatch.setattr(OpenAIGenerator, "run", mock_run)
    return OpenAIGenerator(api_key=Secret.from_token("test-key"))


@pytest.fixture
def documents():
    """Create test documents for the document store."""
    return [
        Document(content="Paris is the capital of France."),
        Document(content="Berlin is the capital of Germany."),
        Document(content="Rome is the capital of Italy."),
    ]


@pytest.fixture
def document_store(documents):
    """Create and populate a test document store."""
    store = InMemoryDocumentStore()
    store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    return store


@pytest.fixture
def rag_pipeline(document_store):
    """Create a simple RAG pipeline."""

    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str, **kwargs):
            return {"replies": ["This is a test response about capitals."]}

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template="Given these documents: {{documents|join(', ',attribute='content')}} Answer: {{query}}",
            required_variables="*",
        ),
    )
    pipeline.add_component("llm", FakeGenerator())
    pipeline.add_component("answer_builder", AnswerBuilder())
    pipeline.add_component("joiner", DocumentJoiner())

    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    pipeline.connect("llm.replies", "answer_builder.replies")
    pipeline.connect("retriever.documents", "joiner.documents")

    return pipeline


@pytest.fixture
def async_rag_pipeline(document_store):
    """Create a simple asyncRAG pipeline."""

    @component
    class FakeGenerator:
        @component.output_types(replies=List[str])
        def run(self, prompt: str, **kwargs):
            return {"replies": ["This is a test response about capitals."]}

    pipeline = AsyncPipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template="Given these documents: {{documents|join(', ',attribute='content')}} Answer: {{query}}",
            required_variables="*",
        ),
    )
    pipeline.add_component("llm", FakeGenerator())
    pipeline.add_component("answer_builder", AnswerBuilder())
    pipeline.add_component("joiner", DocumentJoiner())

    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    pipeline.connect("llm.replies", "answer_builder.replies")
    pipeline.connect("retriever.documents", "joiner.documents")

    return pipeline


@super_component
class CustomSuperComponent:
    def __init__(self, var1: int, var2: str = "test"):
        self.var1 = var1
        self.var2 = var2
        pipeline = Pipeline()
        pipeline.add_component("joiner", DocumentJoiner())
        self.pipeline = pipeline


class TestSuperComponent:
    def test_split_component_path(self):
        path = "router.chat_query"
        components = SuperComponent._split_component_path(path)
        assert components == ("router", "chat_query")

    def test_split_component_path_error(self):
        path = "router"
        with pytest.raises(InvalidMappingValueError):
            SuperComponent._split_component_path(path)

    def test_invalid_input_mapping_type(self, rag_pipeline):
        input_mapping = {"search_query": "not_a_list"}  # Should be a list
        with pytest.raises(InvalidMappingTypeError):
            SuperComponent(pipeline=rag_pipeline, input_mapping=input_mapping)

    def test_invalid_input_mapping_value(self, rag_pipeline):
        input_mapping = {"search_query": ["nonexistent_component.query"]}
        with pytest.raises(InvalidMappingValueError):
            SuperComponent(pipeline=rag_pipeline, input_mapping=input_mapping)

    def test_invalid_output_mapping_type(self, rag_pipeline):
        output_mapping = {"answer_builder.answers": 123}  # Should be a string
        with pytest.raises(InvalidMappingTypeError):
            SuperComponent(pipeline=rag_pipeline, output_mapping=output_mapping)

    def test_invalid_output_mapping_value(self, rag_pipeline):
        output_mapping = {"nonexistent_component.answers": "final_answers"}
        with pytest.raises(InvalidMappingValueError):
            SuperComponent(pipeline=rag_pipeline, output_mapping=output_mapping)

    def test_duplicate_output_names(self, rag_pipeline):
        output_mapping = {
            "answer_builder.answers": "final_answers",
            "llm.replies": "final_answers",  # Different path but same output name
        }
        with pytest.raises(InvalidMappingValueError):
            SuperComponent(pipeline=rag_pipeline, output_mapping=output_mapping)

    def test_explicit_input_mapping(self, rag_pipeline):
        input_mapping = {"search_query": ["retriever.query", "prompt_builder.query", "answer_builder.query"]}
        wrapper = SuperComponent(pipeline=rag_pipeline, input_mapping=input_mapping)
        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert set(input_sockets.keys()) == {"search_query"}
        assert input_sockets["search_query"].type == str

    def test_explicit_output_mapping(self, rag_pipeline):
        output_mapping = {"answer_builder.answers": "final_answers"}
        wrapper = SuperComponent(pipeline=rag_pipeline, output_mapping=output_mapping)
        output_sockets = wrapper.__haystack_output__._sockets_dict
        assert set(output_sockets.keys()) == {"final_answers"}
        assert output_sockets["final_answers"].type == List[GeneratedAnswer]

    def test_auto_input_mapping(self, rag_pipeline):
        wrapper = SuperComponent(pipeline=rag_pipeline)
        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert set(input_sockets.keys()) == {
            "documents",
            "filters",
            "meta",
            "pattern",
            "query",
            "reference_pattern",
            "scale_score",
            "template",
            "template_variables",
            "top_k",
        }

    def test_auto_output_mapping(self, rag_pipeline):
        wrapper = SuperComponent(pipeline=rag_pipeline)
        output_sockets = wrapper.__haystack_output__._sockets_dict
        assert set(output_sockets.keys()) == {"answers", "documents"}

    def test_auto_mapping_sockets(self, rag_pipeline):
        wrapper = SuperComponent(pipeline=rag_pipeline)

        output_sockets = wrapper.__haystack_output__._sockets_dict
        assert set(output_sockets.keys()) == {"answers", "documents"}
        assert output_sockets["answers"].type == List[GeneratedAnswer]

        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert set(input_sockets.keys()) == {
            "documents",
            "filters",
            "meta",
            "pattern",
            "query",
            "reference_pattern",
            "scale_score",
            "template",
            "template_variables",
            "top_k",
        }
        assert input_sockets["query"].type == str

    def test_super_component_run(self, rag_pipeline):
        input_mapping = {"search_query": ["retriever.query", "prompt_builder.query", "answer_builder.query"]}
        output_mapping = {"answer_builder.answers": "final_answers"}
        wrapper = SuperComponent(pipeline=rag_pipeline, input_mapping=input_mapping, output_mapping=output_mapping)
        wrapper.warm_up()
        result = wrapper.run(search_query="What is the capital of France?")
        assert "final_answers" in result
        assert isinstance(result["final_answers"][0], GeneratedAnswer)

    @pytest.mark.asyncio
    async def test_super_component_run_async(self, async_rag_pipeline):
        input_mapping = {"search_query": ["retriever.query", "prompt_builder.query", "answer_builder.query"]}
        output_mapping = {"answer_builder.answers": "final_answers"}
        wrapper = SuperComponent(
            pipeline=async_rag_pipeline, input_mapping=input_mapping, output_mapping=output_mapping
        )
        wrapper.warm_up()
        result = await wrapper.run_async(search_query="What is the capital of France?")
        assert "final_answers" in result
        assert isinstance(result["final_answers"][0], GeneratedAnswer)

    def test_wrapper_serialization(self, document_store):
        """Test serialization and deserialization of pipeline wrapper."""
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))

        wrapper = SuperComponent(
            pipeline=pipeline,
            input_mapping={"query": ["retriever.query"]},
            output_mapping={"retriever.documents": "documents"},
        )

        # Test serialization
        serialized = wrapper.to_dict()
        assert "type" in serialized
        assert "init_parameters" in serialized
        assert "pipeline" in serialized["init_parameters"]

        # Test deserialization
        deserialized = SuperComponent.from_dict(serialized)
        assert isinstance(deserialized, SuperComponent)
        assert deserialized.input_mapping == wrapper.input_mapping
        assert deserialized.output_mapping == wrapper.output_mapping

        deserialized.warm_up()
        result = deserialized.run(query="What is the capital of France?")
        assert "documents" in result
        assert result["documents"][0].content == "Paris is the capital of France."

    def test_subclass_serialization(self, rag_pipeline):
        super_comp = SuperComponent(rag_pipeline)
        serialized = super_comp.to_dict()

        @component
        class CustomSuperComponent(SuperComponent):
            def __init__(self, pipeline, instance_attribute="test"):
                self.instance_attribute = instance_attribute
                super(CustomSuperComponent, self).__init__(pipeline)

            def to_dict(self):
                return default_to_dict(
                    self, instance_attribute=self.instance_attribute, pipeline=self.pipeline.to_dict()
                )

            @classmethod
            def from_dict(cls, data):
                data["init_parameters"]["pipeline"] = Pipeline.from_dict(data["init_parameters"]["pipeline"])
                return default_from_dict(cls, data)

        custom_super_component = CustomSuperComponent(rag_pipeline)
        custom_serialized = custom_super_component.to_dict()

        assert custom_serialized["type"] == "test_super_component.CustomSuperComponent"
        assert custom_super_component._to_super_component_dict() == serialized

    def test_super_component_non_leaf_output(self, rag_pipeline):
        # 'retriever' is not a leaf, but should now be allowed
        output_mapping = {"retriever.documents": "retrieved_docs", "answer_builder.answers": "final_answers"}
        wrapper = SuperComponent(pipeline=rag_pipeline, output_mapping=output_mapping)
        wrapper.warm_up()
        result = wrapper.run(query="What is the capital of France?")
        assert "final_answers" in result  # leaf output
        assert "retrieved_docs" in result  # non-leaf output
        assert isinstance(result["retrieved_docs"][0], Document)

    def test_custom_super_component_to_dict(self, rag_pipeline):
        custom_super_component = CustomSuperComponent(1)
        data = component_to_dict(custom_super_component, "custom_super_component")
        assert data == {
            "type": "test_super_component.CustomSuperComponent",
            "init_parameters": {"var1": 1, "var2": "test"},
        }

    def test_custom_super_component_from_dict(self):
        data = {"type": "test_super_component.CustomSuperComponent", "init_parameters": {"var1": 1, "var2": "test"}}
        custom_super_component = component_from_dict(CustomSuperComponent, data, "custom_super_component")
        assert isinstance(custom_super_component, CustomSuperComponent)
        assert custom_super_component.var1 == 1
        assert custom_super_component.var2 == "test"

    def test_input_types_reconciliation(self):
        """Test that input types are properly reconciled when they are compatible but not identical."""

        @component
        class TypeTestComponent:
            @component.output_types(result_int=int, result_any=Any)
            def run(self, input_int: int, input_any: Any):
                return {"result_int": input_int, "result_any": input_any}

        pipeline = Pipeline()
        pipeline.add_component("test1", TypeTestComponent())
        pipeline.add_component("test2", TypeTestComponent())

        input_mapping = {"number": ["test1.input_int", "test2.input_any"]}
        output_mapping = {"test2.result_int": "result_int"}
        wrapper = SuperComponent(pipeline=pipeline, input_mapping=input_mapping, output_mapping=output_mapping)

        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert "number" in input_sockets
        assert input_sockets["number"].type == int

    def test_union_type_reconciliation(self):
        """Test that Union types are properly reconciled when creating a SuperComponent."""

        @component
        class UnionTypeComponent1:
            @component.output_types(result=Union[int, str])
            def run(self, input: Union[int, str]):
                return {"result": input}

        @component
        class UnionTypeComponent2:
            @component.output_types(result=Union[float, str])
            def run(self, input: Union[float, str]):
                return {"result": input}

        pipeline = Pipeline()
        pipeline.add_component("test1", UnionTypeComponent1())
        pipeline.add_component("test2", UnionTypeComponent2())

        input_mapping = {"data": ["test1.input", "test2.input"]}
        output_mapping = {"test2.result": "result"}
        wrapper = SuperComponent(pipeline=pipeline, input_mapping=input_mapping, output_mapping=output_mapping)

        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert "data" in input_sockets
        assert input_sockets["data"].type == Union[str]

    def test_input_types_with_any(self):
        """Test that Any type is properly handled when reconciling types."""

        @component
        class AnyTypeComponent:
            @component.output_types(result=str)
            def run(self, specific: str, generic: Any):
                return {"result": specific}

        pipeline = Pipeline()
        pipeline.add_component("test", AnyTypeComponent())

        input_mapping = {"text": ["test.specific", "test.generic"]}
        wrapper = SuperComponent(pipeline=pipeline, input_mapping=input_mapping)

        input_sockets = wrapper.__haystack_input__._sockets_dict
        assert "text" in input_sockets
        assert input_sockets["text"].type == str
