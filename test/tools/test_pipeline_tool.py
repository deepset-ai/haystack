# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import ANY

import pytest

from haystack import AsyncPipeline, Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder
from haystack.components.embedders.openai_text_embedder import OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.rankers.sentence_transformers_similarity import SentenceTransformersSimilarityRanker
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import PipelineTool


@pytest.fixture
def sample_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=InMemoryDocumentStore()))
    pipeline.add_component("ranker", SentenceTransformersSimilarityRanker(model="fake-model"))
    pipeline.connect("bm25_retriever", "ranker")
    return pipeline


@pytest.fixture
def sample_async_pipeline():
    pipeline = AsyncPipeline()
    pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=InMemoryDocumentStore()))
    pipeline.add_component("ranker", SentenceTransformersSimilarityRanker(model="fake-model"))
    pipeline.connect("bm25_retriever", "ranker")
    return pipeline


@pytest.fixture
def sample_pipeline_dict():
    return {
        "metadata": {},
        "max_runs_per_component": 100,
        "components": {
            "bm25_retriever": {
                "type": "haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever",
                "init_parameters": {
                    "document_store": {
                        "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                        "init_parameters": {
                            "bm25_tokenization_regex": "(?u)\\b\\w\\w+\\b",
                            "bm25_algorithm": "BM25L",
                            "bm25_parameters": {},
                            "embedding_similarity_function": "dot_product",
                            "index": ANY,
                            "return_embedding": True,
                        },
                    },
                    "filters": None,
                    "top_k": 10,
                    "scale_score": False,
                    "filter_policy": "replace",
                },
            },
            "ranker": {
                "type": "haystack.components.rankers.sentence_transformers_similarity."
                "SentenceTransformersSimilarityRanker",
                "init_parameters": {
                    "device": {"type": "single", "device": ANY},
                    "model": "fake-model",
                    "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                    "top_k": 10,
                    "query_prefix": "",
                    "document_prefix": "",
                    "meta_fields_to_embed": [],
                    "embedding_separator": "\n",
                    "scale_score": True,
                    "score_threshold": None,
                    "trust_remote_code": False,
                    "model_kwargs": None,
                    "tokenizer_kwargs": None,
                    "config_kwargs": None,
                    "backend": "torch",
                    "batch_size": 16,
                },
            },
        },
        "connections": [{"sender": "bm25_retriever.documents", "receiver": "ranker.documents"}],
        "connection_type_validation": True,
    }


class TestPipelineTool:
    def test_init_invalid_pipeline(self):
        with pytest.raises(
            ValueError, match="The 'pipeline' parameter must be an instance of Pipeline or AsyncPipeline."
        ):
            PipelineTool(pipeline="invalid_pipeline", name="test_tool", description="A test tool")

    def test_to_dict(self, sample_pipeline, sample_pipeline_dict):
        tool = PipelineTool(
            pipeline=sample_pipeline,
            input_mapping={"query": ["bm25_retriever.query"]},
            output_mapping={"ranker.documents": "documents"},
            name="test_tool",
            description="A test tool",
        )

        tool_dict = tool.to_dict()
        assert tool_dict == {
            "type": "haystack.tools.pipeline_tool.PipelineTool",
            "data": {
                "pipeline": sample_pipeline_dict,
                "name": "test_tool",
                "input_mapping": {"query": ["bm25_retriever.query"]},
                "output_mapping": {"ranker.documents": "documents"},
                "description": "A test tool",
                "parameters": None,
                "inputs_from_state": None,
                "outputs_to_state": None,
                "is_pipeline_async": False,
                "outputs_to_string": None,
            },
        }

    def test_to_dict_async_pipeline(self, sample_async_pipeline, sample_pipeline_dict):
        tool = PipelineTool(
            pipeline=sample_async_pipeline,
            input_mapping={"query": ["bm25_retriever.query"]},
            output_mapping={"ranker.documents": "documents"},
            name="test_tool",
            description="A test tool",
        )

        tool_dict = tool.to_dict()
        assert tool_dict == {
            "type": "haystack.tools.pipeline_tool.PipelineTool",
            "data": {
                "pipeline": sample_pipeline_dict,
                "name": "test_tool",
                "input_mapping": {"query": ["bm25_retriever.query"]},
                "output_mapping": {"ranker.documents": "documents"},
                "description": "A test tool",
                "parameters": None,
                "inputs_from_state": None,
                "outputs_to_state": None,
                "is_pipeline_async": True,
                "outputs_to_string": None,
            },
        }

    def test_from_dict(self, sample_pipeline):
        tool = PipelineTool(
            pipeline=sample_pipeline,
            input_mapping={"query": ["bm25_retriever.query"]},
            output_mapping={"ranker.documents": "documents"},
            name="test_tool",
            description="A test tool",
        )

        tool_dict = tool.to_dict()
        recreated_tool = PipelineTool.from_dict(tool_dict)

        assert tool.name == recreated_tool.name
        assert tool.description == recreated_tool.description
        assert tool._input_mapping == recreated_tool._input_mapping
        assert tool._output_mapping == recreated_tool._output_mapping
        assert tool.parameters == recreated_tool.parameters
        assert isinstance(recreated_tool._pipeline, Pipeline)

    def test_from_dict_async_pipeline(self, sample_async_pipeline):
        tool = PipelineTool(
            pipeline=sample_async_pipeline,
            input_mapping={"query": ["bm25_retriever.query"]},
            output_mapping={"ranker.documents": "documents"},
            name="test_tool",
            description="A test tool",
        )

        tool_dict = tool.to_dict()
        recreated_tool = PipelineTool.from_dict(tool_dict)

        assert tool.name == recreated_tool.name
        assert tool.description == recreated_tool.description
        assert tool._input_mapping == recreated_tool._input_mapping
        assert tool._output_mapping == recreated_tool._output_mapping
        assert tool.parameters == recreated_tool.parameters
        assert isinstance(recreated_tool._pipeline, AsyncPipeline)

    def test_auto_generated_tool_params(self, sample_pipeline):
        tool = PipelineTool(
            pipeline=sample_pipeline,
            input_mapping={"query": ["bm25_retriever.query", "ranker.query"]},
            output_mapping={"ranker.documents": "documents"},
            name="test_tool",
            description="A test tool",
        )

        assert tool.parameters == {
            "description": "A component that combines: 'bm25_retriever': Run the InMemoryBM25Retriever on the "
            "given input data., 'ranker': Returns a list of documents ranked by their similarity "
            "to the given query.",
            "properties": {
                "query": {
                    "description": "Provided to the 'bm25_retriever' component as: 'The query string for the Retriever."
                    "', and Provided to the 'ranker' component as: 'The input query to compare the "
                    "documents to.'.",
                    "type": "string",
                }
            },
            "required": ["query"],
            "type": "object",
        }

    def test_auto_generated_tool_params_no_mappings(self, sample_pipeline):
        tool = PipelineTool(pipeline=sample_pipeline, name="test_tool", description="A test tool")
        assert tool.parameters == {
            "description": "A component that combines: 'bm25_retriever': Run the InMemoryBM25Retriever on the given "
            "input data., 'ranker': Returns a list of documents ranked by their similarity to the "
            "given query.",
            "properties": {
                "query": {
                    "description": "Provided to the 'bm25_retriever' component as: 'The query string for the "
                    "Retriever.', and Provided to the 'ranker' component as: 'The input query to "
                    "compare the documents to.'.",
                    "type": "string",
                },
                "filters": {
                    "anyOf": [{"additionalProperties": True, "type": "object"}, {"type": "null"}],
                    "description": "Provided to the 'bm25_retriever' component as: 'A dictionary with filters to "
                    "narrow down the search space when retrieving documents.'.",
                },
                "top_k": {
                    "description": "Provided to the 'bm25_retriever' component as: 'The maximum number of documents "
                    "to return.', and Provided to the 'ranker' component as: 'The maximum number "
                    "of documents to return.'.",
                    "type": "integer",
                },
                "scale_score": {
                    "description": "Provided to the 'bm25_retriever' component as: 'When `True`, scales the score "
                    "of retrieved documents to a range of 0 to 1, where 1 means extremely relevant."
                    "\nWhen `False`, uses raw similarity scores.', and Provided to the 'ranker' "
                    "component as: 'If `True`, scales the raw logit predictions using a Sigmoid "
                    "activation function.\nIf `False`, disables scaling of the raw logit predictions."
                    "\nIf set, overrides the value set at initialization.'.",
                    "type": "boolean",
                },
                "score_threshold": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "description": "Provided to the 'ranker' component as: 'Use it to return documents only with "
                    "a score above this threshold.\nIf set, overrides the value set at initialization.'"
                    ".",
                },
            },
            "required": ["query"],
            "type": "object",
        }

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_live_pipeline_tool(self):
        # Initialize a document store and add some documents
        document_store = InMemoryDocumentStore()
        document_embedder = OpenAIDocumentEmbedder()
        documents = [
            Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
            Document(
                content="He is best known for his contributions to the design of the modern alternating current (AC) "
                "electricity supply system."
            ),
        ]
        docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
        document_store.write_documents(docs_with_embeddings)

        # Build a simple retrieval pipeline
        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component("embedder", OpenAITextEmbedder())
        retrieval_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

        retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

        # Wrap the pipeline as a tool
        retriever_tool = PipelineTool(
            pipeline=retrieval_pipeline,
            input_mapping={"query": ["embedder.text"]},
            output_mapping={"retriever.documents": "documents"},
            name="document_retriever",
            description="This tool retrieves documents relevant to Nikola Tesla from the document store",
        )

        # Create an Agent with the tool
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
            system_prompt="For any questions about Nikola Tesla, always use the document_retriever.",
            tools=[retriever_tool],
        )

        # Let the Agent handle a query
        result = agent.run([ChatMessage.from_user("Who was Nikola Tesla?")])

        assert len(result["messages"]) == 5  # System msg, User msg, Agent msg, Tool call result, Agent mgs
        assert "nikola" in result["messages"][-1].text.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    async def test_live_async_pipeline_tool(self):
        # Initialize a document store and add some documents
        document_store = InMemoryDocumentStore()
        document_embedder = OpenAIDocumentEmbedder()
        documents = [
            Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
            Document(
                content="He is best known for his contributions to the design of the modern alternating current (AC) "
                "electricity supply system."
            ),
        ]
        docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
        document_store.write_documents(docs_with_embeddings)

        # Build a simple retrieval pipeline
        retrieval_pipeline = AsyncPipeline()
        retrieval_pipeline.add_component("embedder", OpenAITextEmbedder())
        retrieval_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

        retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

        # Wrap the pipeline as a tool
        retriever_tool = PipelineTool(
            pipeline=retrieval_pipeline,
            input_mapping={"query": ["embedder.text"]},
            output_mapping={"retriever.documents": "documents"},
            name="document_retriever",
            description="For any questions about Nikola Tesla, always use this tool",
        )

        # Create an Agent with the tool
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
            system_prompt="For any questions about Nikola Tesla, always use the document_retriever.",
            tools=[retriever_tool],
        )

        # Let the Agent handle a query
        result = await agent.run_async([ChatMessage.from_user("Who was Nikola Tesla?")])

        assert len(result["messages"]) == 5  # System msg, User msg, Agent msg, Tool call result, Agent mgs
        assert "nikola" in result["messages"][-1].text.lower()
