# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from haystack import Document, Pipeline, component
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.core.errors import BreakpointException
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore


@component
class FakeEmbedder:
    @component.output_types(documents=list[Document], embedding=list[float])
    def run(self, text: str):
        return {"embedding": np.ones(384).tolist()}


@component
class FakeRanker:
    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int = None):
        for i, doc in enumerate(documents):
            doc.score = 1.0 / (i + 1)
        return {"documents": sorted(documents, key=lambda x: x.score, reverse=True)[:top_k]}


@component
class FakeGenerator:
    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(self, prompt: str):
        return {"replies": ["Mark lives in Berlin."], "meta": [{"model": "fake"}]}


class TestPipelineBreakpoints:
    """
    This class contains tests for pipelines with breakpoints.
    """

    @pytest.fixture
    def document_store(self):
        """Create and populate a document store for testing."""
        documents = [
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome."),
        ]
        ds = InMemoryDocumentStore()
        # Add embeddings
        for doc in documents:
            doc.embedding = np.ones(384).tolist()
        ds.write_documents(documents)
        return ds

    @pytest.fixture
    def hybrid_rag_pipeline(self, document_store):
        """Create a hybrid RAG pipeline for testing."""
        prompt_template = "Documents: {% for doc in documents %}{{ doc.content }}{% endfor %} Question: {{question}}"

        pipeline = Pipeline()
        pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store=document_store))
        pipeline.add_component("query_embedder", FakeEmbedder())
        pipeline.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=document_store))
        pipeline.add_component("doc_joiner", DocumentJoiner())
        pipeline.add_component("ranker", FakeRanker())
        pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
        pipeline.add_component("llm", FakeGenerator())
        pipeline.add_component("answer_builder", AnswerBuilder())

        pipeline.connect("query_embedder.embedding", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever", "doc_joiner.documents")
        pipeline.connect("bm25_retriever", "doc_joiner.documents")
        pipeline.connect("doc_joiner", "ranker.documents")
        pipeline.connect("ranker", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        pipeline.connect("llm.replies", "answer_builder.replies")
        pipeline.connect("llm.meta", "answer_builder.meta")
        pipeline.connect("doc_joiner", "answer_builder.documents")

        return pipeline

    BREAKPOINT_COMPONENTS = [
        "bm25_retriever",
        "query_embedder",
        "embedding_retriever",
        "doc_joiner",
        "ranker",
        "prompt_builder",
        "llm",
        "answer_builder",
    ]

    @pytest.mark.parametrize("component", BREAKPOINT_COMPONENTS, ids=BREAKPOINT_COMPONENTS)
    @pytest.mark.integration
    def test_pipeline_breakpoints_hybrid_rag(
        self, hybrid_rag_pipeline, output_directory, component, load_and_resume_pipeline_snapshot
    ):
        """
        Test that a hybrid RAG pipeline can be executed with breakpoints at each component.
        """
        question = "Where does Mark live?"
        data = {
            "query_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "ranker": {"query": question, "top_k": 5},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # Create a Breakpoint on-the-fly using the shared output directory
        break_point = Breakpoint(component_name=component, visit_count=0, snapshot_file_path=str(output_directory))

        try:
            _ = hybrid_rag_pipeline.run(data, break_point=break_point)
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_snapshot(
            pipeline=hybrid_rag_pipeline,
            output_directory=output_directory,
            component_name=break_point.component_name,
            data=data,
        )
        assert "answer_builder" in result
        assert result["answer_builder"]["answers"][0].data == "Mark lives in Berlin."
