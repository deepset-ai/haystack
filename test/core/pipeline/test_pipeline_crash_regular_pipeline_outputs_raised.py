# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from haystack import AsyncPipeline, Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.core.component import component
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret


def setup_document_store():
    """Create and populate a document store with test documents."""
    documents = [
        Document(content="My name is Jean and I live in Paris.", embedding=[0.1, 0.3, 0.6]),
        Document(content="My name is Mark and I live in Berlin.", embedding=[0.2, 0.4, 0.7]),
        Document(content="My name is Giorgio and I live in Rome.", embedding=[0.3, 0.5, 0.8]),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_writer.run(documents=documents)

    return document_store


# Create a mock component that returns invalid output (int instead of documents list)
@component
class InvalidOutputEmbeddingRetriever:
    @component.output_types(documents=list[Document])
    def run(self, query_embedding: list[float]):
        # Return an int instead of the expected documents list
        # This will cause the pipeline to crash when trying to pass it to the next component
        return 42


template = [
    ChatMessage.from_system(
        "You are a helpful AI assistant. Answer the following question based on the given context information "
        "only. If the context is empty or just a '\n' answer with None, example: 'None'."
    ),
    ChatMessage.from_user(
        """
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        """
    ),
]


class TestPipelineOutputsRaisedInException:
    @pytest.fixture
    def mock_sentence_transformers_text_embedder(self):
        with patch(
            "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
        ) as mock_text_embedder:
            mock_model = MagicMock()
            mock_text_embedder.return_value = mock_model

            def mock_encode(
                texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None, precision=None, **kwargs
            ):  # noqa E501
                return [np.ones(384).tolist() for _ in texts]

            mock_model.encode = mock_encode
            embedder = SentenceTransformersTextEmbedder(model="mock-model", progress_bar=False)

            def mock_run(text):
                if not isinstance(text, str):
                    raise TypeError(
                        "SentenceTransformersTextEmbedder expects a string as input."
                        "In case you want to embed a list of Documents, please use the "
                        "SentenceTransformersDocumentEmbedder."
                    )

                embedding = np.ones(384).tolist()
                return {"embedding": embedding}

            embedder.run = mock_run
            embedder.warm_up()
            return embedder

    def test_hybrid_rag_pipeline_crash_on_embedding_retriever(
        self, mock_sentence_transformers_text_embedder, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        document_store = setup_document_store()
        text_embedder = mock_sentence_transformers_text_embedder
        invalid_embedding_retriever = InvalidOutputEmbeddingRetriever()
        bm25_retriever = InMemoryBM25Retriever(document_store)
        document_joiner = DocumentJoiner(join_mode="concatenate")

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("embedding_retriever", invalid_embedding_retriever)
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("document_joiner", document_joiner)
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"])
        )
        pipeline.add_component("llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")))
        pipeline.add_component("answer_builder", AnswerBuilder())

        pipeline.connect("text_embedder", "embedding_retriever")
        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("document_joiner.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        pipeline.connect("llm.replies", "answer_builder.replies")

        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        # run pipeline and expect it to crash due to invalid output type
        with pytest.raises(PipelineRuntimeError) as exc_info:
            pipeline.run(
                data=test_data,
                include_outputs_from={
                    "text_embedder",
                    "embedding_retriever",
                    "bm25_retriever",
                    "document_joiner",
                    "prompt_builder",
                    "llm",
                    "answer_builder",
                },
            )

        pipeline_outputs = exc_info.value.pipeline_outputs

        assert pipeline_outputs is not None, "Pipeline outputs should be captured in the exception"

        # verify that bm25_retriever and text_embedder ran successfully before the crash
        assert "bm25_retriever" in pipeline_outputs, "BM25 retriever output not captured"
        assert "documents" in pipeline_outputs["bm25_retriever"], "BM25 retriever should have produced documents"
        assert "text_embedder" in pipeline_outputs, "Text embedder output not captured"
        assert "embedding" in pipeline_outputs["text_embedder"], "Text embedder should have produced embeddings"

        # components after the crash point are not in the outputs
        assert "document_joiner" not in pipeline_outputs, "Document joiner should not have run due to crash"
        assert "prompt_builder" not in pipeline_outputs, "Prompt builder should not have run due to crash"
        assert "llm" not in pipeline_outputs, "LLM should not have run due to crash"
        assert "answer_builder" not in pipeline_outputs, "Answer builder should not have run due to crash"

    @pytest.mark.asyncio
    async def test_async_hybrid_rag_pipeline_crash_on_embedding_retriever(
        self, mock_sentence_transformers_text_embedder, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        document_store = setup_document_store()
        text_embedder = mock_sentence_transformers_text_embedder
        invalid_embedding_retriever = InvalidOutputEmbeddingRetriever()
        bm25_retriever = InMemoryBM25Retriever(document_store)
        document_joiner = DocumentJoiner(join_mode="concatenate")

        pipeline = AsyncPipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("embedding_retriever", invalid_embedding_retriever)
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("document_joiner", document_joiner)
        pipeline.add_component(
            "prompt_builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"])
        )
        pipeline.add_component("llm", OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")))
        pipeline.add_component("answer_builder", AnswerBuilder())

        pipeline.connect("text_embedder", "embedding_retriever")
        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("document_joiner.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        pipeline.connect("llm.replies", "answer_builder.replies")

        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        with pytest.raises(PipelineRuntimeError) as exc_info:
            await pipeline.run_async(
                data=test_data,
                include_outputs_from={
                    "text_embedder",
                    "embedding_retriever",
                    "bm25_retriever",
                    "document_joiner",
                    "prompt_builder",
                    "llm",
                    "answer_builder",
                },
            )

        pipeline_outputs = exc_info.value.pipeline_outputs
        assert pipeline_outputs is not None, "Pipeline outputs should be captured in the exception"

        # verify that bm25_retriever and text_embedder ran successfully before the crash
        assert "bm25_retriever" in pipeline_outputs, "BM25 retriever output not captured"
        assert "documents" in pipeline_outputs["bm25_retriever"], "BM25 retriever should have produced documents"
        assert "text_embedder" in pipeline_outputs, "Text embedder output not captured"
        assert "embedding" in pipeline_outputs["text_embedder"], "Text embedder should have produced embeddings"

        # components after the crash point are not in the outputs
        assert "document_joiner" not in pipeline_outputs, "Document joiner should not have run due to crash"
        assert "prompt_builder" not in pipeline_outputs, "Prompt builder should not have run due to crash"
        assert "llm" not in pipeline_outputs, "LLM should not have run due to crash"
        assert "answer_builder" not in pipeline_outputs, "Answer builder should not have run due to crash"

        # check that a pipeline snapshot file was created in the "debug" directory
        snapshot_files = os.listdir("debug")
        assert any(f.endswith(".json") for f in snapshot_files), "No pipeline snapshot file found in debug directory"
