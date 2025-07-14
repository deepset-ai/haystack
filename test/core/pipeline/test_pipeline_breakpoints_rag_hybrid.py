# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from haystack import Document
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from test.conftest import load_and_resume_pipeline_state


class TestPipelineBreakpoints:
    """
    This class contains tests for pipelines with breakpoints.
    """

    @pytest.fixture
    def mock_sentence_transformers_doc_embedder(self):
        with patch(
            "haystack.components.embedders.sentence_transformers_document_embedder._SentenceTransformersEmbeddingBackendFactory"  # noqa: E501
        ) as mock_doc_embedder:
            mock_model = MagicMock()
            mock_doc_embedder.return_value = mock_model

            # the mock returns a fixed embedding
            def mock_encode(
                documents, batch_size=None, show_progress_bar=None, normalize_embeddings=None, precision=None, **kwargs
            ):
                import numpy as np

                return [np.ones(384).tolist() for _ in documents]

            mock_model.encode = mock_encode
            embedder = SentenceTransformersDocumentEmbedder(model="mock-model", progress_bar=False)

            # mocked run method to return a fixed embedding
            def mock_run(documents: list[Document]):
                if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
                    raise TypeError(
                        "SentenceTransformersDocumentEmbedder expects a list of Documents as input."
                        "In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder."
                    )

                import numpy as np

                embedding = np.ones(384).tolist()

                # Add the embedding to each document
                for doc in documents:
                    doc.embedding = embedding

                # Return the documents with embeddings, matching the actual implementation
                return {"documents": documents}

            # mocked run
            embedder.run = mock_run

            # initialize the component
            embedder.warm_up()

            return embedder

    @pytest.fixture
    def document_store(self, mock_sentence_transformers_doc_embedder):
        """Create and populate a document store for testing."""
        documents = [
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome."),
        ]

        document_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
        ingestion_pipe = Pipeline()
        ingestion_pipe.add_component(instance=mock_sentence_transformers_doc_embedder, name="doc_embedder")
        ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")
        ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
        ingestion_pipe.run({"doc_embedder": {"documents": documents}})

        return document_store

    @pytest.fixture
    def mock_openai_completion(self):
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            mock_completion = MagicMock()
            mock_completion.model = "gpt-4o-mini"
            mock_completion.choices = [
                MagicMock(finish_reason="stop", index=0, message=MagicMock(content="Mark lives in Berlin."))
            ]
            mock_completion.usage = {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97}

            mock_chat_completion_create.return_value = mock_completion
            yield mock_chat_completion_create

    @pytest.fixture
    def mock_transformers_similarity_ranker(self):
        """
        This mock simulates the behavior of the ranker without loading the actual model.
        """
        with (
            patch(
                "haystack.components.rankers.transformers_similarity.AutoModelForSequenceClassification"
            ) as mock_model_class,
            patch("haystack.components.rankers.transformers_similarity.AutoTokenizer") as mock_tokenizer_class,
        ):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()

            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            ranker = TransformersSimilarityRanker(model="mock-model", top_k=5, scale_score=True, calibration_factor=1.0)

            def mock_run(query, documents, top_k=None, scale_score=None, calibration_factor=None, score_threshold=None):
                # assign random scores
                import random

                ranked_docs = documents.copy()
                for doc in ranked_docs:
                    doc.score = random.random()  # random score between 0 and 1

                # sort reverse order and select top_k if provided
                ranked_docs.sort(key=lambda x: x.score, reverse=True)
                if top_k is not None:
                    ranked_docs = ranked_docs[:top_k]
                else:
                    ranked_docs = ranked_docs[: ranker.top_k]

                # apply score threshold if provided
                if score_threshold is not None:
                    ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

                return {"documents": ranked_docs}

            # replace the run method with our mock
            ranker.run = mock_run

            # warm_up to initialize the component
            ranker.warm_up()

            return ranker

    @pytest.fixture
    def mock_sentence_transformers_text_embedder(self):
        """
        Simulates the behavior of the embedder without loading the actual model
        """
        with patch(
            "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
        ) as mock_text_embedder:  # noqa: E501
            mock_model = MagicMock()
            mock_text_embedder.return_value = mock_model

            # the mock returns a fixed embedding
            def mock_encode(
                texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None, precision=None, **kwargs
            ):
                import numpy as np

                return [np.ones(384).tolist() for _ in texts]

            mock_model.encode = mock_encode
            embedder = SentenceTransformersTextEmbedder(model="mock-model", progress_bar=False)

            # mocked run method to return a fixed embedding
            def mock_run(text):
                if not isinstance(text, str):
                    raise TypeError(
                        "SentenceTransformersTextEmbedder expects a string as input."
                        "In case you want to embed a list of Documents, please use the "
                        "SentenceTransformersDocumentEmbedder."
                    )

                import numpy as np

                embedding = np.ones(384).tolist()
                return {"embedding": embedding}

            # mocked run
            embedder.run = mock_run

            # initialize the component
            embedder.warm_up()

            return embedder

    @pytest.fixture
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})
    def hybrid_rag_pipeline(
        self, document_store, mock_transformers_similarity_ranker, mock_sentence_transformers_text_embedder
    ):
        """Create a hybrid RAG pipeline for testing."""

        prompt_template = """
        Given these documents, answer the question based on the document content only.\nDocuments:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        \nQuestion: {{question}}
        \nAnswer:
        """
        pipeline = Pipeline(connection_type_validation=False)
        pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")

        # Use the mocked embedder instead of creating a new one
        pipeline.add_component(instance=mock_sentence_transformers_text_embedder, name="query_embedder")

        pipeline.add_component(
            instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever"
        )
        pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")

        # Use the mocked ranker instead of the real one
        pipeline.add_component(instance=mock_transformers_similarity_ranker, name="ranker")

        pipeline.add_component(
            instance=PromptBuilder(template=prompt_template, required_variables=["documents", "question"]),
            name="prompt_builder",
        )

        # Use a mocked API key for the OpenAIGenerator
        pipeline.add_component(instance=OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY")), name="llm")
        pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        pipeline.connect("query_embedder", "embedding_retriever.query_embedding")
        pipeline.connect("embedding_retriever", "doc_joiner.documents")
        pipeline.connect("bm25_retriever", "doc_joiner.documents")
        pipeline.connect("doc_joiner", "ranker.documents")
        pipeline.connect("ranker", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        pipeline.connect("llm.replies", "answer_builder.replies")
        pipeline.connect("llm.meta", "answer_builder.meta")
        pipeline.connect("doc_joiner", "answer_builder.documents")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    components = [
        Breakpoint("bm25_retriever", 0),
        Breakpoint("query_embedder", 0),
        Breakpoint("embedding_retriever", 0),
        Breakpoint("doc_joiner", 0),
        Breakpoint("ranker", 0),
        Breakpoint("prompt_builder", 0),
        Breakpoint("llm", 0),
        Breakpoint("answer_builder", 0),
    ]

    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    def test_pipeline_breakpoints_hybrid_rag(
        self, hybrid_rag_pipeline, document_store, output_directory, component, mock_openai_completion
    ):
        """
        Test that a hybrid RAG pipeline can be executed with breakpoints at each component.
        """
        # Test data
        question = "Where does Mark live?"
        data = {
            "query_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "ranker": {"query": question, "top_k": 5},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }

        try:
            _ = hybrid_rag_pipeline.run(data, break_point=component, debug_path=str(output_directory))
        except BreakpointException:
            pass

        result = load_and_resume_pipeline_state(hybrid_rag_pipeline, output_directory, component.component_name, data)
        assert result["answer_builder"]
