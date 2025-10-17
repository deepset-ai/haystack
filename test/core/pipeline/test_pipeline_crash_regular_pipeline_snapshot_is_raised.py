# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.component import component
from haystack.core.errors import PipelineRuntimeError
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore


def setup_document_store():
    documents = [Document(content="My name is Jean and I live in Paris.", embedding=[0.1, 0.3, 0.6])]
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    return document_store


@component
class InvalidOutputEmbeddingRetriever:
    @component.output_types(documents=list[Document])
    def run(self, query_embedding: list[float]):
        # Return an int instead of the expected dict with 'documents' key
        # This will cause the pipeline to crash when trying to pass it to the next component
        return 42


@component
class MockTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str):
        embedding = np.ones(384).tolist()  # Mock embedding of size 384
        return {"embedding": embedding}


class TestPipelineOutputsRaisedInException:
    def test_hybrid_rag_pipeline_crash_on_embedding_retriever(self):
        document_store = setup_document_store()

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", MockTextEmbedder())
        pipeline.add_component("embedding_retriever", InvalidOutputEmbeddingRetriever())
        pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(document_store))
        pipeline.add_component("document_joiner", DocumentJoiner(join_mode="concatenate"))
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=[ChatMessage.from_user("Context:\n{{ documents[0].content }}\n\nQuestion: {{question}}\n")],
                required_variables=["question", "documents"],
            ),
        )

        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("text_embedder", "embedding_retriever")
        pipeline.connect("embedding_retriever", "document_joiner")
        pipeline.connect("document_joiner.documents", "prompt_builder.documents")

        question = "Where does Mark live?"
        test_data = {
            "text_embedder": {"text": question},
            "bm25_retriever": {"query": question},
            "prompt_builder": {"question": question},
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
                },
            )

        pipeline_snapshot = exc_info.value.pipeline_snapshot
        pipeline_outputs = pipeline_snapshot.pipeline_state.pipeline_outputs
        assert pipeline_outputs is not None, "Pipeline outputs should be captured in the exception"

        # verify that bm25_retriever and text_embedder ran successfully before the crash
        assert "bm25_retriever" in pipeline_outputs, "BM25 retriever output not captured"
        assert "documents" in pipeline_outputs["bm25_retriever"], "BM25 retriever should have produced documents"
        assert "text_embedder" in pipeline_outputs, "Text embedder output not captured"
        assert "embedding" in pipeline_outputs["text_embedder"], "Text embedder should have produced embeddings"

        # components after the crash point are not in the outputs
        assert "document_joiner" not in pipeline_outputs, "Document joiner should not have run due to crash"
        assert "prompt_builder" not in pipeline_outputs, "Prompt builder should not have run due to crash"
