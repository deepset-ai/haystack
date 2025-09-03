import os

import pytest

from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import PipelineTool


class PipelineToolTest:
    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    def test_live_pipeline_tool(self):
        # Initialize a document store and add some documents
        document_store = InMemoryDocumentStore()
        document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        documents = [
            Document(content="Nikola Tesla was a Serbian-American inventor and electrical engineer."),
            Document(
                content="He is best known for his contributions to the design of the modern alternating current (AC) "
                "electricity supply system."
            ),
        ]
        document_embedder.warm_up()
        docs_with_embeddings = document_embedder.run(documents=documents)["documents"]
        document_store.write_documents(docs_with_embeddings)

        # Build a simple retrieval pipeline
        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component(
            "embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        )
        retrieval_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))

        retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

        # Wrap the pipeline as a tool
        retriever_tool = PipelineTool(
            pipeline=retrieval_pipeline,
            input_mapping={"query": ["embedder.text"]},
            output_mapping={"retriever.documents": "documents"},
            name="document_retriever",
            description="Retrieve documents relevant to a query from the document store",
        )

        # Create an Agent with the tool
        agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"), tools=[retriever_tool])

        # Let the Agent handle a query
        result = agent.run([ChatMessage.from_user("Who was Nikola Tesla?")])

        assert len(result["messages"]) == 4  # User message, Agent message, Tool call result, Agent message
        assert "nikola" in result["messages"][-1].text.lower()
