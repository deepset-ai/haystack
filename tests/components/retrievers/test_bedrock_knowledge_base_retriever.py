# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Amazon Bedrock Knowledge Base Retriever component."""

from unittest.mock import MagicMock, patch

import pytest

from haystack import Document


@pytest.fixture
def mock_boto3_client():
    with patch("boto3.client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


class TestBedrockKnowledgeBaseRetriever:
    def test_init_defaults(self):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        retriever = BedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        assert retriever.knowledge_base_id == "TEST123456"
        assert retriever.number_of_results == 5
        assert retriever.knowledge_base_type == "MANAGED"

    @patch.dict("os.environ", {"KNOWLEDGE_BASE_ID": "ENV_KB", "AWS_REGION": "eu-west-1"})
    def test_init_from_env(self):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        retriever = BedrockKnowledgeBaseRetriever()
        assert retriever.knowledge_base_id == "ENV_KB"
        assert retriever.region_name == "eu-west-1"

    def test_run_managed(self, mock_boto3_client):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        mock_boto3_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Managed KB handles everything automatically."},
                    "location": {"s3Location": {"uri": "s3://bucket/doc.pdf"}},
                    "score": 0.95,
                },
                {
                    "content": {"text": "No vector store needed."},
                    "location": {"s3Location": {"uri": "s3://bucket/doc2.pdf"}},
                    "score": 0.87,
                },
            ]
        }

        retriever = BedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        result = retriever.run(query="What is managed KB?")

        # Verify correct API call
        mock_boto3_client.retrieve.assert_called_once()
        call_kwargs = mock_boto3_client.retrieve.call_args.kwargs
        assert call_kwargs["knowledgeBaseId"] == "TEST123456"
        assert "managedSearchConfiguration" in call_kwargs["retrievalConfiguration"]

        # Verify documents returned
        assert "documents" in result
        docs = result["documents"]
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].content == "Managed KB handles everything automatically."
        assert docs[0].meta["source"] == "s3://bucket/doc.pdf"
        assert docs[0].score == 0.95

    def test_run_vector(self, mock_boto3_client):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        mock_boto3_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Vector result."},
                    "location": {"s3Location": {"uri": "s3://bucket/v.pdf"}},
                    "score": 0.9,
                },
            ]
        }

        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id="TEST123456",
            knowledge_base_type="VECTOR",
        )
        result = retriever.run(query="test")

        call_kwargs = mock_boto3_client.retrieve.call_args.kwargs
        assert "vectorSearchConfiguration" in call_kwargs["retrievalConfiguration"]

    def test_run_top_k_override(self, mock_boto3_client):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        mock_boto3_client.retrieve.return_value = {"retrievalResults": []}

        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id="TEST123456", number_of_results=5
        )
        retriever.run(query="test", top_k=10)

        call_kwargs = mock_boto3_client.retrieve.call_args.kwargs
        assert call_kwargs["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == 10

    def test_run_empty_results(self, mock_boto3_client):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        mock_boto3_client.retrieve.return_value = {"retrievalResults": []}

        retriever = BedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        result = retriever.run(query="no match")

        assert result["documents"] == []

    def test_run_error_handling(self, mock_boto3_client):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        mock_boto3_client.retrieve.side_effect = Exception("Service unavailable")

        retriever = BedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        result = retriever.run(query="test")

        assert result["documents"] == []

    def test_to_dict(self):
        from haystack.components.retrievers.bedrock_knowledge_base_retriever import (
            BedrockKnowledgeBaseRetriever,
        )

        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id="TEST123456",
            region_name="us-west-2",
            number_of_results=10,
            knowledge_base_type="VECTOR",
        )
        serialized = retriever.to_dict()

        assert serialized["init_parameters"]["knowledge_base_id"] == "TEST123456"
        assert serialized["init_parameters"]["region_name"] == "us-west-2"
        assert serialized["init_parameters"]["number_of_results"] == 10
        assert serialized["init_parameters"]["knowledge_base_type"] == "VECTOR"
