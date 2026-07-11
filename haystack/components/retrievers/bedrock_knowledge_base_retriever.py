# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Amazon Bedrock Knowledge Base Retriever for Haystack.

Retrieves documents from an Amazon Bedrock Managed Knowledge Base.
Supports both Managed (recommended) and Vector knowledge base types.

Usage:
    from haystack.components.retrievers.bedrock_knowledge_base_retriever import BedrockKnowledgeBaseRetriever

    retriever = BedrockKnowledgeBaseRetriever(
        knowledge_base_id="ABCDEFGHIJ",
        region_name="us-west-2",
    )

    result = retriever.run(query="What is retrieval augmented generation?")
    documents = result["documents"]
"""

import os
import logging
from typing import Any, Optional

from haystack import Document, component, default_to_dict

logger = logging.getLogger(__name__)


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get('location', {})
    loc_type = location.get('type', '')
    if loc_type == 'S3' or 's3Location' in location:
        return location.get('s3Location', {}).get('uri', '')
    elif loc_type == 'WEB' or 'webLocation' in location:
        return location.get('webLocation', {}).get('url', '')
    elif 'confluenceLocation' in location:
        return location.get('confluenceLocation', {}).get('url', '')
    elif 'salesforceLocation' in location:
        return location.get('salesforceLocation', {}).get('url', '')
    elif 'sharePointLocation' in location:
        return location.get('sharePointLocation', {}).get('url', '')
    elif 'customDocumentLocation' in location:
        return location.get('customDocumentLocation', {}).get('id', '')
    # Fallback to metadata._source_uri (for agentic results)
    return result.get('metadata', {}).get('_source_uri', '')


@component
class BedrockKnowledgeBaseRetriever:
    """
    Retrieves documents from an Amazon Bedrock Knowledge Base.

    Supports Managed Knowledge Bases (recommended, no external vector store needed)
    and traditional Vector Knowledge Bases.

    ### Usage example

    ```python
    from haystack.components.retrievers.bedrock_knowledge_base_retriever import BedrockKnowledgeBaseRetriever

    retriever = BedrockKnowledgeBaseRetriever(
        knowledge_base_id="ABCDEFGHIJ",
        region_name="us-west-2",
        knowledge_base_type="MANAGED",
    )

    result = retriever.run(query="What are the benefits of managed knowledge bases?")
    for doc in result["documents"]:
        print(doc.content)
        print(doc.meta["source"])
        print(doc.score)
    ```
    """

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        region_name: Optional[str] = None,
        number_of_results: int = 5,
        knowledge_base_type: str = "MANAGED",
        use_agentic_retrieval: Optional[bool] = None,
    ):
        """
        Create the BedrockKnowledgeBaseRetriever component.

        :param knowledge_base_id: The ID of the Bedrock Knowledge Base. Falls back to KNOWLEDGE_BASE_ID env var.
        :param region_name: AWS region. Falls back to AWS_REGION env var or us-east-1.
        :param number_of_results: Maximum number of results to return.
        :param knowledge_base_type: 'MANAGED' (recommended) or 'VECTOR'.
        :param use_agentic_retrieval: If True, try AgenticRetrieveStream before plain Retrieve. Defaults to USE_AGENTIC_RETRIEVAL env var or True.
        """
        self.knowledge_base_id = knowledge_base_id or os.environ.get("KNOWLEDGE_BASE_ID", "")
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.number_of_results = number_of_results
        self.knowledge_base_type = knowledge_base_type
        self.use_agentic_retrieval = use_agentic_retrieval if use_agentic_retrieval is not None else os.environ.get('USE_AGENTIC_RETRIEVAL', 'true').lower() != 'false'
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for BedrockKnowledgeBaseRetriever. "
                    "Install with: pip install boto3>=1.41.0"
                )
            self._client = boto3.client(
                "bedrock-agent-runtime", region_name=self.region_name
            )
        return self._client

    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: Optional[int] = None) -> dict[str, list[Document]]:
        """
        Retrieve documents from the Bedrock Knowledge Base.

        :param query: The search query.
        :param top_k: Maximum number of results. Overrides number_of_results if provided.
        :returns: A dictionary with a "documents" key containing the retrieved Documents.
        """
        k = top_k or self.number_of_results
        client = self._get_client()

        # Try agentic retrieval first
        if self.use_agentic_retrieval:
            try:
                response = client.agentic_retrieve_stream(
                    knowledgeBaseId=self.knowledge_base_id,
                    messages=[{"content": {"text": query}, "role": "user"}],
                    retrievers=[{
                        "configuration": {
                            "knowledgeBase": {
                                "knowledgeBaseId": self.knowledge_base_id,
                                "retrievalOverrides": {"maxNumberOfResults": k},
                            }
                        }
                    }],
                    agenticRetrieveConfiguration={
                        "foundationModelType": "MANAGED",
                        "rerankingModelType": "MANAGED",
                    },
                )
                documents = []
                for event in response.get("stream", []):
                    if "result" in event and "results" in event["result"]:
                        for result in event["result"]["results"]:
                            content = result.get("content", {}).get("text", "")
                            source = _get_source_uri(result)
                            score = result.get("score", 0.0)
                            doc = Document(
                                content=content,
                                meta={
                                    "source": source,
                                    "knowledge_base_id": self.knowledge_base_id,
                                    "knowledge_base_type": self.knowledge_base_type,
                                },
                                score=score,
                            )
                            documents.append(doc)
                if documents:
                    return {"documents": documents}
            except Exception:
                pass  # Fall through to plain retrieve

        if self.knowledge_base_type == "MANAGED":
            retrieval_config: dict[str, Any] = {
                "managedSearchConfiguration": {"numberOfResults": k}
            }
        else:
            retrieval_config = {
                "vectorSearchConfiguration": {"numberOfResults": k}
            }

        try:
            response = client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_config,
            )
        except Exception as e:
            logger.error(f"Error retrieving from Bedrock Knowledge Base: {e}")
            return {"documents": []}

        documents = []
        for result in response.get("retrievalResults", []):
            content = result.get("content", {}).get("text", "")
            source = _get_source_uri(result)
            score = result.get("score", 0.0)

            doc = Document(
                content=content,
                meta={
                    "source": source,
                    "knowledge_base_id": self.knowledge_base_id,
                    "knowledge_base_type": self.knowledge_base_type,
                },
                score=score,
            )
            documents.append(doc)

        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            knowledge_base_id=self.knowledge_base_id,
            region_name=self.region_name,
            number_of_results=self.number_of_results,
            knowledge_base_type=self.knowledge_base_type,
            use_agentic_retrieval=self.use_agentic_retrieval,
        )
