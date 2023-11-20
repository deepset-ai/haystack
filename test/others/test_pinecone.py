import os
from haystack.document_stores import PineconeDocumentStore
from dotenv import load_dotenv
import pinecone

load_dotenv()


def test_document_store_properties():
    pods = 1
    pod_type = "starter"
    environment = "gcp-starter"
    index_name = "docs"

    document_store = PineconeDocumentStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=environment,
        pods=pods,
        pod_type=pod_type,
        similarity="cosine",
        embedding_dim=768,
        index=index_name,
    )
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=environment)
    index_description = pinecone.describe_index(document_store.index)
    assert index_description.pods == document_store.pods
    assert index_description.pod_type == document_store.pod_type
    pinecone.delete_index(index_name)
