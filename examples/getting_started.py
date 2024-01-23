# Disable pylint errors for logging basicConfig
# pylint: disable=no-logging-basicconfig
import logging

from typing import Optional

from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import build_pipeline, add_example_data, print_answers

logging.basicConfig(level=logging.DEBUG)

def getting_started(provider, API_KEY, API_BASE: Optional[str] = None):
    """
    This getting_started example shows you how to use LLMs with your data with a technique called Retrieval Augmented Generation - RAG.

    :param provider: We are model agnostic :) Here, you can choose from: "anthropic", "cohere", "huggingface", and "openai".
    :param API_KEY: The API key matching the provider.
    :param API_BASE: The URL to use for a custom endpoint, e.g., if using LM Studio.  Only openai provider supported.  /v1 at the end is needed (e.g., http://localhost:1234/v1)

    """
    # We support many different databases. Here we load a simple and lightweight in-memory database.
    document_store = InMemoryDocumentStore(use_bm25=True)

    # Pipelines are the main abstraction in Haystack, they connect components like LLMs and databases.
    pipeline = build_pipeline(provider, API_KEY, document_store, API_BASE)

    # Download and add Game of Thrones TXT articles to Haystack's database.
    # You can also provide a folder with your local documents.
    # You might need to install additional dependencies - look inside the function for more information.
    add_example_data(document_store, "data/GoT_getting_started")

    # Ask a question on the data you just added.
    result = pipeline.run(query="Who is the father of Arya Stark?", debug=True)

    # For details such as which documents were used to generate the answer, look into the <result> object.
    print_answers(result, details="medium")
    return result

if __name__ == "__main__":    
    # getting_started(provider="openai", API_KEY="NOT NEEDED", API_BASE="http://192.168.1.100:1234/v1")
    getting_started(provider="openai", API_KEY="ADD KEY HERE")
