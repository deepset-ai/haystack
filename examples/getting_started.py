from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import build_pipeline, add_example_data, print_answers


def getting_started(provider, API_KEY):
    """
    This getting_started example shows you how to use LLMs with your data with a technique called Retrieval Augmented Generation - RAG.

    :param provider: We are model agnostic :) Here, you can choose from: "anthropic", "cohere", "huggingface", and "openai".
    :param API_KEY: The API key matching the provider.

    """

    # We support many different databases. Here we load a simple and lightweight in-memory database.
    document_store = InMemoryDocumentStore(use_bm25=True)

    # Pipelines are the main abstraction in Haystack, they connect components like LLMs and databases.
    pipeline = build_pipeline(provider, API_KEY, document_store)

    # Download and add Game of Thrones TXT articles to Haystack's database.
    # You can also provide a folder with your local documents.
    # You might need to install additional dependencies - look inside the function for more information.
    add_example_data(document_store, "data/GoT_getting_started")

    # Ask a question on the data you just added.
    result = pipeline.run(query="Who is the father of Arya Stark?")

    # For details such as which documents were used to generate the answer, look into the <result> object.
    print_answers(result, details="medium")
    return result


if __name__ == "__main__":
    getting_started(provider="openai", API_KEY="ADD KEY HERE")
