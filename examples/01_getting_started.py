import os

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import AnswerParser, PromptNode, PromptTemplate, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.utils import fetch_archive_from_http
from haystack.utils import print_answers


def getting_started():
    # This getting_started shows you how to use LLMs with your data, a technique called Retrieval Augmented Generation - RAG

    # We are model agnostic :) Here, you can choose from: "Anthropic", "Cohere", "Huggingface", "OpenAI".
    provider = "OpenAI"
    # Please add an API key matching the provider.
    API_KEY = "ADD KEY HERE"

    # Loads good default pipelines for each provider.
    indexing_pipeline, query_pipeline = load_pipelines(provider=provider, API_KEY=API_KEY)

    # Downloads and adds a few Game of Thrones wiki articles to haystacks database.
    # You can also provide a folder with your local documents. Be aware that some of your data will be sent to external APIs!
    add_example_data(indexing_pipeline, dir="data/getting_started")

    result = query_pipeline.run(query="Who is the father of Arya Stark?")

    # This prints only the answer.
    # For details, like which documents were used to generate the answer, look into the <result> object
    print_answers(result, details="medium")


def load_pipelines(provider, API_KEY):
    # Query pipeline
    document_store = InMemoryDocumentStore(use_bm25=True)

    retriever = BM25Retriever(document_store=document_store, top_k=3)

    question_answering_with_references = PromptTemplate(
        prompt="deepset/question-answering-with-references",
        output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]"),
    )
    if provider == "Anthropic":
        prompt_node = PromptNode(
            model_name_or_path="claude-2", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "Cohere":
        prompt_node = PromptNode(
            model_name_or_path="command", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "Huggingface":
        # TODO: swap out for meta-llama/Llama-2-7b-chat-hf or the 40b model once supported in Haystack+HF API free tier
        prompt_node = PromptNode(
            model_name_or_path="tiiuae/falcon-40b-instruct",
            api_key=API_KEY,
            default_prompt_template=question_answering_with_references,
        )
    elif provider == "OpenAI":
        prompt_node = PromptNode(
            model_name_or_path="gpt-3.5-turbo-0301",
            api_key=API_KEY,
            default_prompt_template=question_answering_with_references,
        )

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    # Indexing pipeline
    # TODO: add link to example code for vector based indexing with sentence-transformer as default + option to use LLM embedding APIs
    indexing_pipeline = TextIndexingPipeline(document_store)
    return indexing_pipeline, query_pipeline


def add_example_data(indexing_pipeline, dir):
    # TODO add conversion for user supplied local folder (maybe using unstructured.io partition() or similar)
    fetch_archive_from_http(
        url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
        output_dir=dir,
    )

    files_to_index = [dir + "/" + f for f in os.listdir(dir)]
    indexing_pipeline.run_batch(file_paths=files_to_index)


if __name__ == "__main__":
    getting_started()
