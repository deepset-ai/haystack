import logging
import os

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import AnswerParser, PromptNode, PromptTemplate, BM25Retriever, PreProcessor, TextConverter
from haystack.nodes import (
    TextConverter,
    FileTypeClassifier,
    PDFToTextConverter,
    MarkdownConverter,
    DocxToTextConverter,
    PreProcessor,
)
from haystack.pipelines import Pipeline
from haystack.utils import fetch_archive_from_http
from haystack.utils import print_answers

logger = logging.getLogger(__name__)


def getting_started():
    # This getting_started shows you how to use LLMs with your data, a technique called Retrieval Augmented Generation - RAG

    # We are model agnostic :) Here, you can choose from: "anthropic", "cohere", "huggingface", and "openai".
    provider = "openai"
    API_KEY = "ADD KEY HERE"

    # We support many different databases. Here we load a simple and lightweight in-memory database.
    document_store = InMemoryDocumentStore(use_bm25=True)

    # Pipelines are the main abstraction in Haystack. We separate them into data handling (indexing) and AI logic (query) pipelines.
    indexing_pipeline, query_pipeline = build_pipelines(provider, API_KEY, document_store)

    # Downloads and adds Game of Thrones TXT articles to haystacks database.
    add_example_data(indexing_pipeline, "data/GoT_getting_started")

    # Ask a question on the data you just added.
    result = query_pipeline.run(query="What is deepset?")

    # For details, like which documents were used to generate the answer, look into the <result> object
    print_answers(result, details="medium")


def build_pipelines(provider, API_KEY, document_store):
    provider = provider.lower()

    ########################
    ### Indexing pipeline ###
    ########################
    # Load nodes for handling TXT files
    text_converter = TextConverter()
    preprocessor = PreProcessor()
    # Compose the indexing pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

    ########################
    ### Query pipeline ###
    ########################
    # A retriever selects the right documents when given a question.
    retriever = BM25Retriever(document_store=document_store, top_k=3)
    # Load prompt for doing retrieval augmented generation from https://prompthub.deepset.ai/?prompt=deepset%2Fquestion-answering-with-references
    question_answering_with_references = PromptTemplate(
        prompt="deepset/question-answering-with-references",
        output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]"),
    )
    # Load the LLM model
    if provider == "anthropic":
        prompt_node = PromptNode(
            model_name_or_path="claude-2", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "cohere":
        prompt_node = PromptNode(
            model_name_or_path="command", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "huggingface":
        # TODO: swap out for meta-llama/Llama-2-7b-chat-hf or the 40b model once supported in Haystack+HF API free tier
        # The tiiuae/falcon-40b-instruct model cannot handle a complex prompt with references, so we use a very simple one
        prompt_node = PromptNode(
            model_name_or_path="tiiuae/falcon-40b-instruct",
            api_key=API_KEY,
            default_prompt_template=PromptTemplate(prompt="deepset/question-answering"),
        )
    elif provider == "openai":
        prompt_node = PromptNode(
            model_name_or_path="gpt-3.5-turbo-0301",
            api_key=API_KEY,
            default_prompt_template=question_answering_with_references,
        )
    else:
        logger.error('Given <provider> unknown. Please use any of "anthropic", "cohere", "huggingface", or "openai"')
    # Compose the query pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    return indexing_pipeline, query_pipeline


def add_example_data(indexing_pipeline, dir):
    fetch_archive_from_http(
        url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip",
        output_dir=dir,
    )

    files_to_index = [dir + "/" + f for f in os.listdir(dir)]
    indexing_pipeline.run(file_paths=files_to_index)


if __name__ == "__main__":
    getting_started()
