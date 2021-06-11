# ## Task: Question Answering for Game of Thrones
#
# Question Answering can be used in a variety of use cases. A very common one:  Using it to navigate through complex
# knowledge bases or long documents ("search setting").
#
# A "knowledge base" could for example be your website, an internal wiki or a collection of financial reports.
# In this tutorial we will work on a slightly different domain: "Game of Thrones".
#
# Let's see how we can use a bunch of Wikipedia articles to answer a variety of questions about the
# marvellous seven kingdoms...

import logging
import subprocess
import time

from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers, launch_es
from haystack.retriever.sparse import ElasticsearchRetriever


def tutorial1_basic_qa_pipeline():
    logger = logging.getLogger(__name__)

    # ## Document Store
    #
    # Haystack finds answers to queries within the documents stored in a `DocumentStore`. The current implementations of
    # `DocumentStore` include `ElasticsearchDocumentStore`, `FAISSDocumentStore`, `SQLDocumentStore`, and `InMemoryDocumentStore`.
    #
    # **Here:** We recommended Elasticsearch as it comes preloaded with features like full-text queries, BM25 retrieval,
    # and vector storage for text embeddings.
    # **Alternatives:** If you are unable to setup an Elasticsearch instance, then follow the Tutorial 3
    # for using SQL/InMemory document stores.
    # **Hint**:
    # This tutorial creates a new document store instance with Wikipedia articles on Game of Thrones. However, you can
    # configure Haystack to work with your existing document stores.
    #
    # Start an Elasticsearch server
    # You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
    # your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.

    launch_es()

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # ## Preprocessing of documents
    #
    # Haystack provides a customizable pipeline for:
    # - converting files into texts
    # - cleaning texts
    # - splitting texts
    # - writing them to a Document Store

    # In this tutorial, we download Wikipedia articles about Game of Thrones, apply a basic cleaning function, and add
    # them in Elasticsearch.


    # Let's first fetch some documents that we want to query
    # Here: 517 Wikipedia articles for Game of Thrones
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
    # It must take a str as input, and return a str.

    # Now, let's write the docs to our DB.
    document_store.write_documents(dicts)

    # ## Initalize Retriever, Reader,  & Finder
    #
    # ### Retriever
    #
    # Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question
    # could be answered.
    #
    # They use some simple but fast algorithm.
    # **Here:** We use Elasticsearch's default BM25 algorithm
    # **Alternatives:**
    # - Customize the `ElasticsearchRetriever`with custom queries (e.g. boosting) and filters
    # - Use `EmbeddingRetriever` to find candidate documents based on the similarity of
    #   embeddings (e.g. created via Sentence-BERT)
    # - Use `TfidfRetriever` in combination with a SQL or InMemory Document store for simple prototyping and debugging

    retriever = ElasticsearchRetriever(document_store=document_store)

    # Alternative: An in-memory TfidfRetriever based on Pandas dataframes for building quick-prototypes
    # with SQLite document store.
    #
    # from haystack.retriever.tfidf import TfidfRetriever
    # retriever = TfidfRetriever(document_store=document_store)

    # ### Reader
    #
    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
    # on powerful, but slower deep learning models.
    #
    # Haystack currently supports Readers based on the frameworks FARM and Transformers.
    # With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).
    # **Here:** a medium sized RoBERTa QA model using a Reader based on
    #           FARM (https://huggingface.co/deepset/roberta-base-squad2)
    # **Alternatives (Reader):** TransformersReader (leveraging the `pipeline` of the Transformers package)
    # **Alternatives (Models):** e.g. "distilbert-base-uncased-distilled-squad" (fast) or
    #                            "deepset/bert-large-uncased-whole-word-masking-squad2" (good accuracy)
    # **Hint:** You can adjust the model to return "no answer possible" with the no_ans_boost. Higher values mean
    #           the model prefers "no answer possible"
    #
    # #### FARMReader

    # Load a  local model or any of the QA models on
    # Hugging Face's model hub (https://huggingface.co/models)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # #### TransformersReader

    # Alternative:
    # reader = TransformersReader(
    #    model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

    # ### Pipeline
    # 
    # With a Haystack `Pipeline` you can stick together your building blocks to a search pipeline.
    # Under the hood, `Pipelines` are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.
    # To speed things up, Haystack also comes with a few predefined Pipelines. One of them is the `ExtractiveQAPipeline` that combines a retriever and a reader to answer our questions.
    # You can learn more about `Pipelines` in the [docs](https://haystack.deepset.ai/docs/latest/pipelinesmd).
    from haystack.pipeline import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)
    
    ## Voilà! Ask a question!
    prediction = pipe.run(query="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", top_k_reader=5)
    # prediction = pipe.run(query="Who is the sister of Sansa?", top_k_reader=5)

    print_answers(prediction, details="minimal")


if __name__ == "__main__":
    tutorial1_basic_qa_pipeline()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/