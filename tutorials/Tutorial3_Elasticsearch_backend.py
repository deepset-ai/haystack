from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore

from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.reader.transformers import TransformersReader
from haystack.retriever.elasticsearch import ElasticsearchRetriever
from haystack.utils import print_answers


# Our pipeline remains very similar to the one in Tutorial 1, where we had a SQL backend
# (https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial1_Basic_QA_Pipeline.py)
# We therefore only highlight the three key differences here

# Get documents (same as in Tutorial 1)
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Difference 1: Initialize a document store for Elasticsearch
# This requires a running Elasticsearch instance. To run one locally you can execute:
# docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.5.1
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

# Difference 2: Split docs into paragraphs before indexing (set split_paragraphs=True)
write_documents_to_db(document_store=document_store, document_dir=doc_dir, clean_func=clean_wiki_text,
                      only_empty_db=True, split_paragraphs=True)

# Difference 3: Use the native Elasticsearch implementation of BM25 as a Retriever
retriever = ElasticsearchRetriever(document_store=document_store)

# Init reader & and use Finder to get answer (same as in Tutorial 1)
reader = TransformersReader(model="deepset/bert-base-cased-squad2",tokenizer="deepset/bert-base-cased-squad2",use_gpu=-1)
finder = Finder(reader, retriever)
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
print_answers(prediction, details="minimal")
