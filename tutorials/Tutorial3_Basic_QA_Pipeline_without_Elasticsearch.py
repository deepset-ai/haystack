from haystack import Finder
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
# from haystack.database.sql import SQLDocumentStore
from haystack.retriever.tfidf import TfidfRetriever
from haystack.database.memory import InMemoryDocumentStore

# In-Memory Document Store
document_store = InMemoryDocumentStore()

# SQLite Document Store
# document_store = SQLDocumentStore(url="sqlite:///qa.db")


# Cleaning & indexing documents
# 
# Haystack provides a customizable cleaning and indexing pipeline for ingesting documents in Document Stores.
# 
# In this tutorial, we download Wikipedia articles on Game of Thrones, apply a basic cleaning function, and index them in Elasticsearch.
# Let's first get some documents that we want to query
# Here: 517 Wikipedia articles for Game of Thrones
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Now, let's write the docs to our DB.
# You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
# It must take a str as input, and return a str.
write_documents_to_db(document_store=document_store, document_dir=doc_dir, clean_func=clean_wiki_text, only_empty_db=True)


# Initalize Retriever, Reader,  & Finder
# An in-memory TfidfRetriever based on Pandas dataframes
retriever = TfidfRetriever(document_store=document_store)

# we load a RoBERTa QA model trained via FARM on Squad 2.0
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
# reader = TransformersReader(model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

finder = Finder(reader, retriever)

## Voilà! Ask a question!

# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers. 
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
# prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
# prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")

