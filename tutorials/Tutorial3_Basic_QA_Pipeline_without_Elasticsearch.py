# ## Task: Build a Question Answering pipeline without Elasticsearch
#
# Haystack provides alternatives to Elasticsearch for developing quick prototypes.
#
# You can use an `InMemoryDocumentStore` or a `SQLDocumentStore`(with SQLite) as the document store.
#
# If you are interested in more feature-rich Elasticsearch, then please refer to the Tutorial 1.


from haystack import Finder
from haystack.database.memory import InMemoryDocumentStore
from haystack.database.sql import SQLDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.sparse import TfidfRetriever
from haystack.utils import print_answers


# In-Memory Document Store
document_store = InMemoryDocumentStore()

# or, alternatively, SQLite Document Store
# document_store = SQLDocumentStore(url="sqlite:///qa.db")


# ## Cleaning & indexing documents
#
# Haystack provides a customizable cleaning and indexing pipeline for ingesting documents in Document Stores.
#
# In this tutorial, we download Wikipedia articles on Game of Thrones, apply a basic cleaning function, and index
# them in Elasticsearch.
# Let's first get some documents that we want to query
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
# Retrievers help narrowing down the scope for the Reader to smaller units of text where
# a given question could be answered.
#
# With InMemoryDocumentStore or SQLDocumentStore, you can use the TfidfRetriever. For more
# retrievers, please refer to the tutorial-1.

# An in-memory TfidfRetriever based on Pandas dataframes
retriever = TfidfRetriever(document_store=document_store)

# ### Reader
#
# A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
# on powerful, but slower deep learning models.
#
# Haystack currently supports Readers based on the frameworks FARM and Transformers.
# With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).

# **Here:**                   a medium sized RoBERTa QA model using a Reader based on
#                             FARM (https://huggingface.co/deepset/roberta-base-squad2)
# **Alternatives (Reader):**  TransformersReader (leveraging the `pipeline` of the Transformers package)
# **Alternatives (Models):**  e.g. "distilbert-base-uncased-distilled-squad" (fast) or
#                             "deepset/bert-large-uncased-whole-word-masking-squad2" (good accuracy)
# **Hint:**                   You can adjust the model to return "no answer possible" with the no_ans_boost.
#                             Higher values mean the model prefers "no answer possible".

# #### FARMReader
#
# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)


# #### TransformersReader
# Alternative:
# reader = TransformersReader(model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

# ### Finder
#
# The Finder sticks together reader and retriever in a pipeline to answer our actual questions.
finder = Finder(reader, retriever)

# ## Voilà! Ask a question!

# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers.
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
# prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
# prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")
