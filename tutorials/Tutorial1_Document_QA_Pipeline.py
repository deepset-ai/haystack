from haystack import Finder
from haystack.database.sql import SQLDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.tfidf import TfidfRetriever
from haystack.utils import print_answers


## Indexing & cleaning documents

# Let's first get some documents that we want to query
# Here: 517 Wikipedia articles for Game of Thrones
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)


# The documents can be stored in different types of "DocumentStores".
# For dev we suggest a light-weight SQL DB
# For production we suggest elasticsearch
document_store = SQLDocumentStore(url="sqlite:///qa.db")

# Now, let's write the docs to our DB.
# You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
# It must take a str as input, and return a str.
write_documents_to_db(document_store=document_store, document_dir=doc_dir, clean_func=clean_wiki_text, only_empty_db=True)

## Initalize Reader, Retriever & Finder

# A retriever identifies the k most promising chunks of text that might contain the answer for our question
# Retrievers use some simple but fast algorithm, here: TF-IDF
retriever = TfidfRetriever(document_store=document_store)

# A reader scans the text chunks in detail and extracts the k best answers
# Reader use more powerful but slower deep learning models
# You can select a local model or any of the QA models published on huggingface's model hub (https://huggingface.co/models)
# here: a medium sized BERT QA model trained via FARM on Squad 2.0
# You can adjust the model to return "no answer possible" with the no_ans_boost. Higher values mean the model prefers "no answer possible"
reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2", use_gpu=False)

# OR: use alternatively a reader from huggingface's transformers package (https://github.com/huggingface/transformers)
# reader = TransformersReader(model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

# The Finder sticks together retriever and retriever in a pipeline to answer our actual questions
finder = Finder(reader, retriever)

## Voilá! Ask a question!
# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers.
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)

# to test impossible questions we need a large QA model, e.g. deepset/bert-large-uncased-whole-word-masking-squad2
# and we need to enable returning "no answer possible" by setting no_ans_boost=X in FARMReader
# prediction = finder.get_answers(question="Who is the first daughter of Arya Stark?", top_k_retriever=10, top_k_reader=5)

#prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
#prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")
