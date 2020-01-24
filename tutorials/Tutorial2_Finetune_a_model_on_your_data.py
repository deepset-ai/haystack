
from haystack import Finder
from haystack.database.sql import SQLDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.retriever.tfidf import TfidfRetriever
from haystack.utils import print_answers

#### TRAINING #############
# Let's take a reader as a base model
reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)

# and fine-tune it on your own custom dataset (should be in SQuAD like format)
train_data = "PATH/TO_YOUR/TRAIN_DATA"
reader.train(data_dir=train_data, train_filename="train.json", use_gpu=False, n_epochs=1)


#### Use it (same as in Tutorial 1) #############

## Indexing & cleaning documents

# Let's get the data (Game of thrones articles from wikipedia)
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)


# Init Document store & write docs to it
document_store = SQLDocumentStore(url="sqlite:///qa.db")
write_documents_to_db(document_store=document_store, document_dir=doc_dir, clean_func=clean_wiki_text, only_empty_db=True)

## Initalize Reader, Retriever & Finder

# A retriever identifies the k most promising chunks of text that might contain the answer for our question
# Retrievers use some simple but fast algorithm, here: TF-IDF
retriever = TfidfRetriever(document_store=document_store)

# The Finder sticks together retriever and retriever in a pipeline to answer our actual questions
finder = Finder(reader, retriever)

## Voilá! Ask a question!
# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers.
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)

#prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
#prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")



