from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.tfidf import TfidfRetriever
from haystack import Finder
from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.indexing.cleaning import clean_wiki_text
from haystack.utils import print_answers

# Let's take a reader as a base model
fetch_archive_from_http(url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/0.3.0/bert-english-qa-large.tar.gz", output_dir="model")
reader = FARMReader(model_name_or_path="model/bert-english-qa-large", use_gpu=False)

# and fine-tune it on your own custom dataset (should be in squad like format)
reader.train(data_dir="../data/squad_small", train_filename="train.json", use_gpu=False, n_epochs=1)

# Okay, we have a fine-tuned model now. Let's test it on some docs:

## Let's get some docs for testing (see Tutorial 1 for more explanations)
from haystack.database import db
db.create_all()

# Download docs
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# Write docs to our DB.
write_documents_to_db(document_dir=doc_dir, clean_func=clean_wiki_text, only_empty_db=True)

# Initialize Finder Pipeline
retriever = TfidfRetriever()
finder = Finder(reader, retriever)

## Voilá! Ask a question!
# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers.
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)

#prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
#prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")



