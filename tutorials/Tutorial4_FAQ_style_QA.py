from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore

from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers
import pandas as pd
import requests
import logging
import subprocess
import time
## "FAQ-Style QA": Utilizing existing FAQs for Question Answering

# While *extractive Question Answering* works on pure texts and is therefore more generalizable, there's also a common alternative that utilizes existing FAQ data.
#
# Pros:
# - Very fast at inference time
# - Utilize existing FAQ data
# - Quite good control over answers
#
# Cons:
# - Generalizability: We can only answer questions that are similar to existing ones in FAQ
#
# In some use cases, a combination of extractive QA and FAQ-style can also be an interesting option.
LAUNCH_ELASTICSEARCH=True

if LAUNCH_ELASTICSEARCH:
    logging.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
    )
    if status.returncode:
        raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                        "then set LAUNCH_ELASTICSEARCH in the script to False.")
    time.sleep(15)

### Init the DocumentStore
# In contrast to Tutorial 1 (extractive QA), we:
#
# * specify the name of our `text_field` in Elasticsearch that we want to return as an answer
# * specify the name of our `embedding_field` in Elasticsearch where we'll store the embedding of our question and that is used later for calculating our similarity to the incoming user question
# * set `excluded_meta_data=["question_emb"]` so that we don't return the huge embedding vectors in our search results

document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                            index="document",
                                            embedding_field="question_emb",
                                            embedding_dim=768,
                                            excluded_meta_data=["question_emb"],
                                            similarity="cosine")

### Create a Retriever using embeddings
# Instead of retrieving via Elasticsearch's plain BM25, we want to use vector similarity of the questions (user question vs. FAQ ones).
# We can use the `EmbeddingRetriever` for this purpose and specify a model that we use for the embeddings.
#
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert", use_gpu=True)

# Download a csv containing some FAQ data
# Here: Some question-answer pairs related to COVID-19
temp = requests.get("https://raw.githubusercontent.com/deepset-ai/COVID-QA/master/data/faqs/faq_covidbert.csv")
open('small_faq_covid.csv', 'wb').write(temp.content)

# Get dataframe with columns "question", "answer" and some custom metadata
df = pd.read_csv("small_faq_covid.csv")
# Minimal cleaning
df.fillna(value="", inplace=True)
df["question"] = df["question"].apply(lambda x: x.strip())
print(df.head())

# Get embeddings for our questions from the FAQs
questions = list(df["question"].values)
df["question_emb"] = retriever.embed_queries(texts=questions)
df = df.rename(columns={"answer": "text"})

# Convert Dataframe to list of dicts and index them in our DocumentStore
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)


# Init reader & and use Finder to get answer (same as in Tutorial 1)
finder = Finder(reader=None, retriever=retriever)
prediction = finder.get_answers_via_similar_questions(question="How is the virus spreading?", top_k_retriever=10)
print_answers(prediction, details="all")
