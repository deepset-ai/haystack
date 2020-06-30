import logging
import subprocess
import time

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers
from haystack.retriever.sparse import ElasticsearchRetriever

logger = logging.getLogger(__name__)

LAUNCH_ELASTICSEARCH = True

# Start an Elasticsearch server
# You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
# your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.

if LAUNCH_ELASTICSEARCH:
    logging.info("Starting Elasticsearch ...")
    status = subprocess.run(
        ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
    )
    if status.returncode:
        raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                        "then set LAUNCH_ELASTICSEARCH in the script to False.")
    time.sleep(15)

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

# ## Cleaning & indexing documents
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the docs to our DB.
if LAUNCH_ELASTICSEARCH:
    document_store.write_documents(dicts)
else:
    logger.warning("Since we already have a running ES instance we should not index the same documents again. \n"
                   "If you still want to do this call: document_store.write_documents(dicts) manually ")


# Retriever

retriever = ElasticsearchRetriever(document_store=document_store)

# FARMReader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# Finder
finder = Finder(reader, retriever)

prediction = finder.get_answers(question="Who is the father of Sansa Stark?", top_k_retriever=10, top_k_reader=5)


# prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
# prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")
