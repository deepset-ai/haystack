from haystack import Finder
from haystack.database.faiss import FAISSDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers
from haystack.retriever.dense import DensePassageRetriever


# FAISS is a library for efficient similarity search on a cluster of dense vectors.
# The FAISSDocumentStore uses a SQL(SQLite in-memory be default) database under-the-hood
# to store the document text and other meta data. The vector embeddings of the text are
# indexed on a FAISS Index that later is queried for searching answers.
document_store = FAISSDocumentStore()

# ## Cleaning & indexing documents
# Let's first get some documents that we want to query
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the docs to our DB.
document_store.write_documents(dicts)

### Retriever
retriever = DensePassageRetriever(document_store=document_store, embedding_model="dpr-bert-base-nq",
                                  do_lower_case=True, use_gpu=True, embed_title=True)

# Important:
# Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
document_store.update_embeddings(retriever)

### Reader
# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

### Finder
# The Finder sticks together reader and retriever in a pipeline to answer our actual questions.
finder = Finder(reader, retriever)

### Voilà! Ask a question!
# You can configure how many candidates the reader and retriever shall return
# The higher top_k_retriever, the better (but also the slower) your answers.
prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)


# prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
# prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)

print_answers(prediction, details="minimal")
