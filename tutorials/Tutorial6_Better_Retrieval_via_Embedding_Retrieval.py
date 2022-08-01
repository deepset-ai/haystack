import logging

# We configure how logging messages should be displayed and which log level should be used before importing Haystack.
# Example log message:
# INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
# Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import FAISSDocumentStore, MilvusDocumentStore
from haystack.utils import clean_wiki_text, print_answers, launch_milvus, convert_files_to_docs, fetch_archive_from_http
from haystack.nodes import FARMReader, EmbeddingRetriever


def tutorial6_better_retrieval_via_embedding_retrieval():
    # OPTION 1: FAISS is a library for efficient similarity search on a cluster of dense vectors.
    # The FAISSDocumentStore uses a SQL(SQLite in-memory be default) document store under-the-hood
    # to store the document text and other meta data. The vector embeddings of the text are
    # indexed on a FAISS Index that later is queried for searching answers.
    # The default flavour of FAISSDocumentStore is "Flat" but can also be set to "HNSW" for
    # faster search at the expense of some accuracy. Just set the faiss_index_factor_str argument in the constructor.
    # For more info on which suits your use case: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

    # Do not forget to install its dependencies with `pip install farm-haystack[faiss]`
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    # OPTION2: Milvus is an open source database library that is also optimized for vector similarity searches like FAISS.
    # Like FAISS it has both a "Flat" and "HNSW" mode but it outperforms FAISS when it comes to dynamic data management.
    # It does require a little more setup, however, as it is run through Docker and requires the setup of some config files.
    # See https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md

    # Do not forget to install its dependencies with `pip install farm-haystack[milvus]`
    # launch_milvus()
    # document_store = MilvusDocumentStore()

    # ## Preprocessing of documents
    # Let's first get some documents that we want to query
    doc_dir = "data/tutorial6"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Now, let's write the docs to our DB.
    document_store.write_documents(docs)

    ### Retriever
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
    )

    # Important:
    # Now that we initialized the Retriever, we need to call update_embeddings() to iterate over all
    # previously indexed documents and update their embedding representation.
    # While this can be a time consuming operation (depending on the corpus size), it only needs to be done once.
    # At query time, we only need to embed the query and compare it to the existing document embeddings, which is very fast.
    document_store.update_embeddings(retriever)

    ### Reader
    # Load a  local model or any of the QA models on
    # Hugging Face's model hub (https://huggingface.co/models)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    ### Pipeline
    from haystack.pipelines import ExtractiveQAPipeline

    pipe = ExtractiveQAPipeline(reader, retriever)

    ## Voilà! Ask a question!
    prediction = pipe.run(
        query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", params={"Reader": {"top_k": 5}})
    # prediction = pipe.run(query="Who is the sister of Sansa?", params={"Reader": {"top_k": 5}})

    print_answers(prediction, details="minimum")


if __name__ == "__main__":
    tutorial6_better_retrieval_via_embedding_retrieval()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
