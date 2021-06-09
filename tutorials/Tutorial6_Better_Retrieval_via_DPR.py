from haystack import Finder
from haystack.document_store import FAISSDocumentStore, MilvusDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers, launch_milvus
from haystack.retriever.dense import DensePassageRetriever

def tutorial6_better_retrieval_via_dpr():
    # OPTION 1: FAISS is a library for efficient similarity search on a cluster of dense vectors.
    # The FAISSDocumentStore uses a SQL(SQLite in-memory be default) document store under-the-hood
    # to store the document text and other meta data. The vector embeddings of the text are
    # indexed on a FAISS Index that later is queried for searching answers.
    # The default flavour of FAISSDocumentStore is "Flat" but can also be set to "HNSW" for
    # faster search at the expense of some accuracy. Just set the faiss_index_factor_str argument in the constructor.
    # For more info on which suits your use case: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    # OPTION2: Milvus is an open source database library that is also optimized for vector similarity searches like FAISS.
    # Like FAISS it has both a "Flat" and "HNSW" mode but it outperforms FAISS when it comes to dynamic data management.
    # It does require a little more setup, however, as it is run through Docker and requires the setup of some config files.
    # See https://milvus.io/docs/v1.0.0/milvus_docker-cpu.md
    # launch_milvus()
    # document_store = MilvusDocumentStore()

    # ## Preprocessing of documents
    # Let's first get some documents that we want to query
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Now, let's write the docs to our DB.
    document_store.write_documents(dicts)

    ### Retriever
    retriever = DensePassageRetriever(document_store=document_store,
                                      query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      max_seq_len_query=64,
                                      max_seq_len_passage=256,
                                      batch_size=2,
                                      use_gpu=True,
                                      embed_title=True,
                                      use_fast_tokenizers=True
                                      )

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

    ### Pipeline
    from haystack.pipeline import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    ## Voilà! Ask a question!
    prediction = pipe.run(query="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", top_k_reader=5)
    # prediction = pipe.run(query="Who is the sister of Sansa?", top_k_reader=5)

    print_answers(prediction, details="minimal")


if __name__ == "__main__":
    tutorial6_better_retrieval_via_dpr()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/