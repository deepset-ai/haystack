from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.preprocessor.utils import fetch_archive_from_http
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader.farm import FARMReader
from haystack.finder import Finder
from farm.utils import initialize_device_settings

import logging
import subprocess
import time


def tutorial5_evaluation():
    logger = logging.getLogger(__name__)

    ##############################################
    # Settings
    ##############################################
    LAUNCH_ELASTICSEARCH = True

    eval_retriever_only = True
    eval_reader_only = False
    eval_both = False

    # make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted
    doc_index = "tutorial5_docs"
    label_index = "tutorial5_labels"
    ##############################################
    # Code
    ##############################################
    device, n_gpu = initialize_device_settings(use_cuda=True)
    # Start an Elasticsearch server
    # You can start Elasticsearch on your local machine instance using Docker. If Docker is not readily available in
    # your environment (eg., in Colab notebooks), then you can manually download and execute Elasticsearch from source.
    if LAUNCH_ELASTICSEARCH:
        logging.info("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.9.2'], shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                            "then set LAUNCH_ELASTICSEARCH in the script to False.")
        time.sleep(30)

    # Download evaluation data, which is a subset of Natural Questions development set containing 50 documents
    doc_dir = "../data/nq"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                create_index=False, embedding_field="emb",
                                                embedding_dim=768, excluded_meta_data=["emb"])


    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    document_store.delete_all_documents(index=doc_index)
    document_store.delete_all_documents(index=label_index)
    document_store.add_eval_data(filename="../data/nq/nq_dev_subset_v2.json", doc_index=doc_index, label_index=label_index)

    # Initialize Retriever
    retriever = ElasticsearchRetriever(document_store=document_store)

    # Alternative: Evaluate DensePassageRetriever
    # Note, that DPR works best when you index short passages < 512 tokens as only those tokens will be used for the embedding.
    # Here, for nq_dev_subset_v2.json we have avg. num of tokens = 5220(!).
    # DPR still outperforms Elastic's BM25 by a small margin here.
    # retriever = DensePassageRetriever(document_store=document_store,
    #                                   query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    #                                   passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    #                                   use_gpu=True,
    #                                   embed_title=True,
    #                                   remove_sep_tok_from_untitled_passages=True)
    # document_store.update_embeddings(retriever, index=doc_index)


    # Initialize Reader
    reader = FARMReader("deepset/roberta-base-squad2", top_k_per_candidate=4)

    # Initialize Finder which sticks together Reader and Retriever
    finder = Finder(reader, retriever)


    ## Evaluate Retriever on its own
    if eval_retriever_only:
        retriever_eval_results = retriever.eval(top_k=10, label_index=label_index, doc_index=doc_index)
        ## Retriever Recall is the proportion of questions for which the correct document containing the answer is
        ## among the correct documents
        print("Retriever Recall:", retriever_eval_results["recall"])
        ## Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
        print("Retriever Mean Avg Precision:", retriever_eval_results["map"])

    # Evaluate Reader on its own
    if eval_reader_only:
        reader_eval_results = reader.eval(document_store=document_store, device=device, label_index=label_index, doc_index=doc_index)
        # Evaluation of Reader can also be done directly on a SQuAD-formatted file without passing the data to Elasticsearch
        #reader_eval_results = reader.eval_on_file("../data/nq", "nq_dev_subset_v2.json", device=device)

        ## Reader Top-N-Accuracy is the proportion of predicted answers that match with their corresponding correct answer
        print("Reader Top-N-Accuracy:", reader_eval_results["top_n_accuracy"])
        ## Reader Exact Match is the proportion of questions where the predicted answer is exactly the same as the correct answer
        print("Reader Exact Match:", reader_eval_results["EM"])
        ## Reader F1-Score is the average overlap between the predicted answers and the correct answers
        print("Reader F1-Score:", reader_eval_results["f1"])


    # Evaluate combination of Reader and Retriever through Finder
    if eval_both:
        finder_eval_results = finder.eval(top_k_retriever=1, top_k_reader=10, label_index=label_index, doc_index=doc_index)
        finder.print_eval_results(finder_eval_results)


if __name__ == "__main__":
    tutorial5_evaluation()
