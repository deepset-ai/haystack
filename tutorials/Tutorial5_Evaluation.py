from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import (
    ElasticsearchRetriever,
    DensePassageRetriever,
    EmbeddingRetriever,
    FARMReader,
    PreProcessor
)
from haystack.utils import fetch_archive_from_http, launch_es
from haystack.modeling.utils import initialize_device_settings
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel, Span

import logging

logger = logging.getLogger(__name__)


def tutorial5_evaluation():

    # make sure these indices do not collide with existing ones, the indices will be wiped clean before data is inserted
    doc_index = "tutorial5_docs"
    label_index = "tutorial5_labels"

    ##############################################
    # Code
    ##############################################
    launch_es()
    devices, n_gpu = initialize_device_settings(use_cuda=True)

    # Download evaluation data, which is a subset of Natural Questions development set containing 50 documents with one question per document and multiple annotated answers
    doc_dir = "../data/nq"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/nq_dev_subset_v2.json.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(
        host="localhost", username="", password="", index=doc_index,
        label_index=label_index, embedding_field="emb",
        embedding_dim=768, excluded_meta_data=["emb"]
    )

    # Add evaluation data to Elasticsearch document store
    # We first delete the custom tutorial indices to not have duplicate elements
    # and also split our documents into shorter passages using the PreProcessor
    preprocessor = PreProcessor(
        split_by="word",
        split_length=200,
        split_overlap=0,
        split_respect_sentence_boundary=False,
        clean_empty_lines=False,
        clean_whitespace=False
    )
    document_store.delete_documents(index=doc_index)
    document_store.delete_documents(index=label_index)

    # The add_eval_data() method converts the given dataset in json format into Haystack document and label objects.
    # Those objects are then indexed in their respective document and label index in the document store.
    # The method can be used with any dataset in SQuAD format.
    document_store.add_eval_data(
        filename="../data/nq/nq_dev_subset_v2.json",
        doc_index=doc_index,
        label_index=label_index,
        preprocessor=preprocessor
    )


    # Initialize Retriever
    retriever = ElasticsearchRetriever(document_store=document_store)

    # Alternative: Evaluate dense retrievers (DensePassageRetriever or EmbeddingRetriever)
    # DensePassageRetriever uses two separate transformer based encoders for query and document.
    # In contrast, EmbeddingRetriever uses a single encoder for both.
    # Please make sure the "embedding_dim" parameter in the DocumentStore above matches the output dimension of your models!
    # Please also take care that the PreProcessor splits your files into chunks that can be completely converted with
    #        the max_seq_len limitations of Transformers
    # The SentenceTransformer model "all-mpnet-base-v2" generally works well with the EmbeddingRetriever on any kind of English text.
    # For more information check out the documentation at: https://www.sbert.net/docs/pretrained_models.html
    # retriever = DensePassageRetriever(document_store=document_store,
    #                                   query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    #                                   passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    #                                   use_gpu=True,
    #                                   max_seq_len_passage=256,
    #                                   embed_title=True)
    # retriever = EmbeddingRetriever(document_store=document_store, model_format="sentence_transformers",
    #                                embedding_model="all-mpnet-base-v2")
    # document_store.update_embeddings(retriever, index=doc_index)

    # Initialize Reader
    reader = FARMReader(
        model_name_or_path="deepset/roberta-base-squad2",
        top_k=4,
        return_no_answer=True
    )

    # Define a pipeline consisting of the initialized retriever and reader
    # Here we evaluate retriever and reader in open domain fashion on the full corpus of documents i.e. a document is considered
    # correctly retrieved if it contains the gold answer string within it. The reader is evaluated based purely on the
    # predicted answer string, regardless of which document this came from and the position of the extracted span.
    # The generation of predictions is seperated from the calculation of metrics.
    # This allows you to run the computation-heavy model predictions only once and then iterate flexibly on the metrics or reports you want to generate.

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    # The evaluation also works with any other pipeline.
    # For example you could use a DocumentSearchPipeline as an alternative:
    # pipeline = DocumentSearchPipeline(retriever=retriever)

    # We can load evaluation labels from the document store
    eval_labels = document_store.get_all_labels_aggregated(drop_negative_labels=True, drop_no_answers=False)

    # Alternative: Define queries and labels directly
    # eval_labels = [
    #        MultiLabel(labels=[Label(query="who is written in the book of life",
    #        answer=Answer(answer="every person who is destined for Heaven or the World to Come",
    #        offsets_in_context=[Span(374, 434)]),
    #        document=Document(id='1b090aec7dbd1af6739c4c80f8995877-0',
    #        content_type="text",
    #        content='Book of Life - wikipedia Book of Life Jump to: navigation, search This article is about the book mentioned in Christian and Jewish religious teachings. For other uses, see The Book of Life. In Christianity and Judaism, the Book of Life (Hebrew: ספר החיים, transliterated Sefer HaChaim; Greek: βιβλίον τῆς ζωῆς Biblíon tēs Zōēs) is the book in which God records the names of every person who is destined for Heaven or the World to Come. According to the Talmud it is open on Rosh Hashanah, as is its analog for the wicked, the Book of the Dead. For this reason extra mention is made for the Book of Life during Amidah recitations during the Days of Awe, the ten days between Rosh Hashanah, the Jewish new year, and Yom Kippur, the day of atonement (the two High Holidays, particularly in the prayer Unetaneh Tokef). Contents (hide) 1 In the Hebrew Bible 2 Book of Jubilees 3 References in the New Testament 4 The eschatological or annual roll-call 5 Fundraising 6 See also 7 Notes 8 References In the Hebrew Bible(edit) In the Hebrew Bible the Book of Life - the book or muster-roll of God - records forever all people considered righteous before God'),
    #        is_correct_answer=True,
    #        is_correct_document=True,
    #        origin="gold-label")])
    #    ]

    # Similar to pipeline.run() we can execute pipeline.eval()
    eval_result = pipeline.eval(
        labels=eval_labels,
        params={"Retriever": {"top_k": 5}}
    )

    # The EvaluationResult contains a pandas dataframe for each pipeline node.
    # That's why there are two dataframes in the EvaluationResult of an ExtractiveQAPipeline.

    retriever_result = eval_result["Retriever"]
    retriever_result.head()

    reader_result = eval_result["Reader"]
    reader_result.head()

    # We can filter for all documents retrieved for a given query
    retriever_book_of_life = retriever_result[retriever_result['query'] == "who is written in the book of life"]

    # We can also filter for all answers predicted for a given query
    reader_book_of_life = reader_result[reader_result['query'] == "who is written in the book of life"]

    # Save the evaluation result so that we can reload it later and calculate evaluation metrics without running the pipeline again.
    eval_result.save("../")

    ## Calculating Evaluation Metrics
    # Load an EvaluationResult to quickly calculate standard evaluation metrics for all predictions, such as F1-score of each individual prediction of the Reader node or recall of the retriever.

    saved_eval_result = EvaluationResult.load("../")
    metrics = saved_eval_result.calculate_metrics()
    print(f'Retriever - Recall (single relevant document): {metrics["Retriever"]["recall_single_hit"]}')
    print(f'Retriever - Recall (multiple relevant documents): {metrics["Retriever"]["recall_multi_hit"]}')
    print(f'Retriever - Mean Reciprocal Rank: {metrics["Retriever"]["mrr"]}')
    print(f'Retriever - Precision: {metrics["Retriever"]["precision"]}')
    print(f'Retriever - Mean Average Precision: {metrics["Retriever"]["map"]}')

    print(f'Reader - F1-Score: {metrics["Reader"]["f1"]}')
    print(f'Reader - Exact Match: {metrics["Reader"]["exact_match"]}')

    ## Generating an Evaluation Report
    # A summary of the evaluation results can be printed to get a quick overview. It includes some aggregated metrics and also shows a few wrongly predicted examples.

    pipeline.print_eval_report(saved_eval_result)

    ## Advanced Evaluation Metrics
    # As an advanced evaluation metric, semantic answer similarity (SAS) can be calculated. This metric takes into account whether the meaning of a predicted answer is similar to the annotated gold answer rather than just doing string comparison.
    # To this end SAS relies on pre-trained models. For English, we recommend "cross-encoder/stsb-roberta-large", whereas for German we recommend "deepset/gbert-large-sts". A good multilingual model is "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".
    # More info on this metric can be found in our [paper](https://arxiv.org/abs/2108.06130) or in our [blog post](https://www.deepset.ai/blog/semantic-answer-similarity-to-evaluate-qa).

    advanced_eval_result = pipeline.eval(
            labels=eval_labels,
            params={"Retriever": {"top_k": 1}},
            sas_model_name_or_path="cross-encoder/stsb-roberta-large"
        )

    metrics = advanced_eval_result.calculate_metrics()
    print(metrics["Reader"]["sas"])


    # Evaluate Retriever on its own
    # Here we evaluate only the retriever, based on whether the gold_label document is retrieved.
    retriever_eval_results = retriever.eval(top_k=10, label_index=label_index, doc_index=doc_index)
    ## Retriever Recall is the proportion of questions for which the correct document containing the answer is
    ## among the correct documents
    print("Retriever Recall:", retriever_eval_results["recall"])
    ## Retriever Mean Avg Precision rewards retrievers that give relevant documents a higher rank
    print("Retriever Mean Avg Precision:", retriever_eval_results["map"])


    # Evaluate Reader on its own
    # Here we evaluate only the reader in a closed domain fashion i.e. the reader is given one query
    # and its corresponding relevant document and metrics are calculated on whether the right position in this text is selected by
    # the model as the answer span (i.e. SQuAD style)
    reader_eval_results = reader.eval(document_store=document_store, device=devices[0], label_index=label_index, doc_index=doc_index)
    # Evaluation of Reader can also be done directly on a SQuAD-formatted file without passing the data to Elasticsearch
    #reader_eval_results = reader.eval_on_file("../data/nq", "nq_dev_subset_v2.json", device=device)

    ## Reader Top-N-Accuracy is the proportion of predicted answers that match with their corresponding correct answer
    print("Reader Top-N-Accuracy:", reader_eval_results["top_n_accuracy"])
    ## Reader Exact Match is the proportion of questions where the predicted answer is exactly the same as the correct answer
    print("Reader Exact Match:", reader_eval_results["EM"])
    ## Reader F1-Score is the average overlap between the predicted answers and the correct answers
    print("Reader F1-Score:", reader_eval_results["f1"])


if __name__ == "__main__":
    tutorial5_evaluation()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/