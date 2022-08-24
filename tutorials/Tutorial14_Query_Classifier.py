import logging

# We configure how logging messages should be displayed and which log level should be used before importing Haystack.
# Example log message:
# INFO - haystack.utils.preprocessing -  Converting data/tutorial1/218_Olenna_Tyrell.txt
# Default log level in basicConfig is WARNING so the explicit parameter is not necessary but can be changed easily:
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.utils import (
    fetch_archive_from_http,
    convert_files_to_docs,
    clean_wiki_text,
    launch_es,
    print_answers,
    print_documents,
)
from haystack.pipelines import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import (
    BM25Retriever,
    EmbeddingRetriever,
    FARMReader,
    TransformersQueryClassifier,
    SklearnQueryClassifier,
)
import pandas as pd


def tutorial14_query_classifier():
    """Tutorial 14: Query Classifiers"""

    # Useful for framing headers
    def print_header(header):
        equal_line = "=" * len(header)
        print(f"\n{equal_line}\n{header}\n{equal_line}\n")

    # Try out the SklearnQueryClassifier on its own
    # Keyword vs. Question/Statement Classification
    keyword_classifier = SklearnQueryClassifier()
    queries = [
        "Arya Stark father",  # Keyword Query
        "Who was the father of Arya Stark",  # Interrogative Query
        "Lord Eddard was the father of Arya Stark",  # Statement Query
    ]
    k_vs_qs_results = {"Query": [], "Output Branch": [], "Class": []}
    for query in queries:
        result = keyword_classifier.run(query=query)
        k_vs_qs_results["Query"].append(query)
        k_vs_qs_results["Output Branch"].append(result[1])
        k_vs_qs_results["Class"].append("Question/Statement" if result[1] == "output_1" else "Keyword")
    print_header("Keyword vs. Question/Statement Classification")
    print(pd.DataFrame.from_dict(k_vs_qs_results))
    print("")

    # Question vs. Statement Classification
    model_url = (
        "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/model.pickle"
    )
    vectorizer_url = "https://ext-models-haystack.s3.eu-central-1.amazonaws.com/gradboost_query_classifier_statements/vectorizer.pickle"
    question_classifier = SklearnQueryClassifier(model_name_or_path=model_url, vectorizer_name_or_path=vectorizer_url)
    queries = [
        "Who was the father of Arya Stark",  # Interrogative Query
        "Lord Eddard was the father of Arya Stark",  # Statement Query
    ]
    q_vs_s_results = {"Query": [], "Output Branch": [], "Class": []}
    for query in queries:
        result = question_classifier.run(query=query)
        q_vs_s_results["Query"].append(query)
        q_vs_s_results["Output Branch"].append(result[1])
        q_vs_s_results["Class"].append("Question" if result[1] == "output_1" else "Statement")
    print_header("Question vs. Statement Classification")
    print(pd.DataFrame.from_dict(q_vs_s_results))
    print("")

    # Use in pipelines
    # Download and prepare data - 517 Wikipedia articles for Game of Thrones
    doc_dir = "data/tutorial14"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt14.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Initialize DocumentStore and index documents
    launch_es()
    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()
    document_store.write_documents(got_docs)

    # Pipelines with Keyword vs. Question/Statement Classification
    print_header("PIPELINES WITH KEYWORD VS. QUESTION/STATEMENT CLASSIFICATION")

    # Initialize sparse retriever for keyword queries
    bm25_retriever = BM25Retriever(document_store=document_store)

    # Initialize dense retriever for question/statement queries
    embedding_retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    # Pipeline 1: SklearnQueryClassifier
    print_header("Pipeline 1: SklearnQueryClassifier")
    sklearn_keyword_classifier = Pipeline()
    sklearn_keyword_classifier.add_node(component=SklearnQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    sklearn_keyword_classifier.add_node(
        component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_1"]
    )
    sklearn_keyword_classifier.add_node(
        component=bm25_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"]
    )
    sklearn_keyword_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "EmbeddingRetriever"])
    sklearn_keyword_classifier.draw("sklearn_keyword_classifier.png")

    # Run only the dense retriever on the full sentence query
    res_1 = sklearn_keyword_classifier.run(query="Who is the father of Arya Stark?")
    print_header("Question Query Results")
    print_answers(res_1, details="minimum")
    print("")

    # Run only the sparse retriever on a keyword based query
    res_2 = sklearn_keyword_classifier.run(query="arya stark father")
    print_header("Keyword Query Results")
    print_answers(res_2, details="minimum")
    print("")

    # Pipeline 2: TransformersQueryClassifier
    print_header("Pipeline 2: TransformersQueryClassifier")

    transformer_keyword_classifier = Pipeline()
    transformer_keyword_classifier.add_node(
        component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"]
    )
    transformer_keyword_classifier.add_node(
        component=embedding_retriever, name="EmbeddingRetriever", inputs=["QueryClassifier.output_1"]
    )
    transformer_keyword_classifier.add_node(
        component=bm25_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"]
    )
    transformer_keyword_classifier.add_node(
        component=reader, name="QAReader", inputs=["ESRetriever", "EmbeddingRetriever"]
    )

    # Run only the dense retriever on the full sentence query
    res_1 = transformer_keyword_classifier.run(query="Who is the father of Arya Stark?")
    print_header("Question Query Results")
    print_answers(res_1, details="minimum")
    print("")

    # Run only the sparse retriever on a keyword based query
    res_2 = transformer_keyword_classifier.run(query="arya stark father")
    print_header("Keyword Query Results")
    print_answers(res_2, details="minimum")
    print("")

    # Pipeline with Question vs. Statement Classification
    print_header("PIPELINE WITH QUESTION VS. STATEMENT CLASSIFICATION")
    transformer_question_classifier = Pipeline()
    transformer_question_classifier.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
    transformer_question_classifier.add_node(
        component=TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier"),
        name="QueryClassifier",
        inputs=["EmbeddingRetriever"],
    )
    transformer_question_classifier.add_node(component=reader, name="QAReader", inputs=["QueryClassifier.output_1"])
    transformer_question_classifier.draw("transformer_question_classifier.png")

    # Run only the QA reader on the question query
    res_1 = transformer_question_classifier.run(query="Who is the father of Arya Stark?")
    print_header("Question Query Results")
    print_answers(res_1, details="minimum")
    print("")

    res_2 = transformer_question_classifier.run(query="Arya Stark was the daughter of a Lord.")
    print_header("Statement Query Results")
    print_documents(res_2)
    print("")

    # Other use cases for Query Classifiers

    # Custom classification models

    # Remember to compile a list with the exact model labels
    # The first label you provide corresponds to output_1, the second label to output_2, and so on.
    labels = ["LABEL_0", "LABEL_1", "LABEL_2"]

    sentiment_query_classifier = TransformersQueryClassifier(
        model_name_or_path="cardiffnlp/twitter-roberta-base-sentiment",
        use_gpu=True,
        task="text-classification",
        labels=labels,
    )

    queries = [
        "What's the answer?",  # neutral query
        "Would you be so lovely to tell me the answer?",  # positive query
        "Can you give me the damn right answer for once??",  # negative query
    ]

    sent_results = {"Query": [], "Output Branch": [], "Class": []}

    for query in queries:
        result = sentiment_query_classifier.run(query=query)
        sent_results["Query"].append(query)
        sent_results["Output Branch"].append(result[1])
        if result[1] == "output_1":
            sent_results["Class"].append("negative")
        elif result[1] == "output_2":
            sent_results["Class"].append("neutral")
        elif result[1] == "output_3":
            sent_results["Class"].append("positive")

    print_header("Query Sentiment Classification with custom transformer model")
    print(pd.DataFrame.from_dict(sent_results))
    print("")

    # Zero-shot classification

    # In zero-shot-classification, you can choose the labels
    labels = ["music", "cinema"]

    query_classifier = TransformersQueryClassifier(
        model_name_or_path="typeform/distilbert-base-uncased-mnli",
        use_gpu=True,
        task="zero-shot-classification",
        labels=labels,
    )

    queries = [
        "In which films does John Travolta appear?",  # query about cinema
        "What is the Rolling Stones first album?",  # query about music
        "Who was Sergio Leone?",  # query about cinema
    ]

    query_classification_results = {"Query": [], "Output Branch": [], "Class": []}

    for query in queries:
        result = query_classifier.run(query=query)
        query_classification_results["Query"].append(query)
        query_classification_results["Output Branch"].append(result[1])
        query_classification_results["Class"].append("music" if result[1] == "output_1" else "cinema")

    print_header("Query Zero-shot Classification")
    print(pd.DataFrame.from_dict(query_classification_results))
    print("")


if __name__ == "__main__":
    tutorial14_query_classifier()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
