import os
from subprocess import Popen, PIPE, STDOUT
from haystack.utils import fetch_archive_from_http, convert_files_to_dicts, clean_wiki_text, launch_es, print_answers
from haystack.pipelines import Pipeline, RootNode
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever, DensePassageRetriever, FARMReader, TransformersQueryClassifier, SklearnQueryClassifier


def tutorial14_query_classifier():

    #Download and prepare data - 517 Wikipedia articles for Game of Thrones
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # convert files to dicts containing documents that can be indexed to our datastore
    got_dicts = convert_files_to_dicts(
        dir_path=doc_dir,
        clean_func=clean_wiki_text,
        split_paragraphs=True
    )

    # Initialize DocumentStore and index documents
    launch_es()
    document_store = ElasticsearchDocumentStore()
    document_store.delete_documents()
    document_store.write_documents(got_dicts)

    # Initialize Sparse retriever
    es_retriever = ElasticsearchRetriever(document_store=document_store)

    # Initialize dense retriever
    dpr_retriever = DensePassageRetriever(document_store)
    document_store.update_embeddings(dpr_retriever, update_existing_embeddings=False)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    print()
    print("Sklearn keyword classifier")
    print("==========================")
    # Here we build the pipeline
    sklearn_keyword_classifier = Pipeline()
    sklearn_keyword_classifier.add_node(component=SklearnQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    sklearn_keyword_classifier.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    sklearn_keyword_classifier.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
    sklearn_keyword_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "DPRRetriever"])
    sklearn_keyword_classifier.draw("pipeline_classifier.png")

    # Run only the dense retriever on the full sentence query
    res_1 = sklearn_keyword_classifier.run(
        query="Who is the father of Arya Stark?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_1, details="minimum")

    # Run only the sparse retriever on a keyword based query
    res_2 = sklearn_keyword_classifier.run(
        query="arya stark father",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_2, details="minimum")

    # Run only the dense retriever on the full sentence query
    res_3 = sklearn_keyword_classifier.run(
        query="which country was jon snow filmed ?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_3, details="minimum")

    # Run only the sparse retriever on a keyword based query
    res_4 = sklearn_keyword_classifier.run(
        query="jon snow country",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_4, details="minimum")

    # Run only the dense retriever on the full sentence query
    res_5 = sklearn_keyword_classifier.run(
        query="who are the younger brothers of arya stark ?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_5, details="minimum")

    # Run only the sparse retriever on a keyword based query
    res_6 = sklearn_keyword_classifier.run(
        query="arya stark younger brothers",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_6, details="minimum")

    print()
    print("Transformer keyword classifier")
    print("==============================")
    # Here we build the pipeline
    transformer_keyword_classifier = Pipeline()
    transformer_keyword_classifier.add_node(component=TransformersQueryClassifier(), name="QueryClassifier", inputs=["Query"])
    transformer_keyword_classifier.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["QueryClassifier.output_1"])
    transformer_keyword_classifier.add_node(component=es_retriever, name="ESRetriever", inputs=["QueryClassifier.output_2"])
    transformer_keyword_classifier.add_node(component=reader, name="QAReader", inputs=["ESRetriever", "DPRRetriever"])
    transformer_keyword_classifier.draw("pipeline_classifier.png")

    # Run only the dense retriever on the full sentence query
    res_1 = transformer_keyword_classifier.run(
        query="Who is the father of Arya Stark?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_1, details="minimum")

    # Run only the sparse retriever on a keyword based query
    res_2 = transformer_keyword_classifier.run(
        query="arya stark father",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_2, details="minimum")

    # Run only the dense retriever on the full sentence query
    res_3 = transformer_keyword_classifier.run(
        query="which country was jon snow filmed ?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_3, details="minimum")

    # Run only the sparse retriever on a keyword based query
    res_4 = transformer_keyword_classifier.run(
        query="jon snow country",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_4, details="minimum")

    # Run only the dense retriever on the full sentence query
    res_5 = transformer_keyword_classifier.run(
        query="who are the younger brothers of arya stark ?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_5, details="minimum")

    # Run only the sparse retriever on a keyword based query
    res_6 = transformer_keyword_classifier.run(
        query="arya stark younger brothers",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_6, details="minimum")

    print()
    print("Transformer question classifier")
    print("===============================")

    # Here we build the pipeline
    transformer_question_classifier = Pipeline()
    transformer_question_classifier.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["Query"])
    transformer_question_classifier.add_node(component=TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier"), name="QueryClassifier", inputs=["DPRRetriever"])
    transformer_question_classifier.add_node(component=reader, name="QAReader", inputs=["QueryClassifier.output_1"])
    transformer_question_classifier.draw("question_classifier.png")

    # Run only the QA reader on the question query
    res_1 = transformer_question_classifier.run(
        query="Who is the father of Arya Stark?",
    )
    print("\n===============================")
    print("DPR Results" + "\n" + "="*15)
    print_answers(res_1, details="minimum")

    # Show only DPR results
    res_2 = transformer_question_classifier.run(
        query="Arya Stark was the daughter of a Lord.",
    )
    print("\n===============================")
    print("ES Results" + "\n" + "="*15)
    print_answers(res_2, details="minimum")

    # Here we create the keyword vs question/statement query classifier

    queries = ["arya stark father","jon snow country",
               "who is the father of arya stark","which country was jon snow filmed?"]

    keyword_classifier = TransformersQueryClassifier()

    for query in queries:
        result = keyword_classifier.run(query=query)
        if result[1] == "output_1":
            category = "question/statement"
        else:
            category = "keyword"

        print(f"Query: {query}, raw_output: {result}, class: {category}")

    # Here we create the question vs statement query classifier

    queries = ["Lord Eddard was the father of Arya Stark.","Jon Snow was filmed in United Kingdom.",
               "who is the father of arya stark?","Which country was jon snow filmed in?"]

    question_classifier = TransformersQueryClassifier(model_name_or_path="shahrukhx01/question-vs-statement-classifier")

    for query in queries:
        result = question_classifier.run(query=query)
        if result[1] == "output_1":
            category = "question"
        else:
            category = "statement"

        print(f"Query: {query}, raw_output: {result}, class: {category}")

if __name__ == "__main__":
    tutorial14_query_classifier()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
