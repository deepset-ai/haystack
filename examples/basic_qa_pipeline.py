import logging
from pathlib import Path

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import fetch_archive_from_http, print_answers, launch_es
from haystack.nodes import FARMReader, BM25Retriever
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.file_converter import TextConverter
from haystack.pipelines import Pipeline


def basic_qa_pipeline():
    # Initialize a DocumentStore
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # fetch, pre-process and write documents
    doc_dir = "data/basic_qa_pipeline"
    s3_url = "https://core-engineering.s3.eu-central-1.amazonaws.com/public/scripts/wiki_gameofthrones_txt1.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    file_paths = [p for p in Path(doc_dir).glob("**/*")]
    files_metadata = [{"name": path.name} for path in file_paths]

    # Indexing Pipeline
    indexing_pipeline = Pipeline()

    # Makes sure the file is a TXT file (FileTypeClassifier node)
    classifier = FileTypeClassifier()
    indexing_pipeline.add_node(classifier, name="Classifier", inputs=["File"])

    # Converts a file into text and performs basic cleaning (TextConverter node)
    text_converter = TextConverter(remove_numeric_tables=True)
    indexing_pipeline.add_node(text_converter, name="Text_converter", inputs=["Classifier.output_1"])

    # - Pre-processes the text by performing splits and adding metadata to the text (Preprocessor node)
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_empty_lines=True,
        split_length=100,
        split_overlap=50,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(preprocessor, name="Preprocessor", inputs=["Text_converter"])

    # - Writes the resulting documents into the document store
    indexing_pipeline.add_node(document_store, name="Document_Store", inputs=["Preprocessor"])

    # Then we run it with the documents and their metadata as input
    indexing_pipeline.run(file_paths=file_paths, meta=files_metadata)

    # Initialize Retriever & Reader
    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Query Pipeline
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

    prediction = pipeline.run(
        query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    print_answers(prediction, details="minimum")
    return prediction


if __name__ == "__main__":
    launch_es()
    basic_qa_pipeline()
