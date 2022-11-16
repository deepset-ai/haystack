import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers, launch_es
from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from pprint import pprint


def basic_qa_pipeline():
    # launch and create DocumentStore
    launch_es()
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

    # fetch, pre-process and write documents
    doc_dir = "data/basic_qa_pipeline"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
  
    document_store.write_documents(docs)

    # Initialize Retriever & Reader
    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Initialize ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Query the pipleine
    prediction = pipe.run(
        query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", params={"Reader": {"top_k": 5}})
    # prediction = pipe.run(query="Who is the sister of Sansa?", params={"Reader": {"top_k": 5}})

    # Now you can either print the object directly
    print("\n\nRaw object:\n")
    pprint(prediction)

    print("\n\nSimplified output:\n")
    print_answers(prediction, details="minimum")


if __name__ == "__main__":
    basic_qa_pipeline()