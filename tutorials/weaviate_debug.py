

import logging
import subprocess
import time

from haystack.document_store import WeaviateDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers, launch_weaviate
from haystack.retriever.dense import DensePassageRetriever

index="Document"

def weaviate_debug():
    logger = logging.getLogger(__name__)
    launch_weaviate()

    document_store = WeaviateDocumentStore(index=index)
    document_store.delete_all_documents(index=index)

    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    document_store.write_documents(dicts, index=index)

    retriever = DensePassageRetriever(document_store=document_store)

    document_store.update_embeddings(retriever)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    from haystack.pipeline import ExtractiveQAPipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    ## Voilà! Ask a question!
    prediction = pipe.run(query="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)


    print_answers(prediction, details="minimal")


if __name__ == "__main__":
    weaviate_debug()