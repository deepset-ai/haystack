import os
import sys
import logging

from haystack.nodes import (
    TextConverter,
    PDFToTextConverter,
    DocxToTextConverter,
     PreProcessor
 )
from haystack.document_stores import ElasticsearchDocumentStore
from s3_storage import S3Storage

# this is horrible until I find a prettier solution
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)/'rest_api'
sys.path.append(str(path))

from rest_api.schema import QuestionAnswerPair


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger("elasticsearch").setLevel(logging.INFO)


def question_answer_pair_to_document_store_format(q_and_a_pair: QuestionAnswerPair):
    """
    Haystack document stores want a dict in the format of {"content": str, "meta": dict}
    """
    return {
        "content": q_and_a_pair.question,

        "meta": {
            "game": q_and_a_pair.game,
            "answer": q_and_a_pair.answer,
            "validated": q_and_a_pair.approved
        }
    }


def populate_document_store(game: str):
    s3_storage = S3Storage()

    #Extraction part
    rulebook_file_path = s3_storage.load_rulebook_path(game)
    extractive_document_store = ElasticsearchDocumentStore(index='rulebook', embedding_dim=768)
    extractive_document_store.delete_documents(index='rulebook')
    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    doc_pdf = converter.convert(file_path=rulebook_file_path, meta=None)[0]

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=False,
    )
    docs_default = preprocessor.process([doc_pdf])
    extractive_document_store.write_documents(docs_default, index='rulebook')

    #FAQ part
    q_and_a_pairs = s3_storage.load_qa_pairs(game)
    q_and_a_pairs = [question_answer_pair_to_document_store_format(qap) for qap in q_and_a_pairs]
    logger.info(f"Loading {len(q_and_a_pairs)} Q and A pairs")
    faq_document_store = ElasticsearchDocumentStore(index='faq', embedding_dim=384, similarity='cosine')
    faq_document_store.delete_documents(index='faq')
    faq_document_store.write_documents(q_and_a_pairs, index='faq')


def check_es():
    from elasticsearch import Elasticsearch
    client = Elasticsearch()
    logger.info(f"Elastic search has {client.count()['count']} Q and A pairs")
    client.close()


if __name__ == "__main__":
    populate_document_store("monopoly")
    check_es()
