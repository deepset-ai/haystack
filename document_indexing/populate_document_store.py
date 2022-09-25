import logging

from haystack.document_stores import ElasticsearchDocumentStore
from s3_storage import S3Storage

import sys
import os
# this is horrible until I find a prettier solution
myDir = os.getcwd()
sys.path.append(myDir)
from pathlib import Path
path = Path(myDir)
sys.path.append(str(path.parent.absolute()))

from rest_api.rest_api.schema import QuestionAnswerPair

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
    document_store_q_and_a_pairs = ElasticsearchDocumentStore()

    q_and_a_pairs = S3Storage().load_qa_pairs(game)
    q_and_a_pairs = [question_answer_pair_to_document_store_format(qap) for qap in q_and_a_pairs]
    logger.info(f"Loading {len(q_and_a_pairs)} Q and A pairs")
    document_store_q_and_a_pairs.delete_documents()
    document_store_q_and_a_pairs.write_documents(q_and_a_pairs)


def check_es():
    from elasticsearch import Elasticsearch
    client = Elasticsearch()
    logger.info(f"Elastic search has {client.count()['count']} Q and A pairs")
    client.close()


if __name__ == "__main__":
    populate_document_store("monopoly")
    check_es()
