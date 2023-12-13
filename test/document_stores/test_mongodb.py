import contextlib
import os
import re
import requests
import pytest
import numpy
from haystack.document_stores.mongodb_filters import _target_filter_to_metadata, _and_or_to_list, mongo_filter_converter
from haystack.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore, pymongo
from haystack.schema import Document
from haystack.nodes import PreProcessor, EmbeddingRetriever

pytestmark = pytest.mark.integration

mongo_atlas_database = "database01"
mongo_atlas_collection = "test_80_days"

mongo_atlas_username = os.getenv("MONGO_ATLAS_USERNAME")
mongo_atlas_password = os.getenv("MONGO_ATLAS_PASSWORD")
mongo_atlas_host = os.getenv("MONGO_ATLAS_HOST")
mongo_atlas_connection_params = {"retryWrites": "true", "w": "majority"}
mongo_atlas_params_string = "&".join([f"{key}={value}" for key, value in mongo_atlas_connection_params.items()])
mongo_atlas_connection_string = (
    f"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"
)

document_store = MongoDBAtlasDocumentStore(
    mongo_connection_string=mongo_atlas_connection_string,
    database_name=mongo_atlas_database,
    collection_name=mongo_atlas_collection,
    embedding_dim=768,
)


def roman_to_int(s: str) -> int:
    mapping = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    result = 0

    for i in range(len(s)):
        if i < len(s) - 1 and mapping[s[i]] < mapping[s[i + 1]]:
            result -= mapping[s[i]]
        else:
            result += mapping[s[i]]

    return result


# Get the book "Around the World in 80 Days" from Project Gutenberg
def get_book_online():
    response = requests.get("https://www.gutenberg.org/ebooks/103.txt.utf-8")
    if response.status_code != 200:
        raise requests.HTTPError(f"HTTP error {response.status_code}")
    else:
        return response.text


def get_book_local():
    with open("test-data/80_days.txt", "r", encoding="utf-8") as file:
        text = file.read()
    return text


get_book = get_book_online


# Divide the book into chapters
def divide_book_into_chapters(book) -> dict:
    lines = book.split("\n")
    current_chapter = None
    chapters = {}
    for line in lines:
        chapter_match = re.match(r"CHAPTER\s+([IVXLCDM]+)\.*", line)
        if chapter_match:
            chapter_roman = chapter_match.group(1)
            chapter_decimal = roman_to_int(chapter_roman)
            current_chapter = f"CHAPTER {chapter_decimal}".title()
            chapters[current_chapter] = ""
        if current_chapter:
            chapters[current_chapter] += line + "\n"
    return chapters


book = get_book()
chapters = divide_book_into_chapters(book)
documents = [
    Document(content=chapters[f"Chapter {n}"], meta={"book": "Around the World in 80 Days", "Chapter": n})
    for n in range(1, len(chapters) + 1)
]


def test_write_documents_skip():
    document_store.delete_documents()

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000,
    )

    processed_documents = processor.process([documents[0]])
    document_store.write_documents(processed_documents)

    collection = document_store._get_collection()

    filters = {"Chapter": 1, "_split_id": 0}
    collection.update_one(mongo_filter_converter(filters), {"$set": {"content": "No Content"}})
    document_store.write_documents(processed_documents, duplicate_documents="skip")
    assert document_store.get_all_documents(filters=filters)[0].content == "No Content"


def test_write_documents_overwrite():
    document_store.delete_documents()

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000,
    )

    processed_documents = processor.process([documents[0]])
    document_store.write_documents(processed_documents)

    collection = document_store._get_collection()

    filters = {"Chapter": 1, "_split_id": 0}
    collection.update_one(mongo_filter_converter(filters), {"$set": {"content": "No Content"}})
    document_store.write_documents(processed_documents, duplicate_documents="overwrite")
    assert document_store.get_all_documents(filters=filters)[0].content != "No Content"


def test_write_documents_fail():
    document_store.delete_documents()

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000,
    )

    processed_documents = processor.process([documents[0]])
    document_store.write_documents(processed_documents)
    document_store.write_documents(processed_documents)
    with pytest.raises(pymongo.errors.BulkWriteError):
        document_store.write_documents(processed_documents, duplicate_documents="fail")


def test_write_documents():
    document_store.delete_documents()

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000,
    )

    processed_documents = processor.process(documents)
    document_store.write_documents(processed_documents)

    assert document_store.get_document_count() == 373
    assert document_store.get_all_documents(return_embedding=True)[0].embedding is None


def test_get_document_count_without_embeddings_a():
    assert document_store.get_document_count(only_documents_without_embedding=True) == 373


def test_get_embedding_count_a():
    assert document_store.get_embedding_count() == 0


def test_get_document_count_without_embeddings_with_filter():
    assert document_store.get_document_count(filters={"Chapter": 1}, only_documents_without_embedding=True) == 8


def test_update_embeddings_filtered():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )
    filters = {"Chapter": 1, "_split_id": 0}
    document_store.update_embeddings(retriever, batch_size=30, filters=filters)
    assert isinstance(
        document_store.get_all_documents(return_embedding=True, filters=filters)[0].embedding, numpy.ndarray
    )


def test_update_embeddings():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )

    document_store.update_embeddings(retriever, batch_size=30)
    assert isinstance(document_store.get_all_documents(return_embedding=True)[0].embedding, numpy.ndarray)


def test_update_embeddings_not_existing():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )
    filters = {"Chapter": 1, "_split_id": 0}
    filters2 = {"Chapter": 1, "_split_id": 1}
    collection = document_store._get_collection()

    collection.update_one(mongo_filter_converter(filters), {"$set": {"embedding": None}})
    collection.update_one(mongo_filter_converter(filters2), {"$set": {"embedding": "not_an_embedding"}})

    document_store.update_embeddings(retriever, batch_size=30, update_existing_embeddings=False)
    assert isinstance(collection.find_one(mongo_filter_converter(filters))["embedding"], list)
    assert collection.find_one(mongo_filter_converter(filters2))["embedding"] == "not_an_embedding"


def test_get_embedding_count_b():
    assert document_store.get_embedding_count() == 373


# Getting documents


def test_get_all_documents_without_embedings():
    assert document_store.get_all_documents()[0].embedding is None
    assert document_store.get_all_documents(return_embedding=False)[0].embedding is None


def test_get_all_documents_with_embedings():
    assert isinstance(document_store.get_all_documents(return_embedding=True)[0].embedding, numpy.ndarray)


def test_get_all_documents():
    assert len(document_store.get_all_documents()) == 373


def test_get_all_documents_filtered():
    assert len(document_store.get_all_documents(filters={"Chapter": 1})) == 8


def test_get_document_by_id_a():
    documents = document_store.get_all_documents(filters={"Chapter": 1, "_split_id": 0})
    assert len(documents) == 1
    document_id = documents[0].id
    assert isinstance(document_store.get_document_by_id(id=document_id), Document)


def test_get_documents_by_id_b():
    documents = document_store.get_all_documents(filters={"Chapter": 1})
    document_ids = [document.id for document in documents]
    assert len(document_ids) > 1
    assert len(document_store.get_documents_by_id(ids=document_ids)) == len(document_ids)


def test_get_all_documents_headers_throws():
    with pytest.raises(NotImplementedError):
        document_store.get_all_documents(headers={"key": "value"})


def test_get_document_count():
    assert document_store.get_document_count() == 373


def test_get_document_count_without_embeddings_b():
    assert document_store.get_document_count(only_documents_without_embedding=True) == 0


def test_get_document_count_filtered():
    assert document_store.get_document_count(filters={"Chapter": 1}) == 8
    assert document_store.get_document_count(filters={"Chapter": 1, "_split_id": 0}) == 1


# Updating document meta


def test_update_document_meta():
    document = document_store.get_all_documents(filters={"Chapter": 1, "_split_id": 0})[0]
    new_meta = document.meta
    new_meta["new_field"] = "New metadata"
    document_store.update_document_meta(id=document.id, meta=new_meta)
    updated_document = document_store.get_all_documents(filters={"Chapter": 1, "_split_id": 0})[0]
    assert "new_field" in updated_document.meta
    assert updated_document.meta["new_field"] == "New metadata"


# Deleting documents


def test_delete_documents_filtered():
    document_store.delete_documents(filters={"Chapter": 1, "_split_id": 0})
    assert document_store.get_document_count() == 372


def test_delete_documents_by_id():
    documents = document_store.get_all_documents(filters={"Chapter": 1})
    document_ids = [document.id for document in documents]
    document_store.delete_documents(ids=document_ids)
    assert document_store.get_document_count() == 365


def test_delete_documents_by_id_filtered():
    documents = document_store.get_all_documents(filters={"Chapter": 2})
    document_ids = [document.id for document in documents]
    document_store.delete_documents(ids=document_ids, filters={"_split_id": 0})  # Only deletes the intersection
    assert document_store.get_document_count() == 364


def test_delete_documents():
    document_store.delete_documents()
    assert document_store.get_document_count() == 0


def test_delete_documents_headers_throws():
    with pytest.raises(NotImplementedError):
        document_store.delete_documents(headers={"key": "value"})


def test_delete_index():
    document_store.delete_index()
    client = pymongo.MongoClient(mongo_atlas_connection_string)
    database = client[mongo_atlas_database]
    assert "test_80_days" not in database.list_collection_names()


def test_delete_index_with_index():
    client = pymongo.MongoClient(mongo_atlas_connection_string)
    database = client[mongo_atlas_database]
    with contextlib.suppress(Exception):
        database.create_collection("deleteme")
    assert "deleteme" in database.list_collection_names()
    document_store.delete_index(index="deleteme")
    assert "deleteme" not in database.list_collection_names()


def test__create_document_field_map_a():
    assert document_store._create_document_field_map() == {"embedding": "embedding"}


def test__create_document_field_map_b():
    document_store = MongoDBAtlasDocumentStore(
        mongo_connection_string=mongo_atlas_connection_string,
        database_name=mongo_atlas_database,
        collection_name=mongo_atlas_collection,
        embedding_dim=768,
        embedding_field="emb",
    )
    assert document_store._create_document_field_map() == {"emb": "embedding"}


def test__get_collection_no_index():
    collection = document_store._get_collection()
    assert collection.name == "test_80_days"


def test__get_collection_with_index():
    collection = document_store._get_collection(index="index_abcdefg")
    assert collection.name == "index_abcdefg"


def test__get_collection_invalid_index():
    with pytest.raises(ValueError):
        document_store._get_collection(index="index_a!!bcdefg")


def test_write_documents_index():
    document_store.delete_documents()

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000,
    )

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )

    processed_documents = processor.process(documents)
    document_store.write_documents(processed_documents)

    assert document_store.get_document_count() == 373
    assert document_store.get_all_documents(return_embedding=True)[0].embedding is None

    document_store.update_embeddings(retriever, batch_size=30)
    assert isinstance(document_store.get_all_documents(return_embedding=True)[0].embedding, numpy.ndarray)


def test_query_by_embedding_default_topk():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
    )
    embedding = retriever.embed_queries(["How much money was stolen from the bank?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding)
    assert results[0].embedding is None
    assert len(results) == 10


def test_query_by_embedding_default_topk_4():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
    )
    embedding = retriever.embed_queries(["How much money was stolen from the bank?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding, top_k=4)
    assert results[0].embedding == None
    assert len(results) == 4


def test_query_by_embedding_filtered():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )
    embedding = retriever.embed_queries(["Who was Phileas Fogg?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding, filters={"Chapter": 1})
    assert len(results) == 3


def test_query_by_embedding_include_embedding():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )
    embedding = retriever.embed_queries(["How much money was stolen from the bank?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding, return_embedding=True)
    assert isinstance(results[0].embedding, numpy.ndarray)
    assert len(results) == 10


def test_and_or_meta_converted():
    test_filter = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {"$or": [{"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"_split_id": 0}]},
        ]
    }

    target_outcome = {
        "$and": [
            {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"meta._split_id": 0},
            {"$or": [{"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"meta._split_id": 0}]},
        ]
    }
    assert _and_or_to_list(_target_filter_to_metadata(test_filter, "meta")) == target_outcome


def test_mongo_filter_converter_and_or_meta_converted():
    test_filter = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {"$or": [{"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"_split_id": 0}]},
        ]
    }

    target_outcome = {
        "$and": [
            {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"meta._split_id": 0},
            {"$or": [{"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"meta._split_id": 0}]},
        ]
    }

    assert mongo_filter_converter(test_filter) == target_outcome


def test_mongo_filter_converter_falsey_empty_dict():
    assert mongo_filter_converter(None) == {}
    assert mongo_filter_converter("") == {}
    assert mongo_filter_converter({}) == {}
    assert mongo_filter_converter([]) == {}
    assert mongo_filter_converter(0) == {}


def test__target_filter_to_metadata_01():
    test_filter = {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes", "_split_id": 0}

    target_outcome = {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes", "meta._split_id": 0}

    assert _target_filter_to_metadata(test_filter, "meta") == target_outcome


def test__target_filter_to_metadata_02():
    test_filter = {"$and": [{"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"_split_id": 0}]}

    target_outcome = {"$and": [{"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"meta._split_id": 0}]}

    assert _target_filter_to_metadata(test_filter, "meta") == target_outcome


def test__target_filter_to_metadata_leave_id():
    test_filter = {"id": {"$in": ["b714102aa7ac3a9622d0d00caa55fa", "b3de1a673c1eb2876585405395a10c3d"]}}

    target_outcome = {"id": {"$in": ["b714102aa7ac3a9622d0d00caa55fa", "b3de1a673c1eb2876585405395a10c3d"]}}

    assert _target_filter_to_metadata(test_filter, "meta") == target_outcome


def test__and_or_to_list_01():
    test_filter = {"$and": {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes", "_split_id": 0}}

    target_outcome = {"$and": [{"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"_split_id": 0}]}

    assert _and_or_to_list(test_filter) == target_outcome


def test__and_or_to_list_02():
    test_filter = {
        "$and": {
            "url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes",
            "_split_id": 0,
            "$or": {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes", "_split_id": 0},
        }
    }

    target_outcome = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {"$or": [{"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"_split_id": 0}]},
        ]
    }

    assert _and_or_to_list(test_filter) == target_outcome
