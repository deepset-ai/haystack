import pytest

from haystack.nodes import TextConverter
from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.nodes.reader import FARMReader
from haystack.pipelines import Pipeline

from haystack.nodes.extractor import EntityExtractor, simplify_ner_for_qa

from ..conftest import SAMPLES_PATH


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractor(document_store_with_docs):

    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    ner = EntityExtractor()
    reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    entities = [entity["word"] for entity in prediction["answers"][0].meta["entities"]]
    assert "Carla" in entities
    assert "Berlin" in entities


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractor_batch_single_query(document_store_with_docs):

    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    ner = EntityExtractor()
    reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run_batch(
        queries=["Who lives in Berlin?"], params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    entities = [entity["word"] for entity in prediction["answers"][0][0].meta["entities"]]
    assert "Carla" in entities
    assert "Berlin" in entities


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractor_batch_multiple_queries(document_store_with_docs):

    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    ner = EntityExtractor()
    reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run_batch(
        queries=["Who lives in Berlin?", "Who lives in New York?"],
        params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}},
    )
    entities_carla = [entity["word"] for entity in prediction["answers"][0][0].meta["entities"]]
    entities_paul = [entity["word"] for entity in prediction["answers"][1][0].meta["entities"]]
    assert "Carla" in entities_carla
    assert "Berlin" in entities_carla
    assert "Paul" in entities_paul
    assert "New York" in entities_paul


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractor_output_simplifier(document_store_with_docs):

    es_retriever = BM25Retriever(document_store=document_store_with_docs)
    ner = EntityExtractor()
    reader = FARMReader(model_name_or_path="deepset/tinyroberta-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    simplified = simplify_ner_for_qa(prediction)
    assert simplified[0] == {"answer": "Carla and I", "entities": ["Carla"]}


@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_extractor_indexing(document_store):
    doc_path = SAMPLES_PATH / "docs" / "doc_2.txt"

    text_converter = TextConverter()
    ner = EntityExtractor(flatten_entities_in_meta_data=True)

    pipeline = Pipeline()
    pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    pipeline.add_node(component=ner, name="NER", inputs=["TextConverter"])
    pipeline.add_node(component=document_store, name="DocumentStore", inputs=["NER"])
    _ = pipeline.run(file_paths=doc_path)
    docs = document_store.get_all_documents()
    meta = docs[0].meta
    assert "ORG" in meta["entity_groups"]
    assert "Haystack" in meta["entity_words"]


def test_extract_method():
    ner = EntityExtractor(max_seq_len=6)

    text = "Hello my name is Arya. I live in Winterfell and my brother is Jon Snow."
    output = ner.extract(text)
    for x in output:
        x.pop("score")
    assert output == [
        {"entity_group": "PER", "word": "Arya", "start": 17, "end": 21},
        {"entity_group": "LOC", "word": "Winterfell", "start": 33, "end": 43},
        {"entity_group": "PER", "word": "Jon Snow", "start": 62, "end": 70},
    ]

    text_batch = [text for _ in range(3)]
    batch_size = 2
    output = ner.extract_batch(text_batch, batch_size=batch_size)
    for item in output:
        for x in item:
            x.pop("score")
    assert output == [
        [
            {"entity_group": "PER", "word": "Arya", "start": 17, "end": 21},
            {"entity_group": "LOC", "word": "Winterfell", "start": 33, "end": 43},
            {"entity_group": "PER", "word": "Jon Snow", "start": 62, "end": 70},
        ],
        [
            {"entity_group": "PER", "word": "Arya", "start": 17, "end": 21},
            {"entity_group": "LOC", "word": "Winterfell", "start": 33, "end": 43},
            {"entity_group": "PER", "word": "Jon Snow", "start": 62, "end": 70},
        ],
        [
            {"entity_group": "PER", "word": "Arya", "start": 17, "end": 21},
            {"entity_group": "LOC", "word": "Winterfell", "start": 33, "end": 43},
            {"entity_group": "PER", "word": "Jon Snow", "start": 62, "end": 70},
        ],
    ]
