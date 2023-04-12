import pytest

from haystack.nodes import TextConverter
from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.nodes.reader import FARMReader
from haystack.pipelines import Pipeline
from haystack import Document

from haystack.nodes.extractor import EntityExtractor, simplify_ner_for_qa


@pytest.fixture
def tiny_reader():
    return FARMReader(model_name_or_path="deepset/tinyroberta-squad2", num_processes=0)


@pytest.fixture
def ner_node():
    return EntityExtractor(model_name_or_path="elastic/distilbert-base-cased-finetuned-conll03-english")


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractor(document_store_with_docs, tiny_reader, ner_node):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner_node, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=tiny_reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    entities = [entity["word"] for entity in prediction["answers"][0].meta["entities"]]
    assert "Carla" in entities
    assert "Berlin" in entities


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractor_batch_single_query(document_store_with_docs, tiny_reader, ner_node):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner_node, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=tiny_reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run_batch(
        queries=["Who lives in Berlin?"], params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    entities = [entity["word"] for entity in prediction["answers"][0][0].meta["entities"]]
    assert "Carla" in entities
    assert "Berlin" in entities


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractor_batch_multiple_queries(document_store_with_docs, tiny_reader, ner_node):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner_node, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=tiny_reader, name="Reader", inputs=["NER"])

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


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_extractor_output_simplifier(document_store_with_docs, tiny_reader, ner_node):
    es_retriever = BM25Retriever(document_store=document_store_with_docs)

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner_node, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=tiny_reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )
    simplified = simplify_ner_for_qa(prediction)
    assert simplified[0] == {"answer": "Carla and I", "entities": ["Carla"]}


@pytest.mark.integration
@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_extractor_indexing(document_store, samples_path):
    doc_path = samples_path / "docs" / "doc_2.txt"

    text_converter = TextConverter()
    ner = EntityExtractor(
        model_name_or_path="elastic/distilbert-base-cased-finetuned-conll03-english", flatten_entities_in_meta_data=True
    )

    pipeline = Pipeline()
    pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    pipeline.add_node(component=ner, name="NER", inputs=["TextConverter"])
    pipeline.add_node(component=document_store, name="DocumentStore", inputs=["NER"])
    _ = pipeline.run(file_paths=doc_path)
    docs = document_store.get_all_documents()
    meta = docs[0].meta
    assert "ORG" in meta["entity_groups"]
    assert "Haystack" in meta["entity_words"]


@pytest.mark.integration
def test_extractor_doc_query(ner_node):
    pipeline = Pipeline()
    pipeline.add_node(component=ner_node, name="NER", inputs=["Query"])

    prediction = pipeline.run(query=None, documents=[Document(content="Carla lives in Berlin", content_type="text")])
    entities = [x["word"] for x in prediction["documents"][0].meta["entities"]]
    assert "Carla" in entities
    assert "Berlin" in entities


@pytest.mark.integration
def test_extract_method():
    ner = EntityExtractor(
        model_name_or_path="Jean-Baptiste/camembert-ner", max_seq_len=12, aggregation_strategy="first"
    )

    text = "Hello my name is Arya. I live in Winterfell and my brother is Jon Snow."
    output = ner.extract(text)
    for x in output:
        x.pop("score")
    assert output == [
        {"entity_group": "PER", "word": "Arya.", "start": 16, "end": 22},
        {"entity_group": "LOC", "word": "Winterfell", "start": 32, "end": 43},
        {"entity_group": "PER", "word": "Jon Snow.", "start": 61, "end": 71},
    ]

    text_batch = [text for _ in range(3)]
    batch_size = 2
    output = ner.extract_batch(text_batch, batch_size=batch_size)
    for item in output:
        for x in item:
            x.pop("score")
    for item in output:
        assert item == [
            {"entity_group": "PER", "word": "Arya.", "start": 16, "end": 22},
            {"entity_group": "LOC", "word": "Winterfell", "start": 32, "end": 43},
            {"entity_group": "PER", "word": "Jon Snow.", "start": 61, "end": 71},
        ]


@pytest.mark.integration
def test_extract_method_pre_split_text():
    ner = EntityExtractor(
        model_name_or_path="elastic/distilbert-base-cased-finetuned-conll03-english", max_seq_len=6, pre_split_text=True
    )

    text = "Hello my name is Arya. I live in Winterfell and my brother is Jon Snow."
    output = ner.extract(text)
    for x in output:
        x.pop("score")
    assert output == [
        {"entity_group": "PER", "word": "Arya.", "start": 17, "end": 22},
        {"entity_group": "LOC", "word": "Winterfell", "start": 33, "end": 43},
        {"entity_group": "PER", "word": "Jon Snow.", "start": 62, "end": 71},
    ]

    text_batch = [text for _ in range(3)]
    batch_size = 2
    output = ner.extract_batch(text_batch, batch_size=batch_size)
    for item in output:
        for x in item:
            x.pop("score")
    for item in output:
        assert item == [
            {"entity_group": "PER", "word": "Arya.", "start": 17, "end": 22},
            {"entity_group": "LOC", "word": "Winterfell", "start": 33, "end": 43},
            {"entity_group": "PER", "word": "Jon Snow.", "start": 62, "end": 71},
        ]


@pytest.mark.integration
def test_extract_method_unknown_token():
    ner = EntityExtractor(
        model_name_or_path="elastic/distilbert-base-cased-finetuned-conll03-english",
        max_seq_len=6,
        pre_split_text=True,
        ignore_labels=[],
    )

    text = "Hi my name is JamesÐ."
    output = ner.extract(text)
    for x in output:
        x.pop("score")
    assert output == [{"entity_group": "O", "word": "Hi my name is JamesÐ.", "start": 0, "end": 21}]

    # Different statement in word detection for unknown tokens used when pre_split_text=False
    ner = EntityExtractor(
        model_name_or_path="elastic/distilbert-base-cased-finetuned-conll03-english",
        max_seq_len=6,
        pre_split_text=False,
        ignore_labels=[],
    )

    text = "Hi my name is JamesÐ."
    output = ner.extract(text)
    for x in output:
        x.pop("score")
    assert output == [{"entity_group": "O", "word": "Hi my name is JamesÐ.", "start": 0, "end": 21}]


@pytest.mark.integration
def test_extract_method_simple_aggregation():
    ner = EntityExtractor(
        model_name_or_path="elastic/distilbert-base-cased-finetuned-conll03-english",
        max_seq_len=6,
        aggregation_strategy="simple",
    )

    text = "I live in Berlin with my wife Debra."
    output = ner.extract(text)
    for x in output:
        x.pop("score")
    assert output == [
        {"entity_group": "LOC", "word": "Berlin", "start": 10, "end": 16},
        {"entity_group": "PER", "word": "De", "start": 30, "end": 32},
        {"entity_group": "LOC", "word": "##bra", "start": 32, "end": 35},
    ]
