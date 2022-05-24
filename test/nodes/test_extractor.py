import pytest

from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.nodes.reader import FARMReader
from haystack.pipelines import Pipeline

from haystack.nodes.extractor import EntityExtractor, simplify_ner_for_qa


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
