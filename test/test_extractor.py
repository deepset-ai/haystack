import pytest

from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader import FARMReader
from haystack.pipeline import Pipeline

from haystack.extractor import EntityExtractor, simplify_ner_for_qa


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractor(document_store_with_docs):
    
    es_retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    ner = EntityExtractor()
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", 
        params={
            "ESRetriever": {"top_k": 1}, 
            "Reader": {"top_k": 1},
        }
    )
    entities = [entity["word"] for entity in prediction["answers"][0].meta["entities"]]
    assert "Carla" in entities
    assert "Berlin" in entities


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_extractor_output_simplifier(document_store_with_docs):
    
    es_retriever = ElasticsearchRetriever(document_store=document_store_with_docs)
    ner = EntityExtractor()
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

    pipeline = Pipeline()
    pipeline.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    pipeline.add_node(component=ner, name="NER", inputs=["ESRetriever"])
    pipeline.add_node(component=reader, name="Reader", inputs=["NER"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", 
        params={
            "ESRetriever": {"top_k": 1}, 
            "Reader": {"top_k": 1},
        }
    )
    simplified = simplify_ner_for_qa(prediction)
    assert simplified[0] == {
        "answer": "Carla",
        "entities": ["Carla"]
    }