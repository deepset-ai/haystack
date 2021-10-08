import pytest

from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader import FARMReader
from haystack.pipeline import Pipeline

from haystack.extractor import EntityExtractor


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
            "ESRetriever": {"top_k": 3}, 
            "Reader": {"top_k": 3},
        }
    )
    assert prediction["answers"]["meta"]["entities"]
    print(prediction)


