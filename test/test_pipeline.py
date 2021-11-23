from pathlib import Path

import os
import pytest

from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.pipeline import (
    JoinDocuments,
    Pipeline,
    FAQPipeline,
    DocumentSearchPipeline,
    RootNode,
    SklearnQueryClassifier,
    TransformersQueryClassifier,
    MostSimilarDocumentsPipeline,
)
from haystack.pipelines import ExtractiveQAPipeline
from haystack.reader import FARMReader
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.schema import Document

@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_load_and_save_yaml(document_store, tmp_path):
    # test correct load of indexing pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline.yaml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(
        file_paths=Path(__file__).parent/"samples"/"pdf"/"sample_pdf_1.pdf"
    )
    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline.yaml", pipeline_name="query_pipeline"
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert "_debug" not in prediction.keys()

    # test invalid pipeline name
    with pytest.raises(Exception):
        Pipeline.load_from_yaml(
            path=Path(__file__).parent/"samples"/"pipeline"/"test_pipeline.yaml", pipeline_name="invalid"
        )
    # test config export
    pipeline.save_to_yaml(tmp_path / "test.yaml")
    with open(tmp_path / "test.yaml", "r", encoding="utf-8") as stream:
        saved_yaml = stream.read()
    expected_yaml = """
        components:
        - name: ESRetriever
          params:
            document_store: ElasticsearchDocumentStore
          type: ElasticsearchRetriever
        - name: ElasticsearchDocumentStore
          params:
            index: haystack_test
            label_index: haystack_test_label
          type: ElasticsearchDocumentStore
        - name: Reader
          params:
            model_name_or_path: deepset/roberta-base-squad2
            no_ans_boost: -10
          type: FARMReader
        pipelines:
        - name: query
          nodes:
          - inputs:
            - Query
            name: ESRetriever
          - inputs:
            - ESRetriever
            name: Reader
          type: Pipeline
        version: '0.8'
    """
    assert saved_yaml.replace(" ", "").replace("\n", "") == expected_yaml.replace(
        " ", ""
    ).replace("\n", "")

@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_load_and_save_yaml_prebuilt_pipelines(document_store, tmp_path):
    # populating index
    pipeline = Pipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline.yaml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(
        file_paths=Path(__file__).parent/"samples"/"pdf"/"sample_pdf_1.pdf"
    )
    # test correct load of query pipeline from yaml
    pipeline = ExtractiveQAPipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline.yaml", pipeline_name="query_pipeline"
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert "_debug" not in prediction.keys()

    # test invalid pipeline name
    with pytest.raises(Exception):
        ExtractiveQAPipeline.load_from_yaml(
            path=Path(__file__).parent/"samples"/"pipeline"/"test_pipeline.yaml", pipeline_name="invalid"
        )
    # test config export
    pipeline.save_to_yaml(tmp_path / "test.yaml")
    with open(tmp_path / "test.yaml", "r", encoding="utf-8") as stream:
        saved_yaml = stream.read()
    expected_yaml = """
        components:
        - name: ESRetriever
          params:
            document_store: ElasticsearchDocumentStore
          type: ElasticsearchRetriever
        - name: ElasticsearchDocumentStore
          params:
            index: haystack_test
            label_index: haystack_test_label
          type: ElasticsearchDocumentStore
        - name: Reader
          params:
            model_name_or_path: deepset/roberta-base-squad2
            no_ans_boost: -10
          type: FARMReader
        pipelines:
        - name: query
          nodes:
          - inputs:
            - Query
            name: ESRetriever
          - inputs:
            - ESRetriever
            name: Reader
          type: Pipeline
        version: '0.8'
    """
    assert saved_yaml.replace(" ", "").replace("\n", "") == expected_yaml.replace(
        " ", ""
    ).replace("\n", "")

def test_load_tfidfretriever_yaml(tmp_path):
    documents = [
        {
            "content": "A Doc specifically talking about haystack. Haystack can be used to scale QA models to large document collections."
        }
    ]
    pipeline = Pipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline_tfidfretriever.yaml", pipeline_name="query_pipeline"
    )
    with pytest.raises(Exception) as exc_info:
        pipeline.run(
            query="What can be used to scale QA models to large document collections?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}
        )
    exception_raised = str(exc_info.value)
    assert "Retrieval requires dataframe df and tf-idf matrix" in exception_raised

    pipeline.get_node(name="Retriever").document_store.write_documents(documents=documents)
    prediction = pipeline.run(
        query="What can be used to scale QA models to large document collections?",
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "What can be used to scale QA models to large document collections?"
    assert prediction["answers"][0].answer == "haystack"


# @pytest.mark.slow
# @pytest.mark.elasticsearch
# @pytest.mark.parametrize(
#     "retriever_with_docs, document_store_with_docs",
#     [("elasticsearch", "elasticsearch")],
#     indirect=True,
# )
@pytest.mark.parametrize(
    "retriever_with_docs,document_store_with_docs",
    [
        ("dpr", "elasticsearch"),
        ("dpr", "faiss"),
        ("dpr", "memory"),
        ("dpr", "milvus"),
        ("embedding", "elasticsearch"),
        ("embedding", "faiss"),
        ("embedding", "memory"),
        ("embedding", "milvus"),
        ("elasticsearch", "elasticsearch"),
        ("es_filter_only", "elasticsearch"),
        ("tfidf", "memory"),
    ],
    indirect=True,
)
def test_graph_creation(retriever_with_docs, document_store_with_docs):
    pipeline = Pipeline()
    pipeline.add_node(name="ES", component=retriever_with_docs, inputs=["Query"])

    with pytest.raises(AssertionError):
        pipeline.add_node(
            name="Reader", component=retriever_with_docs, inputs=["ES.output_2"]
        )

    with pytest.raises(AssertionError):
        pipeline.add_node(
            name="Reader", component=retriever_with_docs, inputs=["ES.wrong_edge_label"]
        )

    with pytest.raises(Exception):
        pipeline.add_node(
            name="Reader", component=retriever_with_docs, inputs=["InvalidNode"]
        )

    with pytest.raises(Exception):
        pipeline = Pipeline()
        pipeline.add_node(
            name="ES", component=retriever_with_docs, inputs=["InvalidNode"]
        )


def test_parallel_paths_in_pipeline_graph():
    class A(RootNode):
        def run(self):
            test = "A"
            return {"test": test}, "output_1"

    class B(RootNode):
        def run(self, test):
            test += "B"
            return {"test": test}, "output_1"

    class C(RootNode):
        def run(self, test):
            test += "C"
            return {"test": test}, "output_1"

    class D(RootNode):
        def run(self, test):
            test += "D"
            return {"test": test}, "output_1"

    class E(RootNode):
        def run(self, test):
            test += "E"
            return {"test": test}, "output_1"

    class JoinNode(RootNode):
        def run(self, inputs):
            test = (
                inputs[0]["test"] + inputs[1]["test"]
            )
            return {"test": test}, "output_1"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=A(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A"])
    pipeline.add_node(name="C", component=C(), inputs=["B"])
    pipeline.add_node(name="E", component=E(), inputs=["C"])
    pipeline.add_node(name="D", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E"])
    output = pipeline.run(query="test")
    assert output["test"] == "ABDABCE"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=A(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A"])
    pipeline.add_node(name="C", component=C(), inputs=["B"])
    pipeline.add_node(name="D", component=D(), inputs=["B"])
    pipeline.add_node(name="E", component=JoinNode(), inputs=["C", "D"])
    output = pipeline.run(query="test")
    assert output["test"] == "ABCABD"


def test_parallel_paths_in_pipeline_graph_with_branching():
    class AWithOutput1(RootNode):
        outgoing_edges = 2

        def run(self):
            output = "A"
            return {"output": output}, "output_1"

    class AWithOutput2(RootNode):
        outgoing_edges = 2

        def run(self):
            output = "A"
            return {"output": output}, "output_2"

    class AWithOutputAll(RootNode):
        outgoing_edges = 2

        def run(self):
            output = "A"
            return {"output": output}, "output_all"

    class B(RootNode):
        def run(self, output):
            output += "B"
            return {"output": output}, "output_1"

    class C(RootNode):
        def run(self, output):
            output += "C"
            return {"output": output}, "output_1"

    class D(RootNode):
        def run(self, output):
            output += "D"
            return {"output": output}, "output_1"

    class E(RootNode):
        def run(self, output):
            output += "E"
            return {"output": output}, "output_1"

    class JoinNode(RootNode):
        def run(self, output=None, inputs=None):
            if inputs:
                output = ""
                for input_dict in inputs:
                    output += input_dict["output"]
            return {"output": output}, "output_1"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=AWithOutput1(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A.output_1"])
    pipeline.add_node(name="C", component=C(), inputs=["A.output_2"])
    pipeline.add_node(name="D", component=E(), inputs=["B"])
    pipeline.add_node(name="E", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E", "C"])
    output = pipeline.run(query="test")
    assert output["output"] == "ABEABD"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=AWithOutput2(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A.output_1"])
    pipeline.add_node(name="C", component=C(), inputs=["A.output_2"])
    pipeline.add_node(name="D", component=E(), inputs=["B"])
    pipeline.add_node(name="E", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E", "C"])
    output = pipeline.run(query="test")
    assert output["output"] == "AC"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=AWithOutputAll(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A.output_1"])
    pipeline.add_node(name="C", component=C(), inputs=["A.output_2"])
    pipeline.add_node(name="D", component=E(), inputs=["B"])
    pipeline.add_node(name="E", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E", "C"])
    output = pipeline.run(query="test")
    assert output["output"] == "ACABEABD"


def test_existing_faiss_document_store():
    clean_faiss_document_store()

    pipeline = Pipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline_faiss_indexing.yaml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(
        file_paths=Path(__file__).parent/"samples"/"pdf"/"sample_pdf_1.pdf"
    )

    new_document_store = pipeline.get_document_store()
    new_document_store.save('existing_faiss_document_store')

    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        Path(__file__).parent/"samples"/"pipeline"/"test_pipeline_faiss_retrieval.yaml", pipeline_name="query_pipeline"
    )

    retriever = pipeline.get_node("DPRRetriever")
    existing_document_store = retriever.document_store
    faiss_index = existing_document_store.faiss_indexes['document']
    assert faiss_index.ntotal == 2

    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"DPRRetriever": {"top_k": 10}}
    )

    assert prediction["query"] == "Who made the PDF specification?"
    assert len(prediction["documents"]) == 2
    clean_faiss_document_store()


def clean_faiss_document_store():
    if Path('existing_faiss_document_store').exists():
        os.remove('existing_faiss_document_store')
    if Path('existing_faiss_document_store.json').exists():
        os.remove('existing_faiss_document_store.json')
    if Path('faiss_document_store.db').exists():
        os.remove('faiss_document_store.db')
