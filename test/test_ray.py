from pathlib import Path

import pytest
from haystack.pipeline import RayPipeline


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_load_pipeline(document_store_with_docs):
    pipeline = RayPipeline.load_from_yaml(
        Path("samples/pipeline/test_pipeline.yaml"), pipeline_name="query_pipeline", num_cpus=8,
    )
    prediction = pipeline.run(query="Who lives in Berlin?", top_k_retriever=10, top_k_reader=3)
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0]["answer"] == "Carla"
