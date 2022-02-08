from pathlib import Path

import pytest
import ray

from haystack.pipelines import RayPipeline

from conftest import SAMPLES_PATH


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_load_pipeline(document_store_with_docs):
    pipeline = RayPipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline.yaml",
        pipeline_name="ray_query_pipeline",
        num_cpus=8,
    )
    prediction = pipeline.run(query="Who lives in Berlin?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}})

    assert ray.serve.get_deployment(name="ESRetriever").num_replicas == 2
    assert ray.serve.get_deployment(name="Reader").num_replicas == 1
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"


@pytest.fixture(scope="function", autouse=True)
def shutdown_ray():
    yield
    try:
        import ray

        ray.shutdown()
    except:
        pass
