from pathlib import Path

import pytest
import ray

from haystack.pipelines import RayPipeline

from ..conftest import SAMPLES_PATH


@pytest.fixture(autouse=True)
def shutdown_ray():
    yield
    try:
        import ray

        ray.serve.shutdown()
        ray.shutdown()
    except:
        pass


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
@pytest.mark.parametrize("serve_detached", [True, False])
def test_load_pipeline(document_store_with_docs, serve_detached):
    pipeline = RayPipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "ray.haystack-pipeline.yml",
        pipeline_name="ray_query_pipeline",
        ray_args={"num_cpus": 8},
        serve_args={"detached": serve_detached},
    )
    prediction = pipeline.run(query="Who lives in Berlin?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}})

    assert pipeline._serve_controller_client._detached == serve_detached
    assert ray.serve.get_deployment(name="ESRetriever").num_replicas == 2
    assert ray.serve.get_deployment(name="Reader").num_replicas == 1
    assert ray.serve.get_deployment(name="ESRetriever").max_concurrent_queries == 17
    assert ray.serve.get_deployment(name="ESRetriever").ray_actor_options["num_cpus"] == 0.5
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
