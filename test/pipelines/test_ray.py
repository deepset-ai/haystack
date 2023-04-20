from pathlib import Path

import pytest
import ray

from haystack.pipelines import RayPipeline


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
def test_load_pipeline(document_store_with_docs, serve_detached, samples_path):
    pipeline = RayPipeline.load_from_yaml(
        samples_path / "pipeline" / "ray.simple.haystack-pipeline.yml",
        pipeline_name="ray_query_pipeline",
        ray_args={"num_cpus": 8},
        serve_args={"detached": serve_detached},
    )
    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )

    assert pipeline._serve_controller_client._detached == serve_detached
    assert ray.serve.get_deployment(name="ESRetriever").num_replicas == 2
    assert ray.serve.get_deployment(name="Reader").num_replicas == 1
    assert ray.serve.get_deployment(name="ESRetriever").max_concurrent_queries == 17
    assert ray.serve.get_deployment(name="ESRetriever").ray_actor_options["num_cpus"] == 0.5
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"


@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_load_advanced_pipeline(document_store_with_docs, samples_path):
    pipeline = RayPipeline.load_from_yaml(
        samples_path / "pipeline" / "ray.advanced.haystack-pipeline.yml",
        pipeline_name="ray_query_pipeline",
        ray_args={"num_cpus": 8},
        serve_args={"detached": True},
    )
    prediction = pipeline.run(
        query="Who lives in Berlin?",
        params={"ESRetriever1": {"top_k": 1}, "ESRetriever2": {"top_k": 2}, "Reader": {"top_k": 3}},
    )

    assert pipeline._serve_controller_client._detached is True
    assert ray.serve.get_deployment(name="ESRetriever1").num_replicas == 2
    assert ray.serve.get_deployment(name="ESRetriever2").num_replicas == 2
    assert ray.serve.get_deployment(name="Reader").num_replicas == 1
    assert ray.serve.get_deployment(name="ESRetriever1").max_concurrent_queries == 17
    assert ray.serve.get_deployment(name="ESRetriever2").max_concurrent_queries == 15
    assert ray.serve.get_deployment(name="ESRetriever1").ray_actor_options["num_cpus"] == 0.25
    assert ray.serve.get_deployment(name="ESRetriever2").ray_actor_options["num_cpus"] == 0.25
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert len(prediction["answers"]) > 1


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
async def test_load_advanced_pipeline_async(document_store_with_docs, samples_path):
    pipeline = RayPipeline.load_from_yaml(
        samples_path / "pipeline" / "ray.advanced.haystack-pipeline.yml",
        pipeline_name="ray_query_pipeline",
        ray_args={"num_cpus": 8},
        serve_args={"detached": True},
    )
    prediction = await pipeline.run_async(
        query="Who lives in Berlin?",
        params={"ESRetriever1": {"top_k": 1}, "ESRetriever2": {"top_k": 2}, "Reader": {"top_k": 3}},
        debug=True,
    )

    assert pipeline._serve_controller_client._detached is True
    assert ray.serve.get_deployment(name="ESRetriever1").num_replicas == 2
    assert ray.serve.get_deployment(name="ESRetriever2").num_replicas == 2
    assert ray.serve.get_deployment(name="Reader").num_replicas == 1
    assert ray.serve.get_deployment(name="ESRetriever1").max_concurrent_queries == 17
    assert ray.serve.get_deployment(name="ESRetriever2").max_concurrent_queries == 15
    assert ray.serve.get_deployment(name="ESRetriever1").ray_actor_options["num_cpus"] == 0.25
    assert ray.serve.get_deployment(name="ESRetriever2").ray_actor_options["num_cpus"] == 0.25
    assert prediction["query"] == "Who lives in Berlin?"
    assert prediction["answers"][0].answer == "Carla"
    assert len(prediction["answers"]) > 1
    assert "exec_time_ms" in prediction["_debug"]["ESRetriever1"].keys()
    assert prediction["_debug"]["ESRetriever1"]["exec_time_ms"]
