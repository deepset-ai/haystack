import os
from typing import Optional, Union, List, Dict, Any, Tuple

import pytest
import torch

from haystack import Document, Pipeline, BaseComponent, MultiLabel
from haystack.errors import OpenAIError
from haystack.nodes.prompt import PromptTemplate, PromptNode, PromptModel


def is_openai_api_key_set(api_key: str):
    return len(api_key) > 0 and api_key != "KEY_NOT_FOUND"


def test_prompt_templates():
    p = PromptTemplate("t1", "Here is some fake template with variable $foo", ["foo"])

    with pytest.raises(ValueError, match="Number of parameters in"):
        PromptTemplate("t2", "Here is some fake template with variable $foo and $bar", ["foo"])

    with pytest.raises(ValueError, match="Invalid parameter"):
        PromptTemplate("t2", "Here is some fake template with variable $footur", ["foo"])

    with pytest.raises(ValueError, match="Number of parameters in"):
        PromptTemplate("t2", "Here is some fake template with variable $foo and $bar", ["foo", "bar", "baz"])

    p = PromptTemplate("t3", "Here is some fake template with variable $for and $bar", ["for", "bar"])

    # last parameter: "prompt_params" can be omitted
    p = PromptTemplate("t4", "Here is some fake template with variable $foo and $bar")
    assert p.prompt_params == ["foo", "bar"]

    p = PromptTemplate("t4", "Here is some fake template with variable $foo1 and $bar2")
    assert p.prompt_params == ["foo1", "bar2"]

    p = PromptTemplate("t4", "Here is some fake template with variable $foo_1 and $bar_2")
    assert p.prompt_params == ["foo_1", "bar_2"]

    p = PromptTemplate("t4", "Here is some fake template with variable $Foo_1 and $Bar_2")
    assert p.prompt_params == ["Foo_1", "Bar_2"]

    p = PromptTemplate("t4", "'Here is some fake template with variable $baz'")
    assert p.prompt_params == ["baz"]
    # strip single quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable $baz"

    p = PromptTemplate("t4", '"Here is some fake template with variable $baz"')
    assert p.prompt_params == ["baz"]
    # strip double quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable $baz"


def test_create_prompt_model():
    model = PromptModel("google/flan-t5-small")
    assert model.model_name_or_path == "google/flan-t5-small"

    model = PromptModel()
    assert model.model_name_or_path == "google/flan-t5-base"

    with pytest.raises(OpenAIError):
        # davinci selected but no API key provided
        model = PromptModel("text-davinci-003")

    model = PromptModel("text-davinci-003", api_key="no need to provide a real key")
    assert model.model_name_or_path == "text-davinci-003"

    with pytest.raises(ValueError, match="Model some-random-model is not supported"):
        PromptModel("some-random-model")

    # we can also pass model kwargs to the PromptModel
    model = PromptModel("google/flan-t5-small", model_kwargs={"model_kwargs": {"torch_dtype": torch.bfloat16}})
    assert model.model_name_or_path == "google/flan-t5-small"

    # we can also pass kwargs directly, see HF Pipeline constructor
    model = PromptModel("google/flan-t5-small", model_kwargs={"torch_dtype": torch.bfloat16})
    assert model.model_name_or_path == "google/flan-t5-small"

    # we can't use device_map auto without accelerate library installed
    with pytest.raises(ImportError, match="requires Accelerate: `pip install accelerate`"):
        model = PromptModel("google/flan-t5-small", model_kwargs={"device_map": "auto"})
        assert model.model_name_or_path == "google/flan-t5-small"


def test_create_prompt_node():
    prompt_node = PromptNode()
    assert prompt_node is not None
    assert prompt_node.prompt_model is not None

    prompt_node = PromptNode("google/flan-t5-small")
    assert prompt_node is not None
    assert prompt_node.model_name_or_path == "google/flan-t5-small"
    assert prompt_node.prompt_model is not None

    with pytest.raises(OpenAIError):
        # davinci selected but no API key provided
        prompt_node = PromptNode("text-davinci-003")

    prompt_node = PromptNode("text-davinci-003", api_key="no need to provide a real key")
    assert prompt_node is not None
    assert prompt_node.model_name_or_path == "text-davinci-003"
    assert prompt_node.prompt_model is not None

    with pytest.raises(ValueError, match="Model vblagoje/bart_lfqa is not supported"):
        # yes vblagoje/bart_lfqa is AutoModelForSeq2SeqLM, can be downloaded, however it is useless for prompting
        # currently support only T5-Flan models
        prompt_node = PromptNode("vblagoje/bart_lfqa")

    with pytest.raises(ValueError, match="Model valhalla/t5-base-e2e-qg is not supported"):
        # yes valhalla/t5-base-e2e-qg is AutoModelForSeq2SeqLM, can be downloaded, however it is useless for prompting
        # currently support only T5-Flan models
        prompt_node = PromptNode("valhalla/t5-base-e2e-qg")

    with pytest.raises(ValueError, match="Model some-random-model is not supported"):
        PromptNode("some-random-model")


def test_add_and_remove_template(prompt_node):
    num_default_tasks = len(prompt_node.get_prompt_template_names())
    custom_task = PromptTemplate(
        name="custom-task", prompt_text="Custom task: $param1, $param2", prompt_params=["param1", "param2"]
    )
    prompt_node.add_prompt_template(custom_task)
    assert len(prompt_node.get_prompt_template_names()) == num_default_tasks + 1
    assert "custom-task" in prompt_node.get_prompt_template_names()

    assert prompt_node.remove_prompt_template("custom-task") is not None
    assert "custom-task" not in prompt_node.get_prompt_template_names()


def test_invalid_template(prompt_node):
    with pytest.raises(ValueError, match="Invalid parameter"):
        PromptTemplate(
            name="custom-task", prompt_text="Custom task: $pram1 $param2", prompt_params=["param1", "param2"]
        )

    with pytest.raises(ValueError, match="Number of parameters"):
        PromptTemplate(name="custom-task", prompt_text="Custom task: $param1", prompt_params=["param1", "param2"])


def test_add_template_and_invoke(prompt_node):
    tt = PromptTemplate(
        name="sentiment-analysis-new",
        prompt_text="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: $documents; Answer:",
        prompt_params=["documents"],
    )
    prompt_node.add_prompt_template(tt)

    r = prompt_node.prompt("sentiment-analysis-new", documents=["Berlin is an amazing city."])
    assert r[0].casefold() == "positive"


def test_on_the_fly_prompt(prompt_node):
    tt = PromptTemplate(
        name="sentiment-analysis-temp",
        prompt_text="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: $documents; Answer:",
        prompt_params=["documents"],
    )
    r = prompt_node.prompt(tt, documents=["Berlin is an amazing city."])
    assert r[0].casefold() == "positive"


def test_direct_prompting(prompt_node):
    r = prompt_node("What is the capital of Germany?")
    assert r[0].casefold() == "berlin"

    r = prompt_node("What is the capital of Germany?", "What is the secret of universe?")
    assert r[0].casefold() == "berlin"
    assert len(r[1]) > 0

    r = prompt_node("Capital of Germany is Berlin", task="question-generation")
    assert len(r[0]) > 10 and "Germany" in r[0]

    r = prompt_node(["Capital of Germany is Berlin", "Capital of France is Paris"], task="question-generation")
    assert len(r) == 2


def test_question_generation(prompt_node):
    r = prompt_node.prompt("question-generation", documents=["Berlin is the capital of Germany."])
    assert len(r) == 1 and len(r[0]) > 0


def test_template_selection(prompt_node):
    qa = prompt_node.set_default_prompt_template("question-answering")
    r = qa(
        ["Berlin is the capital of Germany.", "Paris is the capital of France."],
        ["What is the capital of Germany?", "What is the capital of France"],
    )
    assert r[0].casefold() == "berlin" and r[1].casefold() == "paris"


def test_has_supported_template_names(prompt_node):
    assert len(prompt_node.get_prompt_template_names()) > 0


def test_invalid_template_params(prompt_node):
    with pytest.raises(ValueError, match="Expected prompt params"):
        prompt_node.prompt("question-answering", {"some_crazy_key": "Berlin is the capital of Germany."})


def test_wrong_template_params(prompt_node):
    with pytest.raises(ValueError, match="Expected prompt params"):
        # with don't have options param, multiple choice QA has
        prompt_node.prompt("question-answering", options=["Berlin is the capital of Germany."])


def test_run_invalid_template(prompt_node):
    with pytest.raises(ValueError, match="invalid-task not supported"):
        prompt_node.prompt("invalid-task", {})


def test_invalid_prompting(prompt_node):
    with pytest.raises(ValueError, match="Hey there, what is the best city in the worl"):
        prompt_node.prompt(
            "Hey there, what is the best city in the world?" "Hey there, what is the best city in the world?"
        )

    with pytest.raises(ValueError, match="Hey there, what is the best city in the"):
        prompt_node.prompt(["Hey there, what is the best city in the world?", "Hey, answer me!"])


def test_invalid_state_ops(prompt_node):
    with pytest.raises(ValueError, match="Prompt template no_such_task_exists"):
        prompt_node.remove_prompt_template("no_such_task_exists")
        # remove default task
        prompt_node.remove_prompt_template("question-answering")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_open_ai_prompt_with_params():
    pm = PromptModel("text-davinci-003", api_key=os.environ["OPENAI_API_KEY"])
    pn = PromptNode(pm)
    optional_davinci_params = {"temperature": 0.5, "max_tokens": 10, "top_p": 1, "frequency_penalty": 0.5}
    r = pn.prompt("question-generation", documents=["Berlin is the capital of Germany."], **optional_davinci_params)
    assert len(r) == 1 and len(r[0]) > 0


@pytest.mark.parametrize("prompt_model", ["hf", "openai"], indirect=True)
def test_stop_words(prompt_model):
    if prompt_model.api_key is not None and not is_openai_api_key_set(prompt_model.api_key):
        pytest.skip("No API key found for OpenAI, skipping test")

    node = PromptNode(prompt_model, stop_words=["capital", "Germany"])
    r = node.prompt("question-generation", documents=["Berlin is the capital of Germany."])
    assert r[0] == "What is the"


@pytest.mark.parametrize("prompt_model", ["hf", "openai"], indirect=True)
def test_simple_pipeline(prompt_model):
    if prompt_model.api_key is not None and not is_openai_api_key_set(prompt_model.api_key):
        pytest.skip("No API key found for OpenAI, skipping test")

    node = PromptNode(prompt_model, default_prompt_template="sentiment-analysis")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    assert result["results"][0].casefold() == "positive"


@pytest.mark.parametrize("prompt_model", ["hf", "openai"], indirect=True)
def test_complex_pipeline(prompt_model):
    if prompt_model.api_key is not None and not is_openai_api_key_set(prompt_model.api_key):
        pytest.skip("No API key found for OpenAI, skipping test")

    node = PromptNode(prompt_model, default_prompt_template="question-generation", output_variable="questions")
    node2 = PromptNode(prompt_model, default_prompt_template="question-answering")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    pipe.add_node(component=node2, name="prompt_node_2", inputs=["prompt_node"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert "berlin" in result["results"][0].casefold()


@pytest.mark.parametrize("prompt_model", ["hf", "openai"], indirect=True)
def test_complex_pipeline_with_qa(prompt_model):
    """Test the PromptNode where the `query` is a string instead of a list what the PromptNode would expects,
    because in a question-answering pipeline the retrievers need `query` as a string, so the PromptNode
    need to be able to handle the `query` being a string instead of a list."""
    if prompt_model.api_key is not None and not is_openai_api_key_set(prompt_model.api_key):
        pytest.skip("No API key found for OpenAI, skipping test")

    prompt_template = PromptTemplate(
        name="question-answering-new",
        prompt_text="Given the context please answer the question. Context: $documents; Question: $query; Answer:",
        prompt_params=["documents", "query"],
    )
    node = PromptNode(prompt_model, default_prompt_template=prompt_template)

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(
        query="Who lives in Berlin?",  # this being a string instead of a list what is being tested
        documents=[
            Document("My name is Carla and I live in Berlin"),
            Document("My name is Christelle and I live in Paris"),
        ],
    )

    assert len(result["results"]) == 1
    assert "carla" in result["results"][0].casefold()


def test_complex_pipeline_with_shared_model():
    model = PromptModel()
    node = PromptNode(
        model_name_or_path=model, default_prompt_template="question-generation", output_variable="questions"
    )
    node2 = PromptNode(model_name_or_path=model, default_prompt_template="question-answering")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    pipe.add_node(component=node2, name="prompt_node_2", inputs=["prompt_node"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert result["results"][0] == "Berlin"


def test_simple_pipeline_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: p1
              params:
                default_prompt_template: sentiment-analysis
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    assert result["results"][0] == "positive"


def test_complex_pipeline_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: p1
              params:
                default_prompt_template: question-generation
                output_variable: questions
              type: PromptNode
            - name: p2
              params:
                default_prompt_template: question-answering
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
              - name: p2
                inputs:
                - p1
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    response = result["results"][0]
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert "questions" in result["invocation_context"] and len(result["invocation_context"]["questions"]) > 0


def test_complex_pipeline_with_shared_prompt_model_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: pmodel
              type: PromptModel
            - name: p1
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-generation
                output_variable: questions
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
              - name: p2
                inputs:
                - p1
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    response = result["results"][0]
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert "questions" in result["invocation_context"] and len(result["invocation_context"]["questions"]) > 0


def test_complex_pipeline_with_shared_prompt_model_and_prompt_template_yaml(tmp_path):
    with open(tmp_path / "tmp_config_with_prompt_template.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: pmodel
              type: PromptModel
              params:
                model_name_or_path: google/flan-t5-small
                model_kwargs:
                  torch_dtype: torch.bfloat16
            - name: question_generation_template
              type: PromptTemplate
              params:
                name: question-generation-new
                prompt_text: "Given the context please generate a question. Context: $documents; Question:"
            - name: p1
              params:
                model_name_or_path: pmodel
                default_prompt_template: question_generation_template
                output_variable: questions
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
              - name: p2
                inputs:
                - p1
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config_with_prompt_template.yml")
    result = pipeline.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    response = result["results"][0]
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert "questions" in result["invocation_context"] and len(result["invocation_context"]["questions"]) > 0


def test_complex_pipeline_with_with_dummy_node_between_prompt_nodes_yaml(tmp_path):
    # test that we can stick some random node in between prompt nodes and that everything still works
    # most specifically, we want to ensure that invocation_context is still populated correctly and propagated
    class InBetweenNode(BaseComponent):
        outgoing_edges = 1

        def run(
            self,
            query: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[MultiLabel] = None,
            documents: Optional[List[Document]] = None,
            meta: Optional[dict] = None,
        ) -> Tuple[Dict, str]:
            return {}, "output_1"

        def run_batch(
            self,
            queries: Optional[Union[str, List[str]]] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
            documents: Optional[Union[List[Document], List[List[Document]]]] = None,
            meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            params: Optional[dict] = None,
            debug: Optional[bool] = None,
        ):
            return {}, "output_1"

    with open(tmp_path / "tmp_config_with_prompt_template.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: in_between
              type: InBetweenNode
            - name: pmodel
              type: PromptModel
              params:
                model_name_or_path: google/flan-t5-small
                model_kwargs:
                  torch_dtype: torch.bfloat16
            - name: question_generation_template
              type: PromptTemplate
              params:
                name: question-generation-new
                prompt_text: "Given the context please generate a question. Context: $documents; Question:"
            - name: p1
              params:
                model_name_or_path: pmodel
                default_prompt_template: question_generation_template
                output_variable: questions
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
              - name: in_between
                inputs:
                - p1
              - name: p2
                inputs:
                - in_between
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config_with_prompt_template.yml")
    result = pipeline.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    response = result["results"][0]
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert "questions" in result["invocation_context"] and len(result["invocation_context"]["questions"]) > 0


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_complex_pipeline_with_all_features(tmp_path):
    api_key = os.environ.get("OPENAI_API_KEY", None)
    with open(tmp_path / "tmp_config_with_prompt_template.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: pmodel
              type: PromptModel
              params:
                model_name_or_path: google/flan-t5-small
                model_kwargs:
                  torch_dtype: torch.bfloat16
            - name: pmodel_openai
              type: PromptModel
              params:
                model_name_or_path: text-davinci-003
                model_kwargs:
                  temperature: 0.9
                  max_tokens: 64
                api_key: {api_key}
            - name: question_generation_template
              type: PromptTemplate
              params:
                name: question-generation-new
                prompt_text: "Given the context please generate a question. Context: $documents; Question:"
            - name: p1
              params:
                model_name_or_path: pmodel_openai
                default_prompt_template: question_generation_template
                output_variable: questions
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
              - name: p2
                inputs:
                - p1
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config_with_prompt_template.yml")
    result = pipeline.run(query="not relevant", documents=[Document("Berlin is a city in Germany.")])
    response = result["results"][0]
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert "questions" in result["invocation_context"] and len(result["invocation_context"]["questions"]) > 0
