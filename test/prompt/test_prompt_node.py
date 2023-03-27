import os
import logging
from typing import Optional, Set, Type, Union, List, Dict, Any, Tuple

import pytest
import torch

from haystack import Document, Pipeline, BaseComponent, MultiLabel
from haystack.errors import OpenAIError
from haystack.nodes.prompt import PromptTemplate, PromptNode, PromptModel
from haystack.nodes.prompt import PromptModelInvocationLayer
from haystack.nodes.prompt.prompt_node import PromptTemplateValidationError
from haystack.nodes.prompt.providers import HFLocalInvocationLayer, TokenStreamingHandler
from haystack.schema import Answer


def skip_test_for_invalid_key(prompt_model):
    if prompt_model.api_key is not None and prompt_model.api_key == "KEY_NOT_FOUND":
        pytest.skip("No API key found, skipping test")


class TestTokenStreamingHandler(TokenStreamingHandler):
    stream_handler_invoked = False

    def __call__(self, token_received, *args, **kwargs) -> str:
        """
        This callback method is called when a new token is received from the stream.

        :param token_received: The token received from the stream.
        :param kwargs: Additional keyword arguments passed to the underlying model.
        :return: The token to be sent to the stream.
        """
        self.stream_handler_invoked = True
        return token_received


class CustomInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return ["fake_response"]

    def _ensure_token_limit(self, prompt: str) -> str:
        return prompt

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return model_name_or_path == "fake_model"


@pytest.fixture
def get_api_key(request):
    if request.param == "openai":
        return os.environ.get("OPENAI_API_KEY", None)
    elif request.param == "azure":
        return os.environ.get("AZURE_OPENAI_API_KEY", None)


@pytest.mark.unit
def test_prompt_templates():
    p = PromptTemplate("t1", "Here is some fake template with variable {foo}")
    assert set(p.prompt_params) == {"foo"}

    p = PromptTemplate("t3", "Here is some fake template with variable {foo} and {bar}")
    assert set(p.prompt_params) == {"foo", "bar"}

    p = PromptTemplate("t4", "Here is some fake template with variable {foo1} and {bar2}")
    assert set(p.prompt_params) == {"foo1", "bar2"}

    p = PromptTemplate("t4", "Here is some fake template with variable {foo_1} and {bar_2}")
    assert set(p.prompt_params) == {"foo_1", "bar_2"}

    p = PromptTemplate("t4", "Here is some fake template with variable {Foo_1} and {Bar_2}")
    assert set(p.prompt_params) == {"Foo_1", "Bar_2"}

    p = PromptTemplate("t4", "'Here is some fake template with variable {baz}'")
    assert set(p.prompt_params) == {"baz"}
    # strip single quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable {baz}"

    p = PromptTemplate("t4", '"Here is some fake template with variable {baz}"')
    assert set(p.prompt_params) == {"baz"}
    # strip double quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable {baz}"


@pytest.mark.unit
def test_prompt_template_repr():
    p = PromptTemplate("t", "Here is variable {baz}")
    desired_repr = "PromptTemplate(name=t, prompt_text=Here is variable {baz}, prompt_params=['baz'])"
    assert repr(p) == desired_repr
    assert str(p) == desired_repr


@pytest.mark.unit
def test_prompt_node_with_custom_invocation_layer():
    model = PromptModel("fake_model")
    pn = PromptNode(model_name_or_path=model)
    output = pn("Some fake invocation")

    assert output == ["fake_response"]


@pytest.mark.integration
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


def test_create_prompt_model_dtype():
    model = PromptModel("google/flan-t5-small", model_kwargs={"torch_dtype": "auto"})
    assert model.model_name_or_path == "google/flan-t5-small"

    model = PromptModel("google/flan-t5-small", model_kwargs={"torch_dtype": "torch.bfloat16"})
    assert model.model_name_or_path == "google/flan-t5-small"


@pytest.mark.integration
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

    with pytest.raises(ValueError, match="Model some-random-model is not supported"):
        PromptNode("some-random-model")


@pytest.mark.integration
def test_add_and_remove_template(prompt_node):
    num_default_tasks = len(prompt_node.get_prompt_template_names())
    custom_task = PromptTemplate(name="custom-task", prompt_text="Custom task: {param1}, {param2}")
    prompt_node.add_prompt_template(custom_task)
    assert len(prompt_node.get_prompt_template_names()) == num_default_tasks + 1
    assert "custom-task" in prompt_node.get_prompt_template_names()

    assert prompt_node.remove_prompt_template("custom-task") is not None
    assert "custom-task" not in prompt_node.get_prompt_template_names()


@pytest.mark.integration
def test_add_template_and_invoke(prompt_node):
    tt = PromptTemplate(
        name="sentiment-analysis-new",
        prompt_text="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: {documents}; Answer:",
    )
    prompt_node.add_prompt_template(tt)

    r = prompt_node.prompt("sentiment-analysis-new", documents=["Berlin is an amazing city."])
    assert r[0].casefold() == "positive"


@pytest.mark.integration
def test_on_the_fly_prompt(prompt_node):
    prompt_template = PromptTemplate(
        name="sentiment-analysis-temp",
        prompt_text="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: {documents}; Answer:",
    )
    r = prompt_node.prompt(prompt_template, documents=["Berlin is an amazing city."])
    assert r[0].casefold() == "positive"


@pytest.mark.integration
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


@pytest.mark.integration
def test_question_generation(prompt_node):
    r = prompt_node.prompt("question-generation", documents=["Berlin is the capital of Germany."])
    assert len(r) == 1 and len(r[0]) > 0


@pytest.mark.integration
def test_template_selection(prompt_node):
    qa = prompt_node.set_default_prompt_template("question-answering-per-document")
    r = qa(
        ["Berlin is the capital of Germany.", "Paris is the capital of France."],
        ["What is the capital of Germany?", "What is the capital of France"],
    )
    assert r[0].answer.casefold() == "berlin" and r[1].answer.casefold() == "paris"


@pytest.mark.integration
def test_has_supported_template_names(prompt_node):
    assert len(prompt_node.get_prompt_template_names()) > 0


@pytest.mark.integration
def test_invalid_template_params(prompt_node):
    with pytest.raises(ValueError, match="Expected prompt parameters"):
        prompt_node.prompt("question-answering-per-document", {"some_crazy_key": "Berlin is the capital of Germany."})


@pytest.mark.integration
def test_wrong_template_params(prompt_node):
    with pytest.raises(ValueError, match="Expected prompt parameters"):
        # with don't have options param, multiple choice QA has
        prompt_node.prompt("question-answering-per-document", options=["Berlin is the capital of Germany."])


@pytest.mark.integration
def test_run_invalid_template(prompt_node):
    with pytest.raises(ValueError, match="invalid-task not supported"):
        prompt_node.prompt("invalid-task", {})


@pytest.mark.integration
def test_invalid_prompting(prompt_node):
    with pytest.raises(ValueError, match="Hey there, what is the best city in the"):
        prompt_node.prompt(["Hey there, what is the best city in the world?", "Hey, answer me!"])


@pytest.mark.integration
def test_prompt_at_query_time(prompt_node: PromptNode):
    results = prompt_node.prompt("Hey there, what is the best city in the world?")
    assert len(results) == 1
    assert isinstance(results[0], str)


@pytest.mark.integration
def test_invalid_state_ops(prompt_node):
    with pytest.raises(ValueError, match="Prompt template no_such_task_exists"):
        prompt_node.remove_prompt_template("no_such_task_exists")
        # remove default task
        prompt_node.remove_prompt_template("question-answering-per-document")


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_open_ai_prompt_with_params(prompt_model):
    skip_test_for_invalid_key(prompt_model)
    pn = PromptNode(prompt_model)
    optional_davinci_params = {"temperature": 0.5, "max_tokens": 10, "top_p": 1, "frequency_penalty": 0.5}
    r = pn.prompt("question-generation", documents=["Berlin is the capital of Germany."], **optional_davinci_params)
    assert len(r) == 1 and len(r[0]) > 0


@pytest.mark.integration
def test_open_ai_prompt_with_default_params(azure_conf):
    if not azure_conf:
        pytest.skip("No Azure API key found, skipping test")
    model_kwargs = {"temperature": 0.5, "max_tokens": 2, "top_p": 1, "frequency_penalty": 0.5}
    model_kwargs.update(azure_conf)
    pn = PromptNode(model_name_or_path="text-davinci-003", api_key=azure_conf["api_key"], model_kwargs=model_kwargs)
    result = pn.prompt("question-generation", documents=["Berlin is the capital of Germany."])
    assert len(result) == 1 and len(result[0]) > 0


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_open_ai_warn_if_max_tokens_is_too_short(prompt_model, caplog):
    skip_test_for_invalid_key(prompt_model)
    pn = PromptNode(prompt_model)
    optional_davinci_params = {"temperature": 0.5, "max_tokens": 2, "top_p": 1, "frequency_penalty": 0.5}
    with caplog.at_level(logging.WARNING):
        _ = pn.prompt("question-generation", documents=["Berlin is the capital of Germany."], **optional_davinci_params)
        assert "Increase the max_tokens parameter to allow for longer completions." in caplog.text


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_stop_words(prompt_model):
    skip_test_for_invalid_key(prompt_model)

    # test stop words for both HF and OpenAI
    # set stop words in PromptNode
    node = PromptNode(prompt_model, stop_words=["capital", "Germany"])

    # with default prompt template and stop words set in PN
    r = node.prompt("question-generation", documents=["Berlin is the capital of Germany."])
    assert r[0] == "What is the" or r[0] == "What city is the"

    # with default prompt template and stop words set in kwargs (overrides PN stop words)
    r = node.prompt("question-generation", documents=["Berlin is the capital of Germany."], stop_words=None)
    assert "capital" in r[0] or "Germany" in r[0]

    # simple prompting
    r = node("Given the context please generate a question. Context: Berlin is the capital of Germany.; Question:")
    assert len(r[0]) > 0
    assert "capital" not in r[0]
    assert "Germany" not in r[0]

    # simple prompting with stop words set in kwargs (overrides PN stop words)
    r = node(
        "Given the context please generate a question. Context: Berlin is the capital of Germany.; Question:",
        stop_words=None,
    )
    assert "capital" in r[0] or "Germany" in r[0]

    tt = PromptTemplate(
        name="question-generation-copy",
        prompt_text="Given the context please generate a question. Context: {documents}; Question:",
    )
    # with custom prompt template
    r = node.prompt(tt, documents=["Berlin is the capital of Germany."])
    assert r[0] == "What is the" or r[0] == "What city is the"

    # with custom prompt template and stop words set in kwargs (overrides PN stop words)
    r = node.prompt(tt, documents=["Berlin is the capital of Germany."], stop_words=None)
    assert "capital" in r[0] or "Germany" in r[0]


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_streaming_prompt_node_with_params(prompt_model):
    skip_test_for_invalid_key(prompt_model)

    # test streaming of calls to OpenAI by passing a stream handler to the prompt method
    ttsh = TestTokenStreamingHandler()
    node = PromptNode(prompt_model)
    response = node("What are some of the best cities in the world to live and why?", stream=True, stream_handler=ttsh)

    assert len(response[0]) > 0, "Response should not be empty"
    assert ttsh.stream_handler_invoked, "Stream handler should have been invoked"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key.",
)
def test_streaming_prompt_node():
    ttsh = TestTokenStreamingHandler()

    # test streaming of all calls to OpenAI by registering a stream handler as a model kwarg
    node = PromptNode(
        "text-davinci-003", api_key=os.environ.get("OPENAI_API_KEY"), model_kwargs={"stream_handler": ttsh}
    )
    response = node("What are some of the best cities in the world to live?")

    assert len(response[0]) > 0, "Response should not be empty"
    assert ttsh.stream_handler_invoked, "Stream handler should have been invoked"


def test_prompt_node_with_text_generation_model():
    # test simple prompting with text generation model
    # by default, we force the model not return prompt text
    # Thus text-generation models can be used with PromptNode
    # just like text2text-generation models
    node = PromptNode("bigscience/bigscience-small-testing")
    r = node("Hello big science!")
    assert len(r[0]) > 0

    # test prompting with parameter to return prompt text as well
    # users can use this param to get the prompt text and the generated text
    r = node("Hello big science!", return_full_text=True)
    assert len(r[0]) > 0 and r[0].startswith("Hello big science!")


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_simple_pipeline(prompt_model):
    skip_test_for_invalid_key(prompt_model)

    node = PromptNode(prompt_model, default_prompt_template="sentiment-analysis", output_variable="out")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    assert "positive" in result["out"][0].casefold()


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_complex_pipeline(prompt_model):
    skip_test_for_invalid_key(prompt_model)

    node = PromptNode(prompt_model, default_prompt_template="question-generation", output_variable="query")
    node2 = PromptNode(prompt_model, default_prompt_template="question-answering-per-document")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    pipe.add_node(component=node2, name="prompt_node_2", inputs=["prompt_node"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert "berlin" in result["answers"][0].answer.casefold()


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_simple_pipeline_with_topk(prompt_model):
    skip_test_for_invalid_key(prompt_model)

    node = PromptNode(prompt_model, default_prompt_template="question-generation", output_variable="query", top_k=2)

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert len(result["query"]) == 2


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_pipeline_with_standard_qa(prompt_model):
    skip_test_for_invalid_key(prompt_model)
    node = PromptNode(prompt_model, default_prompt_template="question-answering", top_k=1)

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(
        query="Who lives in Berlin?",  # this being a string instead of a list what is being tested
        documents=[
            Document("My name is Carla and I live in Berlin", id="1"),
            Document("My name is Christelle and I live in Paris", id="2"),
        ],
    )

    assert len(result["answers"]) == 1
    assert "carla" in result["answers"][0].answer.casefold()

    assert result["answers"][0].document_ids == ["1", "2"]
    assert (
        result["answers"][0].meta["prompt"]
        == "Given the context please answer the question. Context: My name is Carla and I live in Berlin My name is Christelle and I live in Paris; "
        "Question: Who lives in Berlin?; Answer:"
    )


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_pipeline_with_qa_with_references(prompt_model):
    skip_test_for_invalid_key(prompt_model)
    node = PromptNode(prompt_model, default_prompt_template="question-answering-with-references", top_k=1)

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(
        query="Who lives in Berlin?",  # this being a string instead of a list what is being tested
        documents=[
            Document("My name is Carla and I live in Berlin", id="1"),
            Document("My name is Christelle and I live in Paris", id="2"),
        ],
    )

    assert len(result["answers"]) == 1
    assert "carla, as stated in document[1]" in result["answers"][0].answer.casefold()

    assert result["answers"][0].document_ids == ["1"]
    assert (
        result["answers"][0].meta["prompt"]
        == "Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents. "
        "You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[number] notation. "
        "If multiple documents contain the answer, cite those documents like ‘as stated in Document[number], Document[number], etc.’. If the documents do not contain the answer to the question, "
        "say that ‘answering is not possible given the available information.’\n\nDocument[1]: My name is Carla and I live in Berlin\n\nDocument[2]: My name is Christelle and I live in Paris \n "
        "Question: Who lives in Berlin?; Answer: "
    )


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_pipeline_with_prompt_text_at_query_time(prompt_model):
    skip_test_for_invalid_key(prompt_model)
    node = PromptNode(prompt_model, default_prompt_template="question-answering-with-references", top_k=1)

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(
        query="Who lives in Berlin?",  # this being a string instead of a list what is being tested
        documents=[
            Document("My name is Carla and I live in Berlin", id="1"),
            Document("My name is Christelle and I live in Paris", id="2"),
        ],
        params={
            "prompt_template": "Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents. Cite the documents using Document[number] notation.\n\n{join(documents, delimiter=new_line+new_line, pattern='Document[$idx]: $content')}\n\nQuestion: {query}\n\nAnswer: "
        },
    )

    assert len(result["answers"]) == 1
    assert "carla" in result["answers"][0].answer.casefold()

    assert result["answers"][0].document_ids == ["1"]
    assert (
        result["answers"][0].meta["prompt"]
        == "Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents. Cite the documents using Document[number] notation.\n\n"
        "Document[1]: My name is Carla and I live in Berlin\n\nDocument[2]: My name is Christelle and I live in Paris\n\n"
        "Question: Who lives in Berlin?\n\nAnswer: "
    )


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_pipeline_with_prompt_template_at_query_time(prompt_model):
    skip_test_for_invalid_key(prompt_model)
    node = PromptNode(prompt_model, default_prompt_template="question-answering-with-references", top_k=1)

    prompt_template_yaml = """
            name: "question-answering-with-references-custom"
            prompt_text: 'Create a concise and informative answer (no more than 50 words) for
                a given question based solely on the given documents. Cite the documents using Doc[number] notation.


                {join(documents, delimiter=new_line+new_line, pattern=''Doc[$idx]: $content'')}


                Question: {query}


                Answer: '
            output_parser:
                type: AnswerParser
                params:
                    reference_pattern: Doc\\[([^\\]]+)\\]
        """

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(
        query="Who lives in Berlin?",  # this being a string instead of a list what is being tested
        documents=[
            Document("My name is Carla and I live in Berlin", id="doc-1"),
            Document("My name is Christelle and I live in Paris", id="doc-2"),
        ],
        params={"prompt_template": prompt_template_yaml},
    )

    assert len(result["answers"]) == 1
    assert "carla" in result["answers"][0].answer.casefold()

    assert result["answers"][0].document_ids == ["doc-1"]
    assert (
        result["answers"][0].meta["prompt"]
        == "Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents. Cite the documents using Doc[number] notation.\n\n"
        "Doc[1]: My name is Carla and I live in Berlin\n\nDoc[2]: My name is Christelle and I live in Paris\n\n"
        "Question: Who lives in Berlin?\n\nAnswer: "
    )


@pytest.mark.integration
def test_pipeline_with_prompt_template_and_nested_shaper_yaml(tmp_path):
    with open(tmp_path / "tmp_config_with_prompt_template.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: template_with_nested_shaper
              type: PromptTemplate
              params:
                name: custom-template-with-nested-shaper
                prompt_text: "Given the context please answer the question. Context: {{documents}}; Question: {{query}}; Answer: "
                output_parser:
                  type: AnswerParser
            - name: p1
              params:
                model_name_or_path: google/flan-t5-small
                default_prompt_template: template_with_nested_shaper
              type: PromptNode
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config_with_prompt_template.yml")
    result = pipeline.run(query="What is an amazing city?", documents=[Document("Berlin is an amazing city.")])
    answer = result["answers"][0].answer
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in answer.casefold())
    assert (
        result["answers"][0].meta["prompt"]
        == "Given the context please answer the question. Context: Berlin is an amazing city.; Question: What is an amazing city?; Answer: "
    )


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf"], indirect=True)
def test_prompt_node_no_debug(prompt_model):
    """Pipeline with PromptNode should not generate debug info if debug is false."""

    node = PromptNode(prompt_model, default_prompt_template="question-generation", top_k=2)
    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])

    # debug explicitely False
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")], debug=False)
    assert result.get("_debug", "No debug info") == "No debug info"

    # debug None
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")], debug=None)
    assert result.get("_debug", "No debug info") == "No debug info"

    # debug True
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")], debug=True)
    assert (
        result["_debug"]["prompt_node"]["runtime"]["prompts_used"][0]
        == "Given the context please generate a question. Context: Berlin is the capital of Germany; Question:"
    )


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_complex_pipeline_with_qa(prompt_model):
    """Test the PromptNode where the `query` is a string instead of a list what the PromptNode would expects,
    because in a question-answering pipeline the retrievers need `query` as a string, so the PromptNode
    need to be able to handle the `query` being a string instead of a list."""
    skip_test_for_invalid_key(prompt_model)

    prompt_template = PromptTemplate(
        name="question-answering-new",
        prompt_text="Given the context please answer the question. Context: {documents}; Question: {query}; Answer:",
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
        debug=True,  # so we can verify that the constructed prompt is returned in debug
    )

    assert len(result["results"]) == 2
    assert "carla" in result["results"][0].casefold()

    # also verify that the PromptNode has included its constructed prompt LLM model input in the returned debug
    assert (
        result["_debug"]["prompt_node"]["runtime"]["prompts_used"][0]
        == "Given the context please answer the question. Context: My name is Carla and I live in Berlin; "
        "Question: Who lives in Berlin?; Answer:"
    )


@pytest.mark.integration
def test_complex_pipeline_with_shared_model():
    model = PromptModel()
    node = PromptNode(model_name_or_path=model, default_prompt_template="question-generation", output_variable="query")
    node2 = PromptNode(model_name_or_path=model, default_prompt_template="question-answering-per-document")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    pipe.add_node(component=node2, name="prompt_node_2", inputs=["prompt_node"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert result["answers"][0].answer == "Berlin"


@pytest.mark.integration
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


@pytest.mark.integration
def test_simple_pipeline_yaml_with_default_params(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: p1
              type: PromptNode
              params:
                default_prompt_template: sentiment-analysis
                model_kwargs:
                  torch_dtype: torch.bfloat16
            pipelines:
            - name: query
              nodes:
              - name: p1
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert pipeline.graph.nodes["p1"]["component"].prompt_model.model_kwargs == {"torch_dtype": "torch.bfloat16"}

    result = pipeline.run(query=None, documents=[Document("Berlin is an amazing city.")])
    assert result["results"][0] == "positive"


@pytest.mark.integration
def test_complex_pipeline_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: p1
              params:
                default_prompt_template: question-generation
                output_variable: query
              type: PromptNode
            - name: p2
              params:
                default_prompt_template: question-answering-per-document
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
    response = result["answers"][0].answer
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert len(result["query"]) > 0
    assert "query" in result["invocation_context"] and len(result["invocation_context"]["query"]) > 0


@pytest.mark.integration
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
                output_variable: query
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering-per-document
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
    response = result["answers"][0].answer
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert len(result["query"]) > 0
    assert "query" in result["invocation_context"] and len(result["invocation_context"]["query"]) > 0


@pytest.mark.integration
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
                  torch_dtype: auto
            - name: question_generation_template
              type: PromptTemplate
              params:
                name: question-generation-new
                prompt_text: "Given the context please generate a question. Context: {{documents}}; Question:"
            - name: p1
              params:
                model_name_or_path: pmodel
                default_prompt_template: question_generation_template
                output_variable: query
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering-per-document
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
    response = result["answers"][0].answer
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert len(result["query"]) > 0
    assert "query" in result["invocation_context"] and len(result["invocation_context"]["query"]) > 0


@pytest.mark.integration
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
                prompt_text: "Given the context please generate a question. Context: {{documents}}; Question:"
            - name: p1
              params:
                model_name_or_path: pmodel
                default_prompt_template: question_generation_template
                output_variable: query
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering-per-document
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
    response = result["answers"][0].answer
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert len(result["query"]) > 0
    assert "query" in result["invocation_context"] and len(result["invocation_context"]["query"]) > 0


@pytest.mark.parametrize("haystack_openai_config", ["openai", "azure"], indirect=True)
def test_complex_pipeline_with_all_features(tmp_path, haystack_openai_config):
    if not haystack_openai_config:
        pytest.skip("No API key found, skipping test")

    if "azure_base_url" in haystack_openai_config:
        # don't change this indentation, it's important for the yaml to be valid
        azure_conf_yaml_snippet = f"""
                  azure_base_url: {haystack_openai_config['azure_base_url']}
                  azure_deployment_name: {haystack_openai_config['azure_deployment_name']}
        """
    else:
        azure_conf_yaml_snippet = ""
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
                  {azure_conf_yaml_snippet}
                api_key: {haystack_openai_config["api_key"]}
            - name: question_generation_template
              type: PromptTemplate
              params:
                name: question-generation-new
                prompt_text: "Given the context please generate a question. Context: {{documents}}; Question:"
            - name: p1
              params:
                model_name_or_path: pmodel_openai
                default_prompt_template: question_generation_template
                output_variable: query
              type: PromptNode
            - name: p2
              params:
                model_name_or_path: pmodel
                default_prompt_template: question-answering-per-document
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
    response = result["answers"][0].answer
    assert any(word for word in ["berlin", "germany", "population", "city", "amazing"] if word in response.casefold())
    assert len(result["invocation_context"]) > 0
    assert len(result["query"]) > 0
    assert "query" in result["invocation_context"] and len(result["invocation_context"]["query"]) > 0


@pytest.mark.integration
def test_complex_pipeline_with_multiple_same_prompt_node_components_yaml(tmp_path):
    # p2 and p3 are essentially the same PromptNode component, make sure we can use them both as is in the pipeline
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: p1
              params:
                default_prompt_template: question-generation
              type: PromptNode
            - name: p2
              params:
                default_prompt_template: question-answering-per-document
              type: PromptNode
            - name: p3
              params:
                default_prompt_template: question-answering-per-document
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
              - name: p3
                inputs:
                - p2
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert pipeline is not None


class TestTokenLimit:
    @pytest.mark.integration
    def test_hf_token_limit_warning(self, prompt_node, caplog):
        prompt_template = PromptTemplate(
            name="too-long-temp", prompt_text="Repeating text" * 200 + "Docs: {documents}; Answer:"
        )
        with caplog.at_level(logging.WARNING):
            _ = prompt_node.prompt(prompt_template, documents=["Berlin is an amazing city."])
            assert "The prompt has been truncated from 812 tokens to 412 tokens" in caplog.text
            assert "and answer length (100 tokens) fits within the max token limit (512 tokens)." in caplog.text

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_openai_token_limit_warning(self, caplog):
        tt = PromptTemplate(name="too-long-temp", prompt_text="Repeating text" * 200 + "Docs: {documents}; Answer:")
        prompt_node = PromptNode("text-ada-001", max_length=2000, api_key=os.environ.get("OPENAI_API_KEY", ""))
        with caplog.at_level(logging.WARNING):
            _ = prompt_node.prompt(tt, documents=["Berlin is an amazing city."])
            assert "The prompt has been truncated from" in caplog.text
            assert "and answer length (2000 tokens) fits within the max token limit (2049 tokens)." in caplog.text


class TestRunBatch:
    @pytest.mark.integration
    @pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
    def test_simple_pipeline_batch_no_query_single_doc_list(self, prompt_model):
        skip_test_for_invalid_key(prompt_model)

        node = PromptNode(prompt_model, default_prompt_template="sentiment-analysis")

        pipe = Pipeline()
        pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
        result = pipe.run_batch(
            queries=None, documents=[Document("Berlin is an amazing city."), Document("I am not feeling well.")]
        )
        assert isinstance(result["results"], list)
        assert isinstance(result["results"][0], list)
        assert isinstance(result["results"][0][0], str)
        assert "positive" in result["results"][0][0].casefold()
        assert "negative" in result["results"][1][0].casefold()

    @pytest.mark.integration
    @pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
    def test_simple_pipeline_batch_no_query_multiple_doc_list(self, prompt_model):
        skip_test_for_invalid_key(prompt_model)

        node = PromptNode(prompt_model, default_prompt_template="sentiment-analysis", output_variable="out")

        pipe = Pipeline()
        pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
        result = pipe.run_batch(
            queries=None,
            documents=[
                [Document("Berlin is an amazing city."), Document("Paris is an amazing city.")],
                [Document("I am not feeling well.")],
            ],
        )
        assert isinstance(result["out"], list)
        assert isinstance(result["out"][0], list)
        assert isinstance(result["out"][0][0], str)
        assert all("positive" in x.casefold() for x in result["out"][0])
        assert "negative" in result["out"][1][0].casefold()

    @pytest.mark.integration
    @pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
    def test_simple_pipeline_batch_query_multiple_doc_list(self, prompt_model):
        skip_test_for_invalid_key(prompt_model)

        prompt_template = PromptTemplate(
            name="question-answering-new",
            prompt_text="Given the context please answer the question. Context: {documents}; Question: {query}; Answer:",
        )
        node = PromptNode(prompt_model, default_prompt_template=prompt_template)

        pipe = Pipeline()
        pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
        result = pipe.run_batch(
            queries=["Who lives in Berlin?"],
            documents=[
                [Document("My name is Carla and I live in Berlin"), Document("My name is James and I live in London")],
                [Document("My name is Christelle and I live in Paris")],
            ],
            debug=True,
        )
        assert isinstance(result["results"], list)
        assert isinstance(result["results"][0], list)
        assert isinstance(result["results"][0][0], str)


@pytest.mark.unit
def test_HFLocalInvocationLayer_supports():
    assert HFLocalInvocationLayer.supports("philschmid/flan-t5-base-samsum")
    assert HFLocalInvocationLayer.supports("bigscience/T0_3B")


class TestPromptTemplateSyntax:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, expected_prompt_params, expected_used_functions",
        [
            ("{documents}", {"documents"}, set()),
            ("Please answer the question: {documents} Question: how?", {"documents"}, set()),
            ("Please answer the question: {documents} Question: {query}", {"documents", "query"}, set()),
            ("Please answer the question: {documents} {{Question}}: {query}", {"documents", "query"}, set()),
            (
                "Please answer the question: {join(documents)} Question: {query.replace('A', 'a')}",
                {"documents", "query"},
                {"join", "replace"},
            ),
            (
                "Please answer the question: {join(documents, 'delim', {'{': '('})} Question: {query.replace('A', 'a')}",
                {"documents", "query"},
                {"join", "replace"},
            ),
            (
                'Please answer the question: {join(documents, "delim", {"{": "("})} Question: {query.replace("A", "a")}',
                {"documents", "query"},
                {"join", "replace"},
            ),
            (
                "Please answer the question: {join(documents, 'delim', {'a': {'b': 'c'}})} Question: {query.replace('A', 'a')}",
                {"documents", "query"},
                {"join", "replace"},
            ),
            (
                "Please answer the question: {join(document=documents, delimiter='delim', str_replace={'{': '('})} Question: {query.replace('A', 'a')}",
                {"documents", "query"},
                {"join", "replace"},
            ),
        ],
    )
    def test_prompt_template_syntax_parser(
        self, prompt_text: str, expected_prompt_params: Set[str], expected_used_functions: Set[str]
    ):
        prompt_template = PromptTemplate(name="test", prompt_text=prompt_text)
        assert set(prompt_template.prompt_params) == expected_prompt_params
        assert set(prompt_template._used_functions) == expected_used_functions

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, documents, query, expected_prompts",
        [
            ("{documents}", [Document("doc1"), Document("doc2")], None, ["doc1", "doc2"]),
            (
                "context: {documents} question: how?",
                [Document("doc1"), Document("doc2")],
                None,
                ["context: doc1 question: how?", "context: doc2 question: how?"],
            ),
            (
                "context: {' '.join([d.content for d in documents])} question: how?",
                [Document("doc1"), Document("doc2")],
                None,
                ["context: doc1 doc2 question: how?"],
            ),
            (
                "context: {documents} question: {query}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["context: doc1 question: how?", "context: doc2 question: how?"],
            ),
            (
                "context: {documents} {{question}}: {query}",
                [Document("doc1")],
                "how?",
                ["context: doc1 {question}: how?"],
            ),
            (
                "context: {join(documents)} question: {query}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["context: doc1 doc2 question: how?"],
            ),
            (
                "Please answer the question: {join(documents, ' delim ', '[$idx] $content', {'{': '('})} question: {query}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["Please answer the question: [1] doc1 delim [2] doc2 question: how?"],
            ),
            (
                "Please answer the question: {join(documents=documents, delimiter=' delim ', pattern='[$idx] $content', str_replace={'{': '('})} question: {query}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["Please answer the question: [1] doc1 delim [2] doc2 question: how?"],
            ),
            (
                "Please answer the question: {' delim '.join(['['+str(idx+1)+'] '+d.content.replace('{', '(') for idx, d in enumerate(documents)])} question: {query}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["Please answer the question: [1] doc1 delim [2] doc2 question: how?"],
            ),
            (
                'Please answer the question: {join(documents, " delim ", "[$idx] $content", {"{": "("})} question: {query}',
                [Document("doc1"), Document("doc2")],
                "how?",
                ["Please answer the question: [1] doc1 delim [2] doc2 question: how?"],
            ),
            (
                "context: {join(documents)} question: {query.replace('how', 'what')}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["context: doc1 doc2 question: what?"],
            ),
            (
                "context: {join(documents)[:6]} question: {query.replace('how', 'what').replace('?', '!')}",
                [Document("doc1"), Document("doc2")],
                "how?",
                ["context: doc1 d question: what!"],
            ),
            ("context", None, None, ["context"]),
        ],
    )
    def test_prompt_template_syntax_fill(
        self, prompt_text: str, documents: List[Document], query: str, expected_prompts: List[str]
    ):
        prompt_template = PromptTemplate(name="test", prompt_text=prompt_text)
        prompts = [prompt for prompt in prompt_template.fill(documents=documents, query=query)]
        assert prompts == expected_prompts

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, documents, expected_prompts",
        [
            ("{join(documents)}", [Document("doc1"), Document("doc2")], ["doc1 doc2"]),
            (
                "{join(documents, ' delim ', '[$idx] $content', {'c': 'C'})}",
                [Document("doc1"), Document("doc2")],
                ["[1] doC1 delim [2] doC2"],
            ),
            (
                "{join(documents, ' delim ', '[$id] $content', {'c': 'C'})}",
                [Document("doc1", id="123"), Document("doc2", id="456")],
                ["[123] doC1 delim [456] doC2"],
            ),
            (
                "{join(documents, ' delim ', '[$file_id] $content', {'c': 'C'})}",
                [Document("doc1", meta={"file_id": "123.txt"}), Document("doc2", meta={"file_id": "456.txt"})],
                ["[123.txt] doC1 delim [456.txt] doC2"],
            ),
        ],
    )
    def test_join(self, prompt_text: str, documents: List[Document], expected_prompts: List[str]):
        prompt_template = PromptTemplate(name="test", prompt_text=prompt_text)
        prompts = [prompt for prompt in prompt_template.fill(documents=documents)]
        assert prompts == expected_prompts

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, documents, expected_prompts",
        [
            ("{to_strings(documents)}", [Document("doc1"), Document("doc2")], ["doc1", "doc2"]),
            (
                "{to_strings(documents, '[$idx] $content', {'c': 'C'})}",
                [Document("doc1"), Document("doc2")],
                ["[1] doC1", "[2] doC2"],
            ),
            (
                "{to_strings(documents, '[$id] $content', {'c': 'C'})}",
                [Document("doc1", id="123"), Document("doc2", id="456")],
                ["[123] doC1", "[456] doC2"],
            ),
            (
                "{to_strings(documents, '[$file_id] $content', {'c': 'C'})}",
                [Document("doc1", meta={"file_id": "123.txt"}), Document("doc2", meta={"file_id": "456.txt"})],
                ["[123.txt] doC1", "[456.txt] doC2"],
            ),
            ("{to_strings(documents, '[$file_id] $content', {'c': 'C'})}", ["doc1", "doc2"], ["doC1", "doC2"]),
            (
                "{to_strings(documents, '[$idx] $answer', {'c': 'C'})}",
                [Answer("doc1"), Answer("doc2")],
                ["[1] doC1", "[2] doC2"],
            ),
        ],
    )
    def test_to_strings(self, prompt_text: str, documents: List[Document], expected_prompts: List[str]):
        prompt_template = PromptTemplate(name="test", prompt_text=prompt_text)
        prompts = [prompt for prompt in prompt_template.fill(documents=documents)]
        assert prompts == expected_prompts

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, exc_type, expected_exc_match",
        [
            ("{__import__('os').listdir('.')}", PromptTemplateValidationError, "Invalid function in prompt text"),
            ("{__import__('os')}", PromptTemplateValidationError, "Invalid function in prompt text"),
            (
                "{requests.get('https://haystack.deepset.ai/')}",
                PromptTemplateValidationError,
                "Invalid function in prompt text",
            ),
            ("{join(__import__('os').listdir('.'))}", PromptTemplateValidationError, "Invalid function in prompt text"),
            ("{for}", SyntaxError, "invalid syntax"),
            ("This is an invalid {variable .", SyntaxError, "f-string: expecting '}'"),
        ],
    )
    def test_prompt_template_syntax_init_raises(
        self, prompt_text: str, exc_type: Type[BaseException], expected_exc_match: str
    ):
        with pytest.raises(exc_type, match=expected_exc_match):
            PromptTemplate(name="test", prompt_text=prompt_text)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, documents, query, exc_type, expected_exc_match",
        [("{join}", None, None, ValueError, "Expected prompt parameters")],
    )
    def test_prompt_template_syntax_fill_raises(
        self,
        prompt_text: str,
        documents: List[Document],
        query: str,
        exc_type: Type[BaseException],
        expected_exc_match: str,
    ):
        with pytest.raises(exc_type, match=expected_exc_match):
            prompt_template = PromptTemplate(name="test", prompt_text=prompt_text)
            next(prompt_template.fill(documents=documents, query=query))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt_text, documents, query, expected_prompts",
        [
            ("__import__('os').listdir('.')", None, None, ["__import__('os').listdir('.')"]),
            (
                "requests.get('https://haystack.deepset.ai/')",
                None,
                None,
                ["requests.get('https://haystack.deepset.ai/')"],
            ),
            ("{query}", None, print, ["<built-in function print>"]),
            ("\b\b__import__('os').listdir('.')", None, None, ["\x08\x08__import__('os').listdir('.')"]),
        ],
    )
    def test_prompt_template_syntax_fill_ignores_dangerous_input(
        self, prompt_text: str, documents: List[Document], query: str, expected_prompts: List[str]
    ):
        prompt_template = PromptTemplate(name="test", prompt_text=prompt_text)
        prompts = [prompt for prompt in prompt_template.fill(documents=documents, query=query)]
        assert prompts == expected_prompts


@pytest.mark.integration
def test_chatgpt_direct_prompting(chatgpt_prompt_model):
    skip_test_for_invalid_key(chatgpt_prompt_model)
    pn = PromptNode(chatgpt_prompt_model)
    result = pn("Hey, I need some Python help. When should I use list comprehension?")
    assert len(result) == 1 and all(w in result[0] for w in ["comprehension", "list"])


@pytest.mark.integration
def test_chatgpt_direct_prompting_w_messages(chatgpt_prompt_model):
    skip_test_for_invalid_key(chatgpt_prompt_model)
    pn = PromptNode(chatgpt_prompt_model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ]

    result = pn(messages)
    assert len(result) == 1 and all(w in result[0].casefold() for w in ["arlington", "texas"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="No OpenAI API key provided. Please export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_chatgpt_promptnode():
    pn = PromptNode(model_name_or_path="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY", None))

    result = pn("Hey, I need some Python help. When should I use list comprehension?")
    assert len(result) == 1 and all(w in result[0] for w in ["comprehension", "list"])

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"},
    ]
    result = pn(messages)
    assert len(result) == 1 and all(w in result[0].casefold() for w in ["arlington", "texas"])
