import os
import logging
from typing import Optional, Union, List, Dict, Any, Tuple
from unittest.mock import patch, Mock

import pytest

from haystack import Document, Pipeline, BaseComponent, MultiLabel
from haystack.nodes.prompt import PromptTemplate, PromptNode, PromptModel
from haystack.nodes.prompt.providers import HFLocalInvocationLayer


def skip_test_for_invalid_key(prompt_model):
    if prompt_model.api_key is not None and prompt_model.api_key == "KEY_NOT_FOUND":
        pytest.skip("No API key found, skipping test")


@pytest.fixture
def get_api_key(request):
    if request.param == "openai":
        return os.environ.get("OPENAI_API_KEY", None)
    elif request.param == "azure":
        return os.environ.get("AZURE_OPENAI_API_KEY", None)


@pytest.mark.unit
def test_add_and_remove_template():
    with patch("haystack.nodes.prompt.prompt_node.PromptModel"):
        node = PromptNode()

    # Verifies default
    assert len(node.get_prompt_template_names()) == 14

    # Add a fake template
    fake_template = PromptTemplate(name="fake-template", prompt_text="Fake prompt")
    node.add_prompt_template(fake_template)
    assert len(node.get_prompt_template_names()) == 15
    assert "fake-template" in node.get_prompt_template_names()

    # Verify that adding the same template throws an expection
    with pytest.raises(ValueError) as e:
        node.add_prompt_template(fake_template)
        assert e.match(
            "Prompt template fake-template already exists. Select a different name for this prompt template."
        )

    # Verify template is correctly removed
    assert node.remove_prompt_template("fake-template")
    assert len(node.get_prompt_template_names()) == 14
    assert "fake-template" not in node.get_prompt_template_names()

    # Verify that removing the same template throws an expection
    with pytest.raises(ValueError) as e:
        node.remove_prompt_template("fake-template")
        assert e.match("Prompt template fake-template does not exist")


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_after_adding_template(mock_model):
    # Make model always return something positive on invoke
    mock_model.return_value.invoke.return_value = ["positive"]

    # Create a template
    template = PromptTemplate(
        name="fake-sentiment-analysis",
        prompt_text="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: {documents}; Answer:",
    )

    # Execute prompt
    node = PromptNode()
    node.add_prompt_template(template)
    result = node.prompt("fake-sentiment-analysis", documents=["Berlin is an amazing city."])

    assert result == ["positive"]


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_passing_template(mock_model):
    # Make model always return something positive on invoke
    mock_model.return_value.invoke.return_value = ["positive"]

    # Create a template
    template = PromptTemplate(
        name="fake-sentiment-analysis",
        prompt_text="Please give a sentiment for this context. Answer with positive, "
        "negative or neutral. Context: {documents}; Answer:",
    )

    # Execute prompt
    node = PromptNode()
    result = node.prompt(template, documents=["Berlin is an amazing city."])

    assert result == ["positive"]


@pytest.mark.unit
@patch.object(PromptNode, "prompt")
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_call_with_no_kwargs(mock_model, mocked_prompt):
    node = PromptNode()
    node()
    mocked_prompt.assert_called_once_with(node.default_prompt_template)


@pytest.mark.unit
@patch.object(PromptNode, "prompt")
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_call_with_custom_kwargs(mock_model, mocked_prompt):
    node = PromptNode()
    node(some_kwarg="some_value")
    mocked_prompt.assert_called_once_with(node.default_prompt_template, some_kwarg="some_value")


@pytest.mark.unit
@patch.object(PromptNode, "prompt")
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_call_with_custom_template(mock_model, mocked_prompt):
    node = PromptNode()
    mock_template = Mock()
    node(prompt_template=mock_template)
    mocked_prompt.assert_called_once_with(mock_template)


@pytest.mark.unit
@patch.object(PromptNode, "prompt")
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_call_with_custom_kwargs_and_template(mock_model, mocked_prompt):
    node = PromptNode()
    mock_template = Mock()
    node(prompt_template=mock_template, some_kwarg="some_value")
    mocked_prompt.assert_called_once_with(mock_template, some_kwarg="some_value")


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_get_prompt_template_without_default_template(mock_model):
    node = PromptNode()
    assert node.get_prompt_template() is None

    template = node.get_prompt_template("question-answering")
    assert template.name == "question-answering"

    template = node.get_prompt_template(PromptTemplate(name="fake-template", prompt_text=""))
    assert template.name == "fake-template"

    with pytest.raises(ValueError) as e:
        node.get_prompt_template("some-unsupported-template")
        assert e.match("some-unsupported-template not supported, select one of:")

    fake_yaml_prompt = "name: fake-yaml-template\nprompt_text: fake prompt text"
    template = node.get_prompt_template(fake_yaml_prompt)
    assert template.name == "fake-yaml-template"

    fake_yaml_prompt = "- prompt_text: fake prompt text"
    template = node.get_prompt_template(fake_yaml_prompt)
    assert template.name == "custom-at-query-time"

    template = node.get_prompt_template("some prompt")
    assert template.name == "custom-at-query-time"


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_get_prompt_template_with_default_template(mock_model):
    node = PromptNode()
    node.set_default_prompt_template("question-answering")

    template = node.get_prompt_template()
    assert template.name == "question-answering"

    template = node.get_prompt_template("sentiment-analysis")
    assert template.name == "sentiment-analysis"

    template = node.get_prompt_template(PromptTemplate(name="fake-template", prompt_text=""))
    assert template.name == "fake-template"

    with pytest.raises(ValueError) as e:
        node.get_prompt_template("some-unsupported-template")
        assert e.match("some-unsupported-template not supported, select one of:")

    fake_yaml_prompt = "name: fake-yaml-template\nprompt_text: fake prompt text"
    template = node.get_prompt_template(fake_yaml_prompt)
    assert template.name == "fake-yaml-template"

    fake_yaml_prompt = "- prompt_text: fake prompt text"
    template = node.get_prompt_template(fake_yaml_prompt)
    assert template.name == "custom-at-query-time"

    template = node.get_prompt_template("some prompt")
    assert template.name == "custom-at-query-time"


@pytest.mark.integration
def test_invalid_template_params():
    # TODO: This can be a PromptTemplate unit test
    node = PromptNode("google/flan-t5-small", devices=["cpu"])
    with pytest.raises(ValueError, match="Expected prompt parameters"):
        node.prompt("question-answering-per-document", {"some_crazy_key": "Berlin is the capital of Germany."})


@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_stop_words(prompt_model):
    # TODO: This can be a unit test for StopWordCriteria
    skip_test_for_invalid_key(prompt_model)

    # test single stop word for both HF and OpenAI
    # set stop words in PromptNode
    node = PromptNode(prompt_model, stop_words=["capital"])

    # with default prompt template and stop words set in PN
    r = node.prompt("question-generation", documents=["Berlin is the capital of Germany."])
    assert r[0] == "What is the" or r[0] == "What city is the"

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


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_node_streaming_handler_on_call(mock_model):
    """
    Verifies model is created using expected stream handler when calling PromptNode.
    """
    mock_handler = Mock()
    node = PromptNode()
    node.prompt_model = mock_model
    node("What are some of the best cities in the world to live and why?", stream=True, stream_handler=mock_handler)
    # Verify model has been constructed with expected model_kwargs
    mock_model.invoke.assert_called_once()
    assert mock_model.invoke.call_args_list[0].kwargs["stream_handler"] == mock_handler


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_node_streaming_handler_on_constructor(mock_model):
    """
    Verifies model is created using expected stream handler when constructing PromptNode.
    """
    model_kwargs = {"stream_handler": Mock()}
    PromptNode(model_kwargs=model_kwargs)
    # Verify model has been constructed with expected model_kwargs
    mock_model.assert_called_once()
    assert mock_model.call_args_list[0].kwargs["model_kwargs"] == model_kwargs


@pytest.mark.skip
@pytest.mark.integration
def test_prompt_node_with_text_generation_model():
    # TODO: This is an integration test for HFLocalInvocationLayer
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


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_simple_pipeline(prompt_model):
    # TODO: This can be another unit test?
    skip_test_for_invalid_key(prompt_model)

    node = PromptNode(prompt_model, default_prompt_template="sentiment-analysis", output_variable="out")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is an amazing city.")])
    assert "positive" in result["out"][0].casefold()


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_complex_pipeline(prompt_model):
    # TODO: This is a unit test?
    skip_test_for_invalid_key(prompt_model)

    node = PromptNode(prompt_model, default_prompt_template="question-generation", output_variable="query")
    node2 = PromptNode(prompt_model, default_prompt_template="question-answering-per-document")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    pipe.add_node(component=node2, name="prompt_node_2", inputs=["prompt_node"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert "berlin" in result["answers"][0].answer.casefold()


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_simple_pipeline_with_topk(prompt_model):
    # TODO: This can be a unit test?
    skip_test_for_invalid_key(prompt_model)

    node = PromptNode(prompt_model, default_prompt_template="question-generation", output_variable="query", top_k=2)

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert len(result["query"]) == 2


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_pipeline_with_standard_qa(prompt_model):
    # TODO: Unit test?
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["openai", "azure"], indirect=True)
def test_pipeline_with_prompt_template_at_query_time(prompt_model):
    # TODO: This should be just an AnswerParser unit test and some PromptTemplate unit tests
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


@pytest.mark.skip
@pytest.mark.integration
def test_pipeline_with_prompt_template_and_nested_shaper_yaml(tmp_path):
    # TODO: This can be a Shaper unit test?
    with open(tmp_path / "tmp_config_with_prompt_template.yml", "w") as tmp_file:
        tmp_file.write(
            """
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


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf"], indirect=True)
def test_prompt_node_no_debug(prompt_model):
    # TODO: This is another unit test
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


@pytest.mark.skip
@pytest.mark.integration
@pytest.mark.parametrize("prompt_model", ["hf", "openai", "azure"], indirect=True)
def test_complex_pipeline_with_qa(prompt_model):
    # TODO: Not a PromptNode test, this maybe can be a unit test
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


@pytest.mark.skip
@pytest.mark.integration
def test_complex_pipeline_with_shared_model():
    # TODO: What is this testing? Can this be a unit test?
    model = PromptModel()
    node = PromptNode(model_name_or_path=model, default_prompt_template="question-generation", output_variable="query")
    node2 = PromptNode(model_name_or_path=model, default_prompt_template="question-answering-per-document")

    pipe = Pipeline()
    pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
    pipe.add_node(component=node2, name="prompt_node_2", inputs=["prompt_node"])
    result = pipe.run(query="not relevant", documents=[Document("Berlin is the capital of Germany")])

    assert result["answers"][0].answer == "Berlin"


@pytest.mark.skip
@pytest.mark.integration
def test_simple_pipeline_yaml(tmp_path):
    # TODO: This can be a unit test just to verify that loading
    # PromptNode from yaml creates a correctly runnable Pipeline.
    # Also it could probably be renamed to test_prompt_node_yaml_loading
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
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


@pytest.mark.skip
@pytest.mark.integration
def test_simple_pipeline_yaml_with_default_params(tmp_path):
    # TODO: Is this testing yaml loading?
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
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


@pytest.mark.skip
@pytest.mark.integration
def test_complex_pipeline_yaml(tmp_path):
    # TODO: Is this testing PromptNode or Pipeline?
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
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


@pytest.mark.skip
@pytest.mark.integration
def test_complex_pipeline_with_shared_prompt_model_yaml(tmp_path):
    # TODO: Is this similar to test_complex_pipeline_with_shared_model?
    # Why are we testing this two times?
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
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


@pytest.mark.skip
@pytest.mark.integration
def test_complex_pipeline_with_shared_prompt_model_and_prompt_template_yaml(tmp_path):
    # TODO: Is this testing PromptNode or Pipeline parsing?
    with open(tmp_path / "tmp_config_with_prompt_template.yml", "w") as tmp_file:
        tmp_file.write(
            """
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


@pytest.mark.skip
@pytest.mark.integration
def test_complex_pipeline_with_with_dummy_node_between_prompt_nodes_yaml(tmp_path):
    # TODO: This can be a unit test. Is it necessary though? Is it testing PromptNode?
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
            """
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


@pytest.mark.skip
@pytest.mark.parametrize("haystack_openai_config", ["openai", "azure"], indirect=True)
def test_complex_pipeline_with_all_features(tmp_path, haystack_openai_config):
    # TODO: Is this testing PromptNode or pipeline yaml parsing?
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


@pytest.mark.skip
@pytest.mark.integration
def test_complex_pipeline_with_multiple_same_prompt_node_components_yaml(tmp_path):
    # TODO: Can this become a unit test? Is it actually worth as a test?
    # p2 and p3 are essentially the same PromptNode component, make sure we can use them both as is in the pipeline
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
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
    def test_hf_token_limit_warning(self, caplog):
        prompt_template = PromptTemplate(
            name="too-long-temp", prompt_text="Repeating text" * 200 + "Docs: {documents}; Answer:"
        )
        with caplog.at_level(logging.WARNING):
            node = PromptNode("google/flan-t5-small", devices=["cpu"])
            node.prompt(prompt_template, documents=["Berlin is an amazing city."])
            assert "The prompt has been truncated from 812 tokens to 412 tokens" in caplog.text
            assert "and answer length (100 tokens) fit within the max token limit (512 tokens)." in caplog.text

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
            assert "and answer length (2000 tokens) fit within the max token limit (2049 tokens)." in caplog.text


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


@pytest.mark.skip
@pytest.mark.integration
def test_HFLocalInvocationLayer_supports():
    # TODO: HFLocalInvocationLayer test, to be moved
    assert HFLocalInvocationLayer.supports("philschmid/flan-t5-base-samsum")
    assert HFLocalInvocationLayer.supports("bigscience/T0_3B")


@pytest.mark.skip
@pytest.mark.integration
def test_chatgpt_direct_prompting(chatgpt_prompt_model):
    # TODO: This is testing ChatGPT, should be removed
    skip_test_for_invalid_key(chatgpt_prompt_model)
    pn = PromptNode(chatgpt_prompt_model)
    result = pn("Hey, I need some Python help. When should I use list comprehension?")
    assert len(result) == 1 and all(w in result[0] for w in ["comprehension", "list"])


@pytest.mark.skip
@pytest.mark.integration
def test_chatgpt_direct_prompting_w_messages(chatgpt_prompt_model):
    # TODO: This is a ChatGPTInvocationLayer unit test
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
