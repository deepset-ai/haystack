from typing import Set, Type, List
import textwrap
import os
from unittest.mock import patch, MagicMock

import pytest
import prompthub

from haystack.nodes.prompt import PromptTemplate
from haystack.nodes.prompt.prompt_node import PromptNode
from haystack.nodes.prompt.prompt_template import PromptTemplateValidationError, LEGACY_DEFAULT_TEMPLATES
from haystack.nodes.prompt import prompt_template
from haystack.nodes.prompt.shapers import AnswerParser
from haystack.pipelines.base import Pipeline
from haystack.schema import Answer, Document


@pytest.fixture
def enable_prompthub_cache(monkeypatch):
    monkeypatch.setenv("PROMPTHUB_CACHE_ENABLED", True)


@pytest.fixture
def prompthub_cache_path(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache"
    monkeypatch.setattr(prompt_template, "PROMPTHUB_CACHE_PATH", cache_path)
    yield cache_path


@pytest.fixture
def mock_prompthub():
    with patch("haystack.nodes.prompt.prompt_template.fetch_from_prompthub") as mock_prompthub:
        mock_prompthub.return_value = prompthub.Prompt(
            name="deepset/test-prompt",
            tags=["test"],
            meta={"author": "test"},
            version="v0.0.0",
            text="This is a test prompt. Use your knowledge to answer this question: {question}",
            description="test prompt",
        )
        yield mock_prompthub


@pytest.mark.unit
def test_prompt_templates_from_hub():
    with patch("haystack.nodes.prompt.prompt_template.prompthub") as mock_prompthub:
        PromptTemplate("deepset/question-answering")
        mock_prompthub.fetch.assert_called_with("deepset/question-answering", timeout=30)


@pytest.mark.unit
def test_prompt_templates_from_hub_prompts_are_cached(prompthub_cache_path, enable_prompthub_cache, mock_prompthub):
    PromptTemplate("deepset/test-prompt")
    assert (prompthub_cache_path / "deepset" / "test-prompt.yml").exists()


@pytest.mark.unit
def test_prompt_templates_from_hub_prompts_are_not_cached_if_disabled(prompthub_cache_path, mock_prompthub):
    PromptTemplate("deepset/test-prompt")
    assert not (prompthub_cache_path / "deepset" / "test-prompt.yml").exists()


@pytest.mark.unit
def test_prompt_templates_from_hub_cached_prompts_are_used(
    prompthub_cache_path, enable_prompthub_cache, mock_prompthub
):
    test_path = prompthub_cache_path / "deepset" / "another-test.yml"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    data = prompthub.Prompt(
        name="deepset/another-test",
        text="this is the prompt text",
        description="test prompt description",
        tags=["another-test"],
        meta={"authors": ["vblagoje"]},
        version="v0.1.1",
    )
    data.to_yaml(test_path)

    template = PromptTemplate("deepset/another-test")
    mock_prompthub.fetch.assert_not_called()
    assert template.prompt_text == "this is the prompt text"


@pytest.mark.unit
def test_prompt_templates_from_legacy_set(mock_prompthub):
    p = PromptTemplate("question-answering")
    assert p.name == "question-answering"
    assert p.prompt_text == LEGACY_DEFAULT_TEMPLATES["question-answering"]["prompt"]
    mock_prompthub.assert_not_called()


@pytest.mark.unit
def test_prompt_templates_from_file(tmp_path):
    path = tmp_path / "test-prompt.yml"
    with open(path, "a") as yamlfile:
        yamlfile.write(
            textwrap.dedent(
                """
        name: deepset/question-answering
        text: |
            Given the context please answer the question. Context: {join(documents)};
            Question: {query};
            Answer:
        description: A simple prompt to answer a question given a set of documents
        tags:
        - question-answering
        meta:
        authors:
            - vblagoje
        version: v0.1.1
        """
            )
        )
    p = PromptTemplate(str(path.absolute()))
    assert p.name == "deepset/question-answering"
    assert "Given the context please answer the question" in p.prompt_text


@pytest.mark.unit
def test_prompt_templates_on_the_fly():
    with patch("haystack.nodes.prompt.prompt_template.yaml") as mocked_yaml:
        with patch("haystack.nodes.prompt.prompt_template.prompthub") as mocked_ph:
            p = PromptTemplate("This is a test prompt. Use your knowledge to answer this question: {question}")
            assert p.name == "custom-at-query-time"
            mocked_ph.fetch.assert_not_called()
            mocked_yaml.safe_load.assert_not_called()


@pytest.mark.unit
def test_custom_prompt_templates():
    p = PromptTemplate("Here is some fake template with variable {foo}")
    assert set(p.prompt_params) == {"foo"}

    p = PromptTemplate("Here is some fake template with variable {foo} and {bar}")
    assert set(p.prompt_params) == {"foo", "bar"}

    p = PromptTemplate("Here is some fake template with variable {foo1} and {bar2}")
    assert set(p.prompt_params) == {"foo1", "bar2"}

    p = PromptTemplate("Here is some fake template with variable {foo_1} and {bar_2}")
    assert set(p.prompt_params) == {"foo_1", "bar_2"}

    p = PromptTemplate("Here is some fake template with variable {Foo_1} and {Bar_2}")
    assert set(p.prompt_params) == {"Foo_1", "Bar_2"}

    p = PromptTemplate("'Here is some fake template with variable {baz}'")
    assert set(p.prompt_params) == {"baz"}
    # strip single quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable {baz}"

    p = PromptTemplate('"Here is some fake template with variable {baz}"')
    assert set(p.prompt_params) == {"baz"}
    # strip double quotes, happens in YAML as we need to use single quotes for the template string
    assert p.prompt_text == "Here is some fake template with variable {baz}"


@pytest.mark.unit
def test_missing_prompt_template_params():
    template = PromptTemplate("Here is some fake template with variable {foo} and {bar}")

    # both params provided - ok
    template.prepare(foo="foo", bar="bar")

    # missing one param
    with pytest.raises(ValueError, match=r".*parameters \['bar', 'foo'\] to be provided but got only \['foo'\].*"):
        template.prepare(foo="foo")

    # missing both params
    with pytest.raises(
        ValueError, match=r".*parameters \['bar', 'foo'\] to be provided but got none of these parameters.*"
    ):
        template.prepare(lets="go")

    # more than both params provided - also ok
    template.prepare(foo="foo", bar="bar", lets="go")


@pytest.mark.unit
def test_prompt_template_repr():
    p = PromptTemplate("Here is variable {baz}")
    desired_repr = (
        "PromptTemplate(name=custom-at-query-time, prompt_text=Here is variable {baz}, prompt_params=['baz'])"
    )
    assert repr(p) == desired_repr
    assert str(p) == desired_repr


@pytest.mark.unit
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def test_prompt_template_deserialization(mock_prompt_model):
    custom_prompt_template = PromptTemplate(
        "Given the context please answer the question. Context: {context}; Question: {query}; Answer:",
        output_parser=AnswerParser(),
    )

    prompt_node = PromptNode(default_prompt_template=custom_prompt_template)

    pipe = Pipeline()
    pipe.add_node(component=prompt_node, name="Generator", inputs=["Query"])

    config = pipe.get_config()
    loaded_pipe = Pipeline.load_from_config(config)

    loaded_generator = loaded_pipe.get_node("Generator")
    assert isinstance(loaded_generator, PromptNode)
    assert isinstance(loaded_generator.default_prompt_template, PromptTemplate)
    assert (
        loaded_generator.default_prompt_template.prompt_text
        == "Given the context please answer the question. Context: {context}; Question: {query}; Answer:"
    )
    assert isinstance(loaded_generator.default_prompt_template.output_parser, AnswerParser)


@pytest.mark.unit
def test_prompt_template_fills_in_missing_documents():
    lfqa_prompt = PromptTemplate(
        prompt="""Synthesize a comprehensive answer from the following text for the given question.
        Provide a clear and concise response that summarizes the key points and information presented in the text.
        Your answer should be in your own words and be no longer than 50 words.
        If answer is not in .text. say i dont know.
        \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:"""
    )
    prepared_prompt = next(lfqa_prompt.fill(query="What is the meaning of life?"))  # no documents provided but expected
    assert "Related text:  \n\n Question: What is the meaning of life?" in prepared_prompt


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
        prompt_template = PromptTemplate(prompt_text)
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
            ("context: ", None, None, ["context: "]),
        ],
    )
    def test_prompt_template_syntax_fill(
        self, prompt_text: str, documents: List[Document], query: str, expected_prompts: List[str]
    ):
        prompt_template = PromptTemplate(prompt_text)
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
        prompt_template = PromptTemplate(prompt_text)
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
        prompt_template = PromptTemplate(prompt_text)
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
            PromptTemplate(prompt_text)

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
            prompt_template = PromptTemplate(prompt_text)
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
        prompt_template = PromptTemplate(prompt_text)
        prompts = [prompt for prompt in prompt_template.fill(documents=documents, query=query)]
        assert prompts == expected_prompts

    def test_prompt_template_remove_template_params(self):
        kwargs = {"query": "query", "documents": "documents", "other": "other"}
        expected_kwargs = {"other": "other"}
        prompt_text = "Here is prompt text with two variables that are also in kwargs: {query} and {documents}"
        prompt_template = PromptTemplate(prompt_text)
        assert prompt_template.remove_template_params(kwargs) == expected_kwargs

    def test_prompt_template_remove_template_params_edge_cases(self):
        """
        Test that the function works with a variety of edge cases
        """
        kwargs = {"query": "query", "documents": "documents"}
        prompt_text = "Here is prompt text with two variables that are also in kwargs: {query} and {documents}"
        prompt_template = PromptTemplate(prompt_text)
        assert prompt_template.remove_template_params(kwargs) == {}

        assert prompt_template.remove_template_params({}) == {}

        assert prompt_template.remove_template_params(None) == {}

        totally_unrelated = {"totally_unrelated": "totally_unrelated"}
        assert prompt_template.remove_template_params(totally_unrelated) == totally_unrelated
