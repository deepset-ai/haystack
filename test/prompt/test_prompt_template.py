from typing import Set, Type, List

import pytest

from haystack.nodes.prompt import PromptTemplate
from haystack.nodes.prompt.prompt_template import PromptTemplateValidationError
from haystack.schema import Answer, Document


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
