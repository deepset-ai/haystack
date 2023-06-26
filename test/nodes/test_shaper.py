from datetime import datetime
import logging

import pytest

import haystack
from haystack import Pipeline, Document, Answer
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes.other.shaper import Shaper
from haystack.nodes.retriever.sparse import BM25Retriever


@pytest.fixture
def mock_function(monkeypatch):
    monkeypatch.setattr(
        haystack.nodes.other.shaper, "REGISTERED_FUNCTIONS", {"test_function": lambda a, b: [a] * len(b)}
    )


@pytest.fixture
def mock_function_two_outputs(monkeypatch):
    monkeypatch.setattr(
        haystack.nodes.other.shaper, "REGISTERED_FUNCTIONS", {"two_output_test_function": lambda a: (a, len(a))}
    )


@pytest.mark.unit
def test_basic_invocation_only_inputs(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "documents"}, outputs=["c"])
    results, _ = shaper.run(query="test query", documents=["doesn't", "really", "matter"])
    assert results["invocation_context"]["c"] == ["test query", "test query", "test query"]


@pytest.mark.unit
def test_basic_invocation_empty_documents_list(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "documents"}, outputs=["c"])
    results, _ = shaper.run(query="test query", documents=[])
    assert results["invocation_context"]["c"] == []


@pytest.mark.unit
def test_multiple_outputs(mock_function_two_outputs):
    shaper = Shaper(func="two_output_test_function", inputs={"a": "query"}, outputs=["c", "d"])
    results, _ = shaper.run(query="test")
    assert results["invocation_context"]["c"] == "test"
    assert results["invocation_context"]["d"] == 4


@pytest.mark.unit
def test_multiple_outputs_error(mock_function_two_outputs, caplog):
    shaper = Shaper(func="two_output_test_function", inputs={"a": "query"}, outputs=["c"])
    with caplog.at_level(logging.WARNING):
        results, _ = shaper.run(query="test")
        assert "Only 1 output(s) will be stored." in caplog.text


@pytest.mark.unit
def test_basic_invocation_only_params(mock_function):
    shaper = Shaper(func="test_function", params={"a": "A", "b": list(range(3))}, outputs=["c"])
    results, _ = shaper.run()
    assert results["invocation_context"]["c"] == ["A", "A", "A"]


@pytest.mark.unit
def test_basic_invocation_inputs_and_params(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query"}, params={"b": list(range(2))}, outputs=["c"])
    results, _ = shaper.run(query="test query")
    assert results["invocation_context"]["c"] == ["test query", "test query"]


@pytest.mark.unit
def test_basic_invocation_inputs_and_params_colliding(mock_function):
    shaper = Shaper(
        func="test_function", inputs={"a": "query"}, params={"a": "default value", "b": list(range(2))}, outputs=["c"]
    )
    results, _ = shaper.run(query="test query")
    assert results["invocation_context"]["c"] == ["test query", "test query"]


@pytest.mark.unit
def test_basic_invocation_inputs_and_params_using_params_as_defaults(mock_function):
    shaper = Shaper(
        func="test_function", inputs={"a": "query"}, params={"a": "default", "b": list(range(2))}, outputs=["c"]
    )
    results, _ = shaper.run()
    assert results["invocation_context"]["c"] == ["default", "default"]


@pytest.mark.unit
def test_missing_argument(mock_function):
    shaper = Shaper(func="test_function", inputs={"b": "documents"}, outputs=["c"])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query="test query", documents=["doesn't", "really", "matter"])


@pytest.mark.unit
def test_excess_argument(mock_function):
    shaper = Shaper(
        func="test_function", inputs={"a": "query", "b": "documents", "something_extra": "query"}, outputs=["c"]
    )
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query="test query", documents=["doesn't", "really", "matter"])


@pytest.mark.unit
def test_value_not_in_invocation_context(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "something_that_does_not_exist"}, outputs=["c"])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query="test query", documents=["doesn't", "really", "matter"])


@pytest.mark.unit
def test_value_only_in_invocation_context(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "invocation_context_specific"}, outputs=["c"])
    results, _s = shaper.run(
        query="test query", invocation_context={"invocation_context_specific": ["doesn't", "really", "matter"]}
    )
    assert results["invocation_context"]["c"] == ["test query", "test query", "test query"]


@pytest.mark.unit
def test_yaml(mock_function, tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: test_function
                inputs:
                  a: query
                params:
                  b: [1, 1]
                outputs:
                  - c
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["c"] == ["test query", "test query"]
    assert result["query"] == "test query"
    assert result["documents"] == [Document(content="first"), Document(content="second"), Document(content="third")]


#
# rename
#


@pytest.mark.unit
def test_rename():
    shaper = Shaper(func="rename", inputs={"value": "query"}, outputs=["questions"])
    results, _ = shaper.run(query="test query")
    assert results["invocation_context"]["questions"] == "test query"


@pytest.mark.unit
def test_rename_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: rename
                inputs:
                  value: query
                outputs:
                  - questions
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="test query")
    assert result["invocation_context"]["query"] == "test query"
    assert result["invocation_context"]["questions"] == "test query"


#
# current_datetime
#


@pytest.mark.unit
def test_current_datetime():
    shaper = Shaper(func="current_datetime", inputs={}, outputs=["date_time"], params={"format": "%y-%m-%d"})
    results, _ = shaper.run()
    assert results["invocation_context"]["date_time"] == datetime.now().strftime("%y-%m-%d")


@pytest.mark.unit
def test_current_datetime_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: current_datetime
                params:
                  format: "%y-%m-%d"
                outputs:
                  - date_time
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run()
    assert result["invocation_context"]["date_time"] == datetime.now().strftime("%y-%m-%d")


#
# value_to_list
#


@pytest.mark.unit
def test_value_to_list():
    shaper = Shaper(func="value_to_list", inputs={"value": "query", "target_list": "documents"}, outputs=["questions"])
    results, _ = shaper.run(query="test query", documents=["doesn't", "really", "matter"])
    assert results["invocation_context"]["questions"] == ["test query", "test query", "test query"]


@pytest.mark.unit
def test_value_to_list_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  value: query
                  target_list: documents
                outputs:
                  - questions
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["questions"] == ["test query", "test query", "test query"]
    # Assert pipeline output is unaffected
    assert result["query"] == "test query"
    assert result["documents"] == [Document(content="first"), Document(content="second"), Document(content="third")]


#
# join_lists
#


@pytest.mark.unit
def test_join_lists():
    shaper = Shaper(func="join_lists", params={"lists": [[1, 2, 3], [4, 5]]}, outputs=["list"])
    results, _ = shaper.run()
    assert results["invocation_context"]["list"] == [1, 2, 3, 4, 5]


@pytest.mark.unit
def test_join_lists_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: join_lists
                inputs:
                  lists:
                   - documents
                   - file_paths
                outputs:
                  - single_list
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=["first", "second", "third"], file_paths=["file1.txt", "file2.txt"])
    assert result["invocation_context"]["single_list"] == ["first", "second", "third", "file1.txt", "file2.txt"]


#
# join_strings
#


@pytest.mark.unit
def test_join_strings():
    shaper = Shaper(
        func="join_strings", params={"strings": ["first", "second"], "delimiter": " | "}, outputs=["single_string"]
    )
    results, _ = shaper.run()
    assert results["invocation_context"]["single_string"] == "first | second"


@pytest.mark.unit
def test_join_strings_default_delimiter():
    shaper = Shaper(func="join_strings", params={"strings": ["first", "second"]}, outputs=["single_string"])
    results, _ = shaper.run()
    assert results["invocation_context"]["single_string"] == "first second"


@pytest.mark.unit
def test_join_strings_with_str_replace():
    shaper = Shaper(
        func="join_strings",
        params={"strings": ["first", "second", "third"], "delimiter": " - ", "str_replace": {"r": "R"}},
        outputs=["single_string"],
    )
    results, _ = shaper.run()
    assert results["invocation_context"]["single_string"] == "fiRst - second - thiRd"


@pytest.mark.unit
def test_join_strings_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: join_strings
                inputs:
                  strings: documents
                params:
                  delimiter: ' - '
                outputs:
                  - single_string
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=["first", "second", "third"])
    assert result["invocation_context"]["single_string"] == "first - second - third"


@pytest.mark.unit
def test_join_strings_default_delimiter_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: join_strings
                inputs:
                  strings: documents
                outputs:
                  - single_string
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=["first", "second", "third"])
    assert result["invocation_context"]["single_string"] == "first second third"


@pytest.mark.unit
def test_join_strings_with_str_replace_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: join_strings
                inputs:
                  strings: documents
                outputs:
                  - single_string
                params:
                  delimiter: ' - '
                  str_replace:
                    r: R
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=["first", "second", "third"])
    assert result["invocation_context"]["single_string"] == "fiRst - second - thiRd"


#
# join_documents
#


@pytest.mark.unit
def test_join_documents():
    shaper = Shaper(
        func="join_documents", inputs={"documents": "documents"}, params={"delimiter": " | "}, outputs=["documents"]
    )
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first | second | third")]
    assert results["documents"] == [Document(content="first | second | third")]


def test_join_documents_without_publish_outputs():
    shaper = Shaper(
        func="join_documents",
        inputs={"documents": "documents"},
        params={"delimiter": " | "},
        outputs=["documents"],
        publish_outputs=False,
    )
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first | second | third")]
    assert "documents" not in results


def test_join_documents_with_publish_outputs_as_list():
    shaper = Shaper(
        func="join_documents",
        inputs={"documents": "documents"},
        params={"delimiter": " | "},
        outputs=["documents"],
        publish_outputs=["documents"],
    )
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first | second | third")]
    assert results["documents"] == [Document(content="first | second | third")]


@pytest.mark.unit
def test_join_documents_default_delimiter():
    shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first second third")]


@pytest.mark.unit
def test_join_documents_with_pattern_and_str_replace():
    shaper = Shaper(
        func="join_documents",
        inputs={"documents": "documents"},
        outputs=["documents"],
        params={"delimiter": " - ", "pattern": "[$idx] $content", "str_replace": {"r": "R"}},
    )
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="[1] fiRst - [2] second - [3] thiRd")]


@pytest.mark.unit
def test_join_documents_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore

            components:
            - name: shaper
              type: Shaper
              params:
                func: join_documents
                inputs:
                  documents: documents
                params:
                  delimiter: ' - '
                outputs:
                  - documents

            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["documents"] == [Document(content="first - second - third")]
    assert result["documents"] == [Document(content="first - second - third")]


@pytest.mark.unit
def test_join_documents_default_delimiter_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: join_documents
                inputs:
                  documents: documents
                outputs:
                  - documents
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["documents"] == [Document(content="first second third")]


@pytest.mark.unit
def test_join_documents_with_pattern_and_str_replace_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: join_documents
                inputs:
                  documents: documents
                outputs:
                  - documents
                params:
                  delimiter: ' - '
                  pattern: '[$idx] $content'
                  str_replace:
                    r: R
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["documents"] == [Document(content="[1] fiRst - [2] second - [3] thiRd")]


#
# strings_to_answers
#


@pytest.mark.unit
def test_strings_to_answers_simple():
    shaper = Shaper(func="strings_to_answers", inputs={"strings": "responses"}, outputs=["answers"])
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"]})
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": None}),
        Answer(answer="second", type="generative", meta={"prompt": None}),
        Answer(answer="third", type="generative", meta={"prompt": None}),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_prompt():
    shaper = Shaper(func="strings_to_answers", inputs={"strings": "responses"}, outputs=["answers"])
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"], "prompts": ["test prompt"]})
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": "test prompt"}),
        Answer(answer="second", type="generative", meta={"prompt": "test prompt"}),
        Answer(answer="third", type="generative", meta={"prompt": "test prompt"}),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_documents():
    shaper = Shaper(func="strings_to_answers", inputs={"strings": "responses"}, outputs=["answers"])
    results, _ = shaper.run(
        invocation_context={
            "responses": ["first", "second", "third"],
            "documents": [Document(id="123", content="test"), Document(id="456", content="test")],
        }
    )
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="second", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="third", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_prompt_per_document():
    shaper = Shaper(func="strings_to_answers", inputs={"strings": "responses"}, outputs=["answers"])
    results, _ = shaper.run(
        invocation_context={
            "responses": ["first", "second"],
            "documents": [Document(id="123", content="test"), Document(id="456", content="test")],
            "prompts": ["prompt1", "prompt2"],
        }
    )
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": "prompt1"}, document_ids=["123"]),
        Answer(answer="second", type="generative", meta={"prompt": "prompt2"}, document_ids=["456"]),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_prompt_per_document_multiple_results():
    shaper = Shaper(func="strings_to_answers", inputs={"strings": "responses"}, outputs=["answers"])
    results, _ = shaper.run(
        invocation_context={
            "responses": ["first", "second", "third", "fourth"],
            "documents": [Document(id="123", content="test"), Document(id="456", content="test")],
            "prompts": ["prompt1", "prompt2"],
        }
    )
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": "prompt1"}, document_ids=["123"]),
        Answer(answer="second", type="generative", meta={"prompt": "prompt1"}, document_ids=["123"]),
        Answer(answer="third", type="generative", meta={"prompt": "prompt2"}, document_ids=["456"]),
        Answer(answer="fourth", type="generative", meta={"prompt": "prompt2"}, document_ids=["456"]),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_pattern_group():
    shaper = Shaper(
        func="strings_to_answers",
        inputs={"strings": "responses"},
        outputs=["answers"],
        params={"pattern": r"Answer: (.*)"},
    )
    results, _ = shaper.run(invocation_context={"responses": ["Answer: first", "Answer: second", "Answer: third"]})
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": None}),
        Answer(answer="second", type="generative", meta={"prompt": None}),
        Answer(answer="third", type="generative", meta={"prompt": None}),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_pattern_no_group():
    shaper = Shaper(
        func="strings_to_answers", inputs={"strings": "responses"}, outputs=["answers"], params={"pattern": r"[^\n]+$"}
    )
    results, _ = shaper.run(invocation_context={"responses": ["Answer\nfirst", "Answer\nsecond", "Answer\n\nthird"]})
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first", type="generative", meta={"prompt": None}),
        Answer(answer="second", type="generative", meta={"prompt": None}),
        Answer(answer="third", type="generative", meta={"prompt": None}),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_references_index():
    shaper = Shaper(
        func="strings_to_answers",
        inputs={"strings": "responses", "documents": "documents"},
        outputs=["answers"],
        params={"reference_pattern": r"\[(\d+)\]"},
    )
    results, _ = shaper.run(
        invocation_context={
            "responses": ["first[1]", "second[2]", "third[1][2]", "fourth"],
            "documents": [Document(id="123", content="test"), Document(id="456", content="test")],
        }
    )
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first[1]", type="generative", meta={"prompt": None}, document_ids=["123"]),
        Answer(answer="second[2]", type="generative", meta={"prompt": None}, document_ids=["456"]),
        Answer(answer="third[1][2]", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="fourth", type="generative", meta={"prompt": None}, document_ids=[]),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_references_id():
    shaper = Shaper(
        func="strings_to_answers",
        inputs={"strings": "responses", "documents": "documents"},
        outputs=["answers"],
        params={"reference_pattern": r"\[(\d+)\]", "reference_mode": "id"},
    )
    results, _ = shaper.run(
        invocation_context={
            "responses": ["first[123]", "second[456]", "third[123][456]", "fourth"],
            "documents": [Document(id="123", content="test"), Document(id="456", content="test")],
        }
    )
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first[123]", type="generative", meta={"prompt": None}, document_ids=["123"]),
        Answer(answer="second[456]", type="generative", meta={"prompt": None}, document_ids=["456"]),
        Answer(answer="third[123][456]", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="fourth", type="generative", meta={"prompt": None}, document_ids=[]),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_references_meta():
    shaper = Shaper(
        func="strings_to_answers",
        inputs={"strings": "responses", "documents": "documents"},
        outputs=["answers"],
        params={"reference_pattern": r"\[([^\]]+)\]", "reference_mode": "meta", "reference_meta_field": "file_id"},
    )
    results, _ = shaper.run(
        invocation_context={
            "responses": ["first[123.txt]", "second[456.txt]", "third[123.txt][456.txt]", "fourth"],
            "documents": [
                Document(id="123", content="test", meta={"file_id": "123.txt"}),
                Document(id="456", content="test", meta={"file_id": "456.txt"}),
            ],
        }
    )
    assert results["invocation_context"]["answers"] == [
        Answer(answer="first[123.txt]", type="generative", meta={"prompt": None}, document_ids=["123"]),
        Answer(answer="second[456.txt]", type="generative", meta={"prompt": None}, document_ids=["456"]),
        Answer(answer="third[123.txt][456.txt]", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="fourth", type="generative", meta={"prompt": None}, document_ids=[]),
    ]


@pytest.mark.unit
def test_strings_to_answers_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: strings_to_answers
                params:
                  strings: ['a', 'b', 'c']
                outputs:
                  - answers
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run()
    assert result["invocation_context"]["answers"] == [
        Answer(answer="a", type="generative", meta={"prompt": None}),
        Answer(answer="b", type="generative", meta={"prompt": None}),
        Answer(answer="c", type="generative", meta={"prompt": None}),
    ]
    assert result["answers"] == [
        Answer(answer="a", type="generative", meta={"prompt": None}),
        Answer(answer="b", type="generative", meta={"prompt": None}),
        Answer(answer="c", type="generative", meta={"prompt": None}),
    ]


@pytest.mark.unit
def test_strings_to_answers_with_reference_meta_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: strings_to_answers
                inputs:
                  documents: documents
                params:
                  reference_meta_field: file_id
                  reference_mode: meta
                  reference_pattern: \[([^\]]+)\]
                  strings: ['first[123.txt]', 'second[456.txt]', 'third[123.txt][456.txt]', 'fourth']
                outputs:
                  - answers
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        documents=[
            Document(id="123", content="test", meta={"file_id": "123.txt"}),
            Document(id="456", content="test", meta={"file_id": "456.txt"}),
        ]
    )
    assert result["invocation_context"]["answers"] == [
        Answer(answer="first[123.txt]", type="generative", meta={"prompt": None}, document_ids=["123"]),
        Answer(answer="second[456.txt]", type="generative", meta={"prompt": None}, document_ids=["456"]),
        Answer(answer="third[123.txt][456.txt]", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="fourth", type="generative", meta={"prompt": None}, document_ids=[]),
    ]
    assert result["answers"] == [
        Answer(answer="first[123.txt]", type="generative", meta={"prompt": None}, document_ids=["123"]),
        Answer(answer="second[456.txt]", type="generative", meta={"prompt": None}, document_ids=["456"]),
        Answer(answer="third[123.txt][456.txt]", type="generative", meta={"prompt": None}, document_ids=["123", "456"]),
        Answer(answer="fourth", type="generative", meta={"prompt": None}, document_ids=[]),
    ]


@pytest.mark.integration
def test_strings_to_answers_after_prompt_node_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: prompt_model
                type: PromptModel

              - name: prompt_template_raw_qa_per_document
                type: PromptTemplate
                params:
                  prompt: 'Given the context please answer the question. Context: {{documents}}; Question: {{query}}; Answer:'

              - name: prompt_node_raw_qa
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: prompt_template_raw_qa_per_document
                  top_k: 2

              - name: prompt_node_question_generation
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: question-generation
                  output_variable: query

              - name: shaper
                type: Shaper
                params:
                  func: strings_to_answers
                  inputs:
                    strings: results
                  outputs:
                    - answers


            pipelines:
              - name: query
                nodes:
                  - name: prompt_node_question_generation
                    inputs:
                      - Query
                  - name: prompt_node_raw_qa
                    inputs:
                      - prompt_node_question_generation
                  - name: shaper
                    inputs:
                      - prompt_node_raw_qa
            """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="What's Berlin like?",
        documents=[
            Document("Berlin is an amazing city.", id="123"),
            Document("Berlin is a cool city in Germany.", id="456"),
        ],
    )
    results = result["answers"]
    assert len(results) == 4
    assert any([True for r in results if "Berlin" in r.answer])
    for answer in results[:2]:
        assert answer.document_ids == ["123"]
        assert (
            answer.meta["prompt"]
            == f"Given the context please answer the question. Context: Berlin is an amazing city.; Question: {result['query'][0]}; Answer:"
        )
    for answer in results[2:]:
        assert answer.document_ids == ["456"]
        assert (
            answer.meta["prompt"]
            == f"Given the context please answer the question. Context: Berlin is a cool city in Germany.; Question: {result['query'][1]}; Answer:"
        )


#
# answers_to_strings
#


@pytest.mark.unit
def test_answers_to_strings():
    shaper = Shaper(func="answers_to_strings", inputs={"answers": "documents"}, outputs=["strings"])
    results, _ = shaper.run(documents=[Answer(answer="first"), Answer(answer="second"), Answer(answer="third")])
    assert results["invocation_context"]["strings"] == ["first", "second", "third"]


@pytest.mark.unit
def test_answers_to_strings_with_pattern_and_str_replace():
    shaper = Shaper(
        func="answers_to_strings",
        inputs={"answers": "documents"},
        outputs=["strings"],
        params={"pattern": "[$idx] $answer", "str_replace": {"r": "R"}},
    )
    results, _ = shaper.run(documents=[Answer(answer="first"), Answer(answer="second"), Answer(answer="third")])
    assert results["invocation_context"]["strings"] == ["[1] fiRst", "[2] second", "[3] thiRd"]


@pytest.mark.unit
def test_answers_to_strings_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: answers_to_strings
                inputs:
                  answers: documents
                outputs:
                  - strings
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=[Answer(answer="a"), Answer(answer="b"), Answer(answer="c")])
    assert result["invocation_context"]["strings"] == ["a", "b", "c"]


@pytest.mark.unit
def test_answers_to_strings_with_pattern_and_str_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: answers_to_strings
                inputs:
                  answers: documents
                outputs:
                  - strings
                params:
                  pattern: '[$idx] $answer'
                  str_replace:
                    r: R
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=[Answer(answer="first"), Answer(answer="second"), Answer(answer="third")])
    assert result["invocation_context"]["strings"] == ["[1] fiRst", "[2] second", "[3] thiRd"]


#
# strings_to_documents
#


@pytest.mark.unit
def test_strings_to_documents_no_meta_no_hashkeys():
    shaper = Shaper(func="strings_to_documents", inputs={"strings": "responses"}, outputs=["documents"])
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"]})
    assert results["invocation_context"]["documents"] == [
        Document(content="first"),
        Document(content="second"),
        Document(content="third"),
    ]


@pytest.mark.unit
def test_strings_to_documents_single_meta_no_hashkeys():
    shaper = Shaper(
        func="strings_to_documents", inputs={"strings": "responses"}, params={"meta": {"a": "A"}}, outputs=["documents"]
    )
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"]})
    assert results["invocation_context"]["documents"] == [
        Document(content="first", meta={"a": "A"}),
        Document(content="second", meta={"a": "A"}),
        Document(content="third", meta={"a": "A"}),
    ]


@pytest.mark.unit
def test_strings_to_documents_wrong_number_of_meta():
    shaper = Shaper(
        func="strings_to_documents",
        inputs={"strings": "responses"},
        params={"meta": [{"a": "A"}]},
        outputs=["documents"],
    )

    with pytest.raises(ValueError, match="Not enough metadata dictionaries."):
        shaper.run(invocation_context={"responses": ["first", "second", "third"]})


@pytest.mark.unit
def test_strings_to_documents_many_meta_no_hashkeys():
    shaper = Shaper(
        func="strings_to_documents",
        inputs={"strings": "responses"},
        params={"meta": [{"a": i + 1} for i in range(3)]},
        outputs=["documents"],
    )
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"]})
    assert results["invocation_context"]["documents"] == [
        Document(content="first", meta={"a": 1}),
        Document(content="second", meta={"a": 2}),
        Document(content="third", meta={"a": 3}),
    ]


@pytest.mark.unit
def test_strings_to_documents_single_meta_with_hashkeys():
    shaper = Shaper(
        func="strings_to_documents",
        inputs={"strings": "responses"},
        params={"meta": {"a": "A"}, "id_hash_keys": ["content", "meta"]},
        outputs=["documents"],
    )
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"]})
    assert results["invocation_context"]["documents"] == [
        Document(content="first", meta={"a": "A"}, id_hash_keys=["content", "meta"]),
        Document(content="second", meta={"a": "A"}, id_hash_keys=["content", "meta"]),
        Document(content="third", meta={"a": "A"}, id_hash_keys=["content", "meta"]),
    ]


@pytest.mark.unit
def test_strings_to_documents_no_meta_no_hashkeys_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: strings_to_documents
                params:
                  strings: ['a', 'b', 'c']
                outputs:
                  - documents
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run()
    assert result["invocation_context"]["documents"] == [
        Document(content="a"),
        Document(content="b"),
        Document(content="c"),
    ]


@pytest.mark.unit
def test_strings_to_documents_meta_and_hashkeys_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: strings_to_documents
                params:
                  strings: ['first', 'second', 'third']
                  id_hash_keys: ['content', 'meta']
                  meta:
                    - a: 1
                    - a: 2
                    - a: 3
                outputs:
                  - documents
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run()
    assert result["invocation_context"]["documents"] == [
        Document(content="first", meta={"a": 1}, id_hash_keys=["content", "meta"]),
        Document(content="second", meta={"a": 2}, id_hash_keys=["content", "meta"]),
        Document(content="third", meta={"a": 3}, id_hash_keys=["content", "meta"]),
    ]


#
# documents_to_strings
#


@pytest.mark.unit
def test_documents_to_strings():
    shaper = Shaper(func="documents_to_strings", inputs={"documents": "documents"}, outputs=["strings"])
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["strings"] == ["first", "second", "third"]


@pytest.mark.unit
def test_documents_to_strings_with_pattern_and_str_replace():
    shaper = Shaper(
        func="documents_to_strings",
        inputs={"documents": "documents"},
        outputs=["strings"],
        params={"pattern": "[$idx] $content", "str_replace": {"r": "R"}},
    )
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["strings"] == ["[1] fiRst", "[2] second", "[3] thiRd"]


@pytest.mark.unit
def test_documents_to_strings_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: documents_to_strings
                inputs:
                  documents: documents
                outputs:
                  - strings
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=[Document(content="a"), Document(content="b"), Document(content="c")])
    assert result["invocation_context"]["strings"] == ["a", "b", "c"]


@pytest.mark.unit
def test_documents_to_strings_with_pattern_and_str_replace_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              type: Shaper
              params:
                func: documents_to_strings
                inputs:
                  documents: documents
                outputs:
                  - strings
                params:
                  pattern: '[$idx] $content'
                  str_replace:
                    r: R
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(documents=[Document(content="first"), Document(content="second"), Document(content="third")])
    assert result["invocation_context"]["strings"] == ["[1] fiRst", "[2] second", "[3] thiRd"]


#
# Chaining and real-world usage
#


@pytest.mark.unit
def test_chain_shapers():
    shaper_1 = Shaper(
        func="join_documents", inputs={"documents": "documents"}, params={"delimiter": " - "}, outputs=["documents"]
    )
    shaper_2 = Shaper(
        func="value_to_list", inputs={"value": "query", "target_list": "documents"}, outputs=["questions"]
    )

    pipe = Pipeline()
    pipe.add_node(shaper_1, name="shaper_1", inputs=["Query"])
    pipe.add_node(shaper_2, name="shaper_2", inputs=["shaper_1"])

    results = pipe.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )

    assert results["invocation_context"]["documents"] == [Document(content="first - second - third")]
    assert results["invocation_context"]["questions"] == ["test query"]


@pytest.mark.unit
def test_chain_shapers_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:

            - name: shaper_1
              type: Shaper
              params:
                func: join_documents
                inputs:
                  documents: documents
                params:
                  delimiter: ' - '
                outputs:
                  - documents

            - name: shaper_2
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  value: query
                  target_list: documents
                outputs:
                  - questions

            pipelines:
              - name: query
                nodes:
                  - name: shaper_1
                    inputs:
                      - Query
                  - name: shaper_2
                    inputs:
                      - shaper_1
        """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")

    results = pipe.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )

    assert results["invocation_context"]["documents"] == [Document(content="first - second - third")]
    assert results["invocation_context"]["questions"] == ["test query"]


@pytest.mark.unit
def test_chain_shapers_yaml_2(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:

            - name: shaper_1
              type: Shaper
              params:
                func: strings_to_documents
                params:
                  strings:
                    - first
                    - second
                    - third
                outputs:
                  - string_documents

            - name: shaper_2
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  target_list: string_documents
                params:
                  value: hello
                outputs:
                  - greetings

            - name: shaper_3
              type: Shaper
              params:
                func: join_strings
                inputs:
                  strings: greetings
                params:
                  delimiter: '. '
                outputs:
                  - many_greetings

            - name: expander
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  value: many_greetings
                params:
                  target_list: [1]
                outputs:
                  - many_greetings

            - name: shaper_4
              type: Shaper
              params:
                func: strings_to_documents
                inputs:
                  strings: many_greetings
                outputs:
                  - documents_with_greetings

            pipelines:
              - name: query
                nodes:
                  - name: shaper_1
                    inputs:
                      - Query
                  - name: shaper_2
                    inputs:
                      - shaper_1
                  - name: shaper_3
                    inputs:
                      - shaper_2
                  - name: expander
                    inputs:
                      - shaper_3
                  - name: shaper_4
                    inputs:
                      - expander
        """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    results = pipe.run()
    assert results["invocation_context"]["documents_with_greetings"] == [Document(content="hello. hello. hello")]


@pytest.mark.integration
def test_with_prompt_node(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: prompt_model
                type: PromptModel

              - name: prompt_node
                type: PromptNode
                params:
                  output_variable: answers
                  model_name_or_path: prompt_model
                  default_prompt_template: question-answering-per-document

            pipelines:
              - name: query
                nodes:
                  - name: prompt_node
                    inputs:
                      - Query
            """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="What's Berlin like?",
        documents=[Document("Berlin is an amazing city."), Document("Berlin is a cool city in Germany.")],
    )
    assert len(result["answers"]) == 2
    raw_answers = [answer.answer for answer in result["answers"]]
    assert any(word for word in ["berlin", "germany", "cool", "city", "amazing"] if word in raw_answers)


@pytest.mark.integration
def test_with_multiple_prompt_nodes(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: prompt_model
                type: PromptModel

              - name: renamer
                type: Shaper
                params:
                  func: rename
                  inputs:
                    value: new-questions
                  outputs:
                    - query

              - name: prompt_node
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: question-answering-per-document

              - name: prompt_node_second
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: question-generation
                  output_variable: new-questions

              - name: prompt_node_third
                type: PromptNode
                params:
                  model_name_or_path: google/flan-t5-small
                  default_prompt_template: question-answering-per-document

            pipelines:
              - name: query
                nodes:
                  - name: prompt_node
                    inputs:
                      - Query
                  - name: prompt_node_second
                    inputs:
                      - prompt_node
                  - name: renamer
                    inputs:
                      - prompt_node_second
                  - name: prompt_node_third
                    inputs:
                      - renamer
            """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="What's Berlin like?",
        documents=[Document("Berlin is an amazing city."), Document("Berlin is a cool city in Germany.")],
    )
    results = result["answers"]
    assert len(results) == 2
    assert any([True for r in results if "Berlin" in r.answer])


@pytest.mark.unit
def test_join_query_and_documents_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore

            components:
            - name: expander
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  value: query
                params:
                  target_list: [1]
                outputs:
                  - query

            - name: joiner
              type: Shaper
              params:
                func: join_lists
                inputs:
                  lists:
                   - documents
                   - query
                outputs:
                  - query

            pipelines:
              - name: query
                nodes:
                  - name: expander
                    inputs:
                      - Query
                  - name: joiner
                    inputs:
                      - expander
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="What is going on here?", documents=["first", "second", "third"])
    assert result["query"] == ["first", "second", "third", "What is going on here?"]


@pytest.mark.unit
def test_join_query_and_documents_into_single_string_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: expander
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  value: query
                params:
                  target_list: [1]
                outputs:
                  - query

            - name: joiner
              type: Shaper
              params:
                func: join_lists
                inputs:
                  lists:
                   - documents
                   - query
                outputs:
                  - query

            - name: concatenator
              type: Shaper
              params:
                func: join_strings
                inputs:
                  strings: query
                outputs:
                  - query

            pipelines:
              - name: query
                nodes:
                  - name: expander
                    inputs:
                      - Query
                  - name: joiner
                    inputs:
                      - expander
                  - name: concatenator
                    inputs:
                      - joiner
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="What is going on here?", documents=["first", "second", "third"])
    assert result["query"] == "first second third What is going on here?"


@pytest.mark.unit
def test_join_query_and_documents_convert_into_documents_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: expander
              type: Shaper
              params:
                func: value_to_list
                inputs:
                  value: query
                params:
                  target_list: [1]
                outputs:
                  - query

            - name: joiner
              type: Shaper
              params:
                func: join_lists
                inputs:
                  lists:
                   - documents
                   - query
                outputs:
                  - query_and_docs

            - name: converter
              type: Shaper
              params:
                func: strings_to_documents
                inputs:
                  strings: query_and_docs
                outputs:
                  - query_and_docs

            pipelines:
              - name: query
                nodes:
                  - name: expander
                    inputs:
                      - Query
                  - name: joiner
                    inputs:
                      - expander
                  - name: converter
                    inputs:
                      - joiner
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(query="What is going on here?", documents=["first", "second", "third"])
    assert result["invocation_context"]["query_and_docs"]
    assert len(result["invocation_context"]["query_and_docs"]) == 4
    assert isinstance(result["invocation_context"]["query_and_docs"][0], Document)


@pytest.mark.unit
def test_shaper_publishes_unknown_arg_does_not_break_pipeline():
    documents = [Document(content="test query")]
    shaper = Shaper(func="rename", inputs={"value": "query"}, outputs=["unknown_by_retriever"], publish_outputs=True)
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(documents)
    retriever = BM25Retriever(document_store=document_store)
    pipeline = Pipeline()
    pipeline.add_node(component=shaper, name="shaper", inputs=["Query"])
    pipeline.add_node(component=retriever, name="retriever", inputs=["shaper"])

    result = pipeline.run(query="test query")
    assert result["invocation_context"]["unknown_by_retriever"] == "test query"
    assert result["unknown_by_retriever"] == "test query"
    assert len(result["documents"]) == 1
