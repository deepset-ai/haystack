import re
import pytest

import haystack
from haystack import Pipeline, Document, Answer
from haystack.nodes.other.shaper import Shaper


@pytest.fixture
def mock_function(monkeypatch):
    monkeypatch.setattr(
        haystack.nodes.other.shaper, "REGISTERED_FUNCTIONS", {"test_function": lambda a, b: ([a] * len(b),)}
    )


def test_basic_invocation_only_inputs(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "documents"}, outputs=["c"])
    results, _ = shaper.run(query="test query", documents=["doesn't", "really", "matter"])
    assert results["invocation_context"]["c"] == ["test query", "test query", "test query"]


def test_basic_invocation_only_params(mock_function):
    shaper = Shaper(func="test_function", params={"a": "A", "b": list(range(3))}, outputs=["c"])
    results, _ = shaper.run()
    assert results["invocation_context"]["c"] == ["A", "A", "A"]


def test_basic_invocation_inputs_and_params(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query"}, params={"b": list(range(2))}, outputs=["c"])
    results, _ = shaper.run(query="test query")
    assert results["invocation_context"]["c"] == ["test query", "test query"]


def test_basic_invocation_inputs_and_params_colliding(mock_function):
    shaper = Shaper(
        func="test_function", inputs={"a": "query"}, params={"a": "default value", "b": list(range(2))}, outputs=["c"]
    )
    results, _ = shaper.run(query="test query")
    assert results["invocation_context"]["c"] == ["test query", "test query"]


def test_basic_invocation_inputs_and_params_using_params_as_defaults(mock_function):
    shaper = Shaper(
        func="test_function", inputs={"a": "query"}, params={"a": "default", "b": list(range(2))}, outputs=["c"]
    )
    results, _ = shaper.run()
    assert results["invocation_context"]["c"] == ["default", "default"]


def test_missing_argument(mock_function):
    shaper = Shaper(func="test_function", inputs={"b": "documents"}, outputs=["c"])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query="test query", documents=["doesn't", "really", "matter"])


def test_excess_argument(mock_function):
    shaper = Shaper(
        func="test_function", inputs={"a": "query", "b": "documents", "something_extra": "query"}, outputs=["c"]
    )
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query="test query", documents=["doesn't", "really", "matter"])


def test_value_not_in_invocation_context(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "something_that_does_not_exist"}, outputs=["c"])
    with pytest.raises(ValueError, match="Shaper couldn't apply the function to your inputs and parameters."):
        shaper.run(query="test query", documents=["doesn't", "really", "matter"])


def test_value_only_in_invocation_context(mock_function):
    shaper = Shaper(func="test_function", inputs={"a": "query", "b": "invocation_context_specific"}, outputs=["c"])
    results, _s = shaper.run(
        query="test query", invocation_context={"invocation_context_specific": ["doesn't", "really", "matter"]}
    )
    assert results["invocation_context"]["c"] == ["test query", "test query", "test query"]


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


def test_rename():
    shaper = Shaper(func="rename", inputs={"value": "query"}, outputs=["questions"])
    results, _ = shaper.run(query="test query")
    assert results["invocation_context"]["questions"] == "test query"


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
# value_to_list
#


def test_value_to_list():
    shaper = Shaper(func="value_to_list", inputs={"value": "query", "target_list": "documents"}, outputs=["questions"])
    results, _ = shaper.run(query="test query", documents=["doesn't", "really", "matter"])
    assert results["invocation_context"]["questions"] == ["test query", "test query", "test query"]


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


def test_join_lists():
    shaper = Shaper(func="join_lists", params={"lists": [[1, 2, 3], [4, 5]]}, outputs=["list"])
    results, _ = shaper.run()
    assert results["invocation_context"]["list"] == [1, 2, 3, 4, 5]


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


def test_join_strings():
    shaper = Shaper(
        func="join_strings", params={"strings": ["first", "second"], "delimiter": " | "}, outputs=["single_string"]
    )
    results, _ = shaper.run()
    assert results["invocation_context"]["single_string"] == ["first | second"]


def test_join_strings_default_delimiter():
    shaper = Shaper(func="join_strings", params={"strings": ["first", "second"]}, outputs=["single_string"])
    results, _ = shaper.run()
    assert results["invocation_context"]["single_string"] == ["first second"]


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
    assert result["invocation_context"]["single_string"] == ["first - second - third"]


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
    assert result["invocation_context"]["single_string"] == ["first second third"]


#
# join_documents
#


def test_join_documents():
    shaper = Shaper(
        func="join_documents", inputs={"documents": "documents"}, params={"delimiter": " | "}, outputs=["documents"]
    )
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first | second | third")]


def test_join_documents_default_delimiter():
    shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first second third")]


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
    assert result["documents"] == [Document(content="first"), Document(content="second"), Document(content="third")]


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


#
# strings_to_documents
#


def test_strings_to_documents_no_meta_no_hashkeys():
    shaper = Shaper(func="strings_to_documents", inputs={"strings": "responses"}, outputs=["documents"])
    results, _ = shaper.run(invocation_context={"responses": ["first", "second", "third"]})
    assert results["invocation_context"]["documents"] == [
        Document(content="first"),
        Document(content="second"),
        Document(content="third"),
    ]


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


def test_strings_to_documents_wrong_number_of_meta():
    shaper = Shaper(
        func="strings_to_documents",
        inputs={"strings": "responses"},
        params={"meta": [{"a": "A"}]},
        outputs=["documents"],
    )

    with pytest.raises(ValueError, match="Not enough metadata dictionaries."):
        shaper.run(invocation_context={"responses": ["first", "second", "third"]})


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


def test_documents_to_strings():
    shaper = Shaper(func="documents_to_strings", inputs={"documents": "documents"}, outputs=["strings"])
    results, _ = shaper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["strings"] == ["first", "second", "third"]


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


#
# Chaining and real-world usage
#


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
                  - name: shaper_4
                    inputs:
                      - shaper_3
        """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")

    results = pipe.run()

    assert results["invocation_context"]["documents_with_greetings"] == [Document(content="hello. hello. hello")]


def test_with_prompt_node(tmp_path):

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: prompt_model
                type: PromptModel

              - name: shaper
                type: Shaper
                params:
                  func: value_to_list
                  inputs:
                    value: query
                    target_list: documents
                  outputs:
                    - questions

              - name: prompt_node
                type: PromptNode
                params:
                  output_variable: answers
                  model_name_or_path: prompt_model
                  default_prompt_template: question-answering

            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
                  - name: prompt_node
                    inputs:
                      - shaper
            """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="What's Berlin like?",
        documents=[Document("Berlin is an amazing city."), Document("Berlin is a cool city in Germany.")],
    )
    assert len(result["answers"]) == 2
    assert any(word for word in ["berlin", "germany", "cool", "city", "amazing"] if word in result["answers"])

    assert len(result["invocation_context"]) > 0
    assert len(result["invocation_context"]["questions"]) == 2


def test_with_multiple_prompt_nodes(tmp_path):

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: prompt_model
                type: PromptModel

              - name: shaper
                type: Shaper
                params:
                  func: value_to_list
                  inputs:
                    value: query
                    target_list: documents
                  outputs: [questions]

              - name: renamer
                type: Shaper
                params:
                  func: rename
                  inputs:
                    value: new-questions
                  outputs:
                    - questions

              - name: prompt_node
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: question-answering

              - name: prompt_node_second
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: question-generation
                  output_variable: new-questions

              - name: prompt_node_third
                type: PromptNode
                params:
                  output_variable: answers
                  model_name_or_path: google/flan-t5-small
                  default_prompt_template: question-answering

            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
                  - name: prompt_node
                    inputs:
                      - shaper
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
    assert any([True for r in results if "Berlin" in r])
