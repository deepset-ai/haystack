import re
import pytest

import haystack
from haystack import Pipeline, Document
from haystack.nodes.prompt.invocation_context_mapper import InvocationContextMapper


@pytest.fixture
def mock_function(monkeypatch):
    monkeypatch.setattr(
        haystack.nodes.prompt.invocation_context_mapper,
        "REGISTERED_FUNCTIONS",
        {"test_function": lambda a, b: ([a] * len(b),)},
    )


def test_basic_invocation(mock_function):
    mapper = InvocationContextMapper(func="test_function", inputs={"a": "query", "b": "documents"}, outputs=["c"])
    results, _ = mapper.run(query="test query", documents=["doesn't", "really", "matter"])
    assert results["invocation_context"]["c"] == ["test query", "test query", "test query"]


def test_missing_argument(mock_function):
    mapper = InvocationContextMapper(func="test_function", inputs={"b": "documents"}, outputs=["c"])
    with pytest.raises(
        ValueError, match="InvocationContextMapper could not apply the function to your inputs and parameters."
    ):
        mapper.run(query="test query", documents=["doesn't", "really", "matter"])


def test_excess_argument(mock_function):
    mapper = InvocationContextMapper(
        func="test_function", inputs={"a": "query", "b": "documents", "something_extra": "query"}, outputs=["c"]
    )
    with pytest.raises(
        ValueError, match="InvocationContextMapper could not apply the function to your inputs and parameters."
    ):
        mapper.run(query="test query", documents=["doesn't", "really", "matter"])


def test_value_not_in_invocation_context(mock_function):
    mapper = InvocationContextMapper(
        func="test_function", inputs={"a": "query", "b": "something_that_does_not_exist"}, outputs=["c"]
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "InvocationContextMapper could not find these values from your inputs list in the invocation context: ['something_that_does_not_exist']"
        ),
    ):
        mapper.run(query="test query", documents=["doesn't", "really", "matter"])


def test_value_only_in_invocation_context(mock_function):
    mapper = InvocationContextMapper(
        func="test_function", inputs={"a": "query", "b": "invocation_context_specific"}, outputs=["c"]
    )
    results, _s = mapper.run(
        query="test query", invocation_context={"invocation_context_specific": ["doesn't", "really", "matter"]}
    )
    assert results["invocation_context"]["c"] == ["test query", "test query", "test query"]


#
# expand_values_to_list
#


def test_expand_values_to_list():
    mapper = InvocationContextMapper(
        func="expand_value_to_list", inputs={"value": "query", "target_list": "documents"}, outputs=["questions"]
    )
    results, _ = mapper.run(query="test query", documents=["doesn't", "really", "matter"])
    assert results["invocation_context"]["questions"] == ["test query", "test query", "test query"]


def test_expand_values_to_list_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: mapper
              type: InvocationContextMapper
              params:
                func: expand_value_to_list
                inputs:
                  value: query
                  target_list: documents
                outputs:
                  - questions
            pipelines:
              - name: query
                nodes:
                  - name: mapper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["questions"] == ["test query", "test query", "test query"]


#
# join_documents
#


def test_join_documents():
    mapper = InvocationContextMapper(
        func="join_documents", inputs={"documents": "documents"}, params={"delimiter": " | "}, outputs=["documents"]
    )
    results, _ = mapper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first | second | third")]

    # Only the invocation_context is affected. It's going to be PromptNode that puts the documents back in the Pipeline
    assert results["documents"] == [Document(content="first"), Document(content="second"), Document(content="third")]


def test_join_documents_default_delimiter():
    mapper = InvocationContextMapper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])
    results, _ = mapper.run(
        documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert results["invocation_context"]["documents"] == [Document(content="first second third")]


def test_join_documents_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: mapper
              type: InvocationContextMapper
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
                  - name: mapper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )
    assert result["invocation_context"]["documents"] == [Document(content="first - second - third")]


def test_join_documents_default_delimiter_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: mapper
              type: InvocationContextMapper
              params:
                func: join_documents
                inputs:
                  documents: documents
                outputs:
                  - documents
            pipelines:
              - name: query
                nodes:
                  - name: mapper
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
# Chaining and real-world usage
#


def test_chain_mappers():
    mapper_1 = InvocationContextMapper(
        func="join_documents", inputs={"documents": "documents"}, params={"delimiter": " - "}, outputs=["documents"]
    )
    mapper_2 = InvocationContextMapper(
        func="expand_value_to_list", inputs={"value": "query", "target_list": "documents"}, outputs=["questions"]
    )

    pipe = Pipeline()
    pipe.add_node(mapper_1, name="mapper_1", inputs=["Query"])
    pipe.add_node(mapper_2, name="mapper_2", inputs=["mapper_1"])

    results = pipe.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )

    assert results["invocation_context"]["documents"] == [Document(content="first - second - third")]
    assert results["invocation_context"]["questions"] == ["test query"]


def test_chain_mappers_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:

            - name: mapper_1
              type: InvocationContextMapper
              params:
                func: join_documents
                inputs:
                  documents: documents
                params:
                  delimiter: ' - '
                outputs:
                  - documents

            - name: mapper_2
              type: InvocationContextMapper
              params:
                func: expand_value_to_list
                inputs:
                  value: query
                  target_list: documents
                outputs:
                  - questions

            pipelines:
              - name: query
                nodes:
                  - name: mapper_1
                    inputs:
                      - Query
                  - name: mapper_2
                    inputs:
                      - mapper_1
        """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")

    results = pipe.run(
        query="test query", documents=[Document(content="first"), Document(content="second"), Document(content="third")]
    )

    assert results["invocation_context"]["documents"] == [Document(content="first - second - third")]
    assert results["invocation_context"]["questions"] == ["test query"]


def test_with_prompt_node(tmp_path):

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: prompt_model
                type: PromptModel

              - name: mapper
                type: InvocationContextMapper
                params:
                  func: expand_value_to_list
                  inputs:
                    value: query
                    target_list: documents
                  outputs: [questions]

              - name: prompt_node
                type: PromptNode
                params:
                  model_name_or_path: prompt_model
                  default_prompt_template: question-answering

            pipelines:
              - name: query
                nodes:
                  - name: mapper
                    inputs:
                      - Query
                  - name: prompt_node
                    inputs:
                      - mapper
            """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run(
        query="What's Berlin like?",
        documents=[Document("Berlin is an amazing city."), Document("Berlin is a cool city in Germany.")],
    )
    assert len(result["results"]) == 2
    for answer in result["results"]:
        assert any(word for word in ["berlin", "germany", "cool", "city", "amazing"] if word in answer.casefold())

    assert len(result["invocation_context"]) > 0
    assert len(result["invocation_context"]["questions"]) == 2
