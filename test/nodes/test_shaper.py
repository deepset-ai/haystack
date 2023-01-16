import pytest

from haystack import Pipeline, Document
from haystack.nodes import Shaper


def test_basic_function_invocation(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                    documents:
                      func: concat_docs
                      output: documents
                      params:
                        docs: documents
                        delimiter: " "
              type: Shaper
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
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    assert isinstance(result["query"], str)
    questions = result["meta"]["invocation_context"]["questions"]

    # questions has been expanded to a list of strings of size 2 (because Documents has 2 elements)
    assert isinstance(questions, list) and len(questions) == 2 and questions[0] == result["query"]

    docs = result["documents"]
    assert isinstance(docs, str) and docs == "Berlin is an amazing city. I love Berlin."


def test_basic_function_invocation_no_output_var(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                    documents:
                      func: concat_docs
                      #output: if output is not specified the result will be bound to input variable (documents)
                      params:
                        docs: documents
                        delimiter: " "
              type: Shaper
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
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    assert isinstance(result["query"], str)
    questions = result["meta"]["invocation_context"]["questions"]

    # questions has been expanded to a list of strings of size 2 (because Documents has 2 elements)
    assert isinstance(questions, list) and len(questions) == 2 and questions[0] == result["query"]
    docs = result["documents"]
    assert isinstance(docs, str) and docs == "Berlin is an amazing city. I love Berlin."


def test_rename_vars(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      output: questions
              type: Shaper
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
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    # query has been renamed to questions
    assert isinstance(result["meta"]["invocation_context"]["questions"], str)
    assert result["meta"]["invocation_context"]["questions"] == result["query"]


def test_rename_vars_non_yaml(tmp_path):
    directives = {"query": {"output": "questions"}}
    shaper = Shaper(inputs=directives)
    pipeline = Pipeline()
    pipeline.add_node(component=shaper, name="shaper", inputs=["Query"])
    result = pipeline.run(
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    # query has been renamed to questions
    assert isinstance(result["meta"]["invocation_context"]["questions"], str)
    assert result["meta"]["invocation_context"]["questions"] == result["query"]


def test_expand_with_some_default_params(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      params:
                        expand_target: query
                      output: questions
              type: Shaper
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
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    assert isinstance(result["query"], str)
    questions = result["meta"]["invocation_context"]["questions"]

    # questions has been expanded to a list of strings of size 2 (because Documents has 2 elements)
    assert isinstance(questions, list) and len(questions) == 2 and questions[0] == result["query"]


def test_expand_with_all_default_params(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
              type: Shaper
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
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    assert isinstance(result["query"], str)
    questions = result["meta"]["invocation_context"]["questions"]

    # questions has been expanded to a list of strings of size 2 (because Documents has 2 elements)
    assert isinstance(questions, list) and len(questions) == 2 and questions[0] == result["query"]


def test_function_invocation_order(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                    documents:
                      func: concat_docs
                      output: documents
                      params:
                        docs: documents
                        delimiter: " "
              type: Shaper
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
        query="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    assert isinstance(result["query"], str)
    assert isinstance(result["documents"], str)
    # Calculations are executed top down:
    # 1) query is expanded to a list of strings of size 2 (stored as questions)
    # 2) documents is concated to a string of size of 2, because we used questions variable from 1) to calculate
    # the number of tokens
    assert result["documents"] == "Berlin is an amazing city. I love Berlin."


def test_invalid_function_used(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand_the_function_invalid_name
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    with pytest.raises(Exception, match="Check the function name") as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_invalid_input_var_used(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    some_invalid_var:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")

    with pytest.raises(
        Exception, match="The following variables, specified in Shaper directives, were not resolved"
    ) as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_function_invocation_invalid_kwarg_used(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target_totally_invalid_kwarg: query
                        size:
                          func: len
                          params:
                            - documents
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    with pytest.raises(Exception, match="Error invoking function") as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_function_invocation_multiple_invalid_kwarg_used(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                        invalid_kwarg: invalid_value
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    with pytest.raises(Exception, match="Error invoking function") as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_function_invocation_missing_params(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    with pytest.raises(Exception, match="Invalid YAML definition") as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_function_invocation_invalid_arg_param_count(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                              - documents
                              - some_invalid_param
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    with pytest.raises(Exception, match="Invalid function arguments") as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_function_invocation_invalid_arg(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                              - some_invalid_param
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    with pytest.raises(Exception, match="Invalid function arguments") as e:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )


def test_basic_function_batch_invocation(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: shaper
              params:
                inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                    documents:
                      func: concat_docs
                      output: documents
                      params:
                        docs: documents
                        delimiter: " "
              type: Shaper
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    result = pipeline.run_batch(
        queries="What can you tell me about Berlin?",
        documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
    )
    assert result
    assert "queries" in result and "query" in result
    assert "questions" in result["meta"]["invocation_context"]

    result = pipeline.run_batch(
        queries=["What can you tell me about Berlin?", "Is Berlin as cool as they say?"],
        documents=[
            [Document("Berlin is an amazing city."), Document("I love Berlin.")],
            [Document("Berlin is a wonderful city."), Document("Berlin is pretty cool.")],
        ],
    )
    assert result
    assert "queries" in result
    assert len(result["meta"]) > 0
    assert "questions" in result["meta"][0]["invocation_context"]


def test_prompt_node_with_shaper(tmp_path):
    # tests that the prompt node works with the shaper node
    # here we use shaper to expand the query to the size of documents and rename the output to questions
    # this use case was the original motivation for the introduction of the shaper
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: pmodel
                type: PromptModel
              - name: shaper
                params:
                  inputs:
                    query:
                      func: expand
                      output: questions
                      params:
                        expand_target: query
                        size:
                          func: len
                          params:
                            - documents
                type: Shaper
              - name: p1
                params:
                  model_name_or_path: pmodel
                  default_prompt_template: question-answering
                type: PromptNode
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
                  - name: p1
                    inputs:
                      - shaper
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
    assert len(result["meta"]["invocation_context"]) > 0
    assert len(result["meta"]["invocation_context"]["questions"]) == 2


def test_prompt_node_with_shaper_using_defaults(tmp_path):
    # tests that the prompt node works with the shaper node but using the default values of input directives
    # in the shaper YAML definition
    # therefore notice the difference in the config file between this test and `test_prompt_node_with_shaper`
    # here we use shaper to expand the query to the size of documents and rename the output to questions
    # this use case was the original motivation for the introduction of the shaper
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
              - name: pmodel
                type: PromptModel
              - name: shaper
                params:
                  inputs:
                    query:
                      func: expand
                      output: questions
                type: Shaper
              - name: p1
                params:
                  model_name_or_path: pmodel
                  default_prompt_template: question-answering
                type: PromptNode
            pipelines:
              - name: query
                nodes:
                  - name: shaper
                    inputs:
                      - Query
                  - name: p1
                    inputs:
                      - shaper
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
    assert len(result["meta"]["invocation_context"]) > 0
    assert len(result["meta"]["invocation_context"]["questions"]) == 2
