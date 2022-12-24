from haystack import Pipeline, Document


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
    assert isinstance(docs, str) and docs == "Berlin is an amazing city. I love Berlin"


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
                        num_tokens:
                            func: len
                            params:
                                - questions
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
    assert result["documents"] == "Be"


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
    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "Check the function name" in str(e)


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

    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "The following variables were not found " in str(e)


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
    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "Error invoking function" in str(e) and "but the provided arguments are" in str(e)


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
    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "Error invoking function" in str(e) and "but the provided arguments are" in str(e)


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
    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "Invalid YAML definition" in str(e)


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
    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "Invalid function arguments" in str(e)


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
    try:
        pipeline.run(
            query="What can you tell me about Berlin?",
            documents=[Document("Berlin is an amazing city."), Document("I love Berlin.")],
        )
    except Exception as e:
        assert "Invalid function arguments" in str(e)


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
