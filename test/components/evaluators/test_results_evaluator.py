from haystack.components.evaluators.evaluation_result import EvaluationResult


def test_init_results_evaluator():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    _ = EvaluationResult(pipeline_name="testing_pipeline_1", results=data)


def test_score_report():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {
                "name": "reciprocal_rank",
                "individual_scores": [0.378064, 0.534964, 0.216058, 0.778642],
                "score": 0.476932,
            },
            {"name": "single_hit", "individual_scores": [1, 1, 0, 1], "score": 0.75},
            {"name": "multi_hit", "individual_scores": [0.706125, 0.454976, 0.445512, 0.250522], "score": 0.46428375},
            {
                "name": "context_relevance",
                "individual_scores": [0.805466, 0.410251, 0.750070, 0.361332],
                "score": 0.58177975,
            },
            {
                "name": "faithfulness",
                "individual_scores": [0.135581, 0.695974, 0.749861, 0.041999],
                "score": 0.40585375,
            },
            {
                "name": "semantic_answer_similarity",
                "individual_scores": [0.971241, 0.159320, 0.019722, 1],
                "score": 0.53757075,
            },
        ],
    }

    evaluator = EvaluationResult(pipeline_name="testing_pipeline_1", results=data)
    result = evaluator.score_report().to_json()
    assert result == (
        '{"score":{"reciprocal_rank":0.476932,"single_hit":0.75,"multi_hit":0.46428375,'
        '"context_relevance":0.58177975,"faithfulness":0.40585375,'
        '"semantic_answer_similarity":0.53757075}}'
    )


def test_to_pandas():
    data = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7", "53c3b3e6", "225f87f7"],
            "question": [
                "What is the capital of France?",
                "What is the capital of Spain?",
                "What is the capital of Luxembourg?",
                "What is the capital of Portugal?",
            ],
            "contexts": ["wiki_France", "wiki_Spain", "wiki_Luxembourg", "wiki_Portugal"],
            "answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"],
            "predicted_answer": ["Paris", "Madrid", "Luxembourg", "Lisbon"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "individual_scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "individual_scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "individual_scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "individual_scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "individual_scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "individual_scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    evaluator = EvaluationResult(pipeline_name="testing_pipeline_1", results=data)
    assert evaluator.to_pandas().to_json() == (
        '{"query_id":{"0":"53c3b3e6","1":"225f87f7","2":"53c3b3e6","3":"225f87f7"},'
        '"question":{"0":"What is the capital of France?","1":"What is the capital of Spain?",'
        '"2":"What is the capital of Luxembourg?","3":"What is the capital of Portugal?"},'
        '"contexts":{"0":"wiki_France","1":"wiki_Spain","2":"wiki_Luxembourg","3":"wiki_Portugal"},'
        '"answer":{"0":"Paris","1":"Madrid","2":"Luxembourg","3":"Lisbon"},'
        '"predicted_answer":{"0":"Paris","1":"Madrid","2":"Luxembourg","3":"Lisbon"},'
        '"reciprocal_rank":{"0":0.378064,"1":0.534964,"2":0.216058,"3":0.778642},'
        '"single_hit":{"0":1,"1":1,"2":0,"3":1},'
        '"multi_hit":{"0":0.706125,"1":0.454976,"2":0.445512,"3":0.250522},'
        '"context_relevance":{"0":0.805466,"1":0.410251,"2":0.75007,"3":0.361332},'
        '"faithfulness":{"0":0.135581,"1":0.695974,"2":0.749861,"3":0.041999},'
        '"semantic_answer_similarity":{"0":0.971241,"1":0.15932,"2":0.019722,"3":1.0}}'
    )


def test_comparative_individual_scores_report():
    data_1 = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "individual_scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "individual_scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "individual_scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "individual_scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "individual_scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "individual_scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    data_2 = {
        "inputs": {
            "query_id": ["53c3b3e6", "225f87f7"],
            "question": ["What is the capital of France?", "What is the capital of Spain?"],
            "contexts": ["wiki_France", "wiki_Spain"],
            "answer": ["Paris", "Madrid"],
            "predicted_answer": ["Paris", "Madrid"],
        },
        "metrics": [
            {"name": "reciprocal_rank", "individual_scores": [0.378064, 0.534964, 0.216058, 0.778642]},
            {"name": "single_hit", "individual_scores": [1, 1, 0, 1]},
            {"name": "multi_hit", "individual_scores": [0.706125, 0.454976, 0.445512, 0.250522]},
            {"name": "context_relevance", "individual_scores": [0.805466, 0.410251, 0.750070, 0.361332]},
            {"name": "faithfulness", "individual_scores": [0.135581, 0.695974, 0.749861, 0.041999]},
            {"name": "semantic_answer_similarity", "individual_scores": [0.971241, 0.159320, 0.019722, 1]},
        ],
    }

    evaluator_1 = EvaluationResult(pipeline_name="testing_pipeline_1", results=data_1)
    evaluator_2 = EvaluationResult(pipeline_name="testing_pipeline_2", results=data_2)
    results = evaluator_1.comparative_individual_scores_report(evaluator_2)

    assert results.to_json() == (
        '{"query_id":{"0":"53c3b3e6","1":"225f87f7"},'
        '"question":{"0":"What is the capital of France?","1":"What is the capital of Spain?"},'
        '"contexts":{"0":"wiki_France","1":"wiki_Spain"},"answer":{"0":"Paris","1":"Madrid"},'
        '"testing_pipeline_1_predicted_answer":{"0":"Paris","1":"Madrid"},'
        '"testing_pipeline_1_reciprocal_rank":{"0":0.378064,"1":0.534964},'
        '"testing_pipeline_1_single_hit":{"0":1,"1":1},'
        '"testing_pipeline_1_multi_hit":{"0":0.706125,"1":0.454976},'
        '"testing_pipeline_1_context_relevance":{"0":0.805466,"1":0.410251},'
        '"testing_pipeline_1_faithfulness":{"0":0.135581,"1":0.695974},'
        '"testing_pipeline_1_semantic_answer_similarity":{"0":0.971241,"1":0.15932},'
        '"testing_pipeline_2_predicted_answer":{"0":"Paris","1":"Madrid"},'
        '"testing_pipeline_2_reciprocal_rank":{"0":0.378064,"1":0.534964},'
        '"testing_pipeline_2_single_hit":{"0":1,"1":1},'
        '"testing_pipeline_2_multi_hit":{"0":0.706125,"1":0.454976},'
        '"testing_pipeline_2_context_relevance":{"0":0.805466,"1":0.410251},'
        '"testing_pipeline_2_faithfulness":{"0":0.135581,"1":0.695974},'
        '"testing_pipeline_2_semantic_answer_similarity":{"0":0.971241,"1":0.15932}}'
    )
