import logging
import pytest
from math import isclose
import numpy as np

from haystack.modeling.infer import QAInferencer
from haystack.modeling.data_handler.inputs import QAInput, Question


DOC_TEXT = """Twilight Princess was released to universal critical acclaim and commercial success. \
It received perfect scores from major publications such as 1UP.com, Computer and Video Games, \
Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators \
GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii \
version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called \
it one of the greatest games ever created."""


@pytest.fixture
def bert_base_squad2(request):
    model = QAInferencer.load(
        "deepset/minilm-uncased-squad2",
        task_type="question_answering",
        batch_size=4,
        num_processes=0,
        multithreading_rust=False,
    )
    return model


@pytest.fixture()
def span_inference_result(bert_base_squad2):
    obj_input = [
        QAInput(
            doc_text=DOC_TEXT, questions=Question("Who counted the game among the best ever made?", uid="best_id_ever")
        )
    ]
    result = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]
    return result


@pytest.fixture()
def no_answer_inference_result(bert_base_squad2, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    obj_input = [
        QAInput(
            doc_text="""\
                The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by
                Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana,
                Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names.
                The Amazon represents over half of the planet\'s remaining rainforests, and comprises the largest
                and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual
                trees divided into 16,000 species.""",
            questions=Question(
                "The Amazon represents less than half of the planets remaining what?", uid="best_id_ever"
            ),
        )
    ]
    result = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]
    return result


def test_inference_different_inputs(bert_base_squad2):
    qa_format_1 = [{"questions": ["Who counted the game among the best ever made?"], "text": DOC_TEXT}]
    q = Question(text="Who counted the game among the best ever made?")
    qa_format_2 = QAInput(questions=[q], doc_text=DOC_TEXT)

    result1 = bert_base_squad2.inference_from_dicts(dicts=qa_format_1)
    result2 = bert_base_squad2.inference_from_objects(objects=[qa_format_2])
    assert result1 == result2


def test_span_inference_result_ranking_by_confidence(bert_base_squad2, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    obj_input = [
        QAInput(
            doc_text=DOC_TEXT, questions=Question("Who counted the game among the best ever made?", uid="best_id_ever")
        )
    ]

    # by default, result is sorted by confidence and not by score
    result_ranked_by_confidence = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]
    assert all(
        result_ranked_by_confidence.prediction[i].confidence >= result_ranked_by_confidence.prediction[i + 1].confidence
        for i in range(len(result_ranked_by_confidence.prediction) - 1)
    )
    assert not all(
        result_ranked_by_confidence.prediction[i].score >= result_ranked_by_confidence.prediction[i + 1].score
        for i in range(len(result_ranked_by_confidence.prediction) - 1)
    )

    # ranking can be adjusted so that result is sorted by score
    bert_base_squad2.model.prediction_heads[0].use_confidence_scores_for_ranking = False
    result_ranked_by_score = bert_base_squad2.inference_from_objects(obj_input, return_json=False)[0]
    assert all(
        result_ranked_by_score.prediction[i].score >= result_ranked_by_score.prediction[i + 1].score
        for i in range(len(result_ranked_by_score.prediction) - 1)
    )
    assert not all(
        result_ranked_by_score.prediction[i].confidence >= result_ranked_by_score.prediction[i + 1].confidence
        for i in range(len(result_ranked_by_score.prediction) - 1)
    )


def test_inference_objs(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    assert span_inference_result


def test_span_performance(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    best_pred = span_inference_result.prediction[0]

    assert best_pred.answer == "GameTrailers"

    best_score_gold = 13.4205
    best_score = best_pred.score
    assert isclose(best_score, best_score_gold, rel_tol=0.001)

    no_answer_gap_gold = 13.9827
    no_answer_gap = span_inference_result.no_answer_gap
    assert isclose(no_answer_gap, no_answer_gap_gold, rel_tol=0.001)


def test_no_answer_performance(no_answer_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)
    best_pred = no_answer_inference_result.prediction[0]

    assert best_pred.answer == "no_answer"

    best_score_gold = 12.1445
    best_score = best_pred.score
    assert isclose(best_score, best_score_gold, rel_tol=0.001)

    no_answer_gap_gold = -14.4646
    no_answer_gap = no_answer_inference_result.no_answer_gap
    assert isclose(no_answer_gap, no_answer_gap_gold, rel_tol=0.001)


def test_qa_pred_attributes(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    qa_pred = span_inference_result
    attributes_gold = [
        "aggregation_level",
        "answer_types",
        "context",
        "context_window_size",
        "ground_truth_answer",
        "id",
        "n_passages",
        "no_answer_gap",
        "prediction",
        "question",
        "to_json",
        "to_squad_eval",
        "token_offsets",
    ]

    for ag in attributes_gold:
        assert ag in dir(qa_pred)


def test_qa_candidate_attributes(span_inference_result, caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    qa_candidate = span_inference_result.prediction[0]
    attributes_gold = [
        "aggregation_level",
        "answer",
        "answer_support",
        "answer_type",
        "context_window",
        "n_passages_in_doc",
        "offset_answer_end",
        "offset_answer_start",
        "offset_answer_support_end",
        "offset_answer_support_start",
        "offset_context_window_end",
        "offset_context_window_start",
        "offset_unit",
        "passage_id",
        "probability",
        "score",
        "set_answer_string",
        "set_context_window",
        "to_doc_level",
        "to_list",
    ]

    for ag in attributes_gold:
        assert ag in dir(qa_candidate)


def test_id(span_inference_result, no_answer_inference_result):
    assert span_inference_result.id == "best_id_ever"
    assert no_answer_inference_result.id == "best_id_ever"


def test_duplicate_answer_filtering(bert_base_squad2):
    qa_input = [
        {
            "questions": ["“In what country lies the Normandy?”"],
            "text": """The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\")
                raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia.
                The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. Weird things happen in Normandy, France.""",
        }
    ]

    bert_base_squad2.model.prediction_heads[0].n_best = 5
    bert_base_squad2.model.prediction_heads[0].n_best_per_sample = 5
    bert_base_squad2.model.prediction_heads[0].duplicate_filtering = 0

    result = bert_base_squad2.inference_from_dicts(dicts=qa_input)
    offset_answer_starts = []
    offset_answer_ends = []
    for answer in result[0]["predictions"][0]["answers"]:
        offset_answer_starts.append(answer["offset_answer_start"])
        offset_answer_ends.append(answer["offset_answer_end"])

    assert len(offset_answer_starts) == len(set(offset_answer_starts))
    assert len(offset_answer_ends) == len(set(offset_answer_ends))


def test_no_duplicate_answer_filtering(bert_base_squad2):
    qa_input = [
        {
            "questions": ["“In what country lies the Normandy?”"],
            "text": """The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\")
                    raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia.
                    The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. Weird things happen in Normandy, France.""",
        }
    ]

    bert_base_squad2.model.prediction_heads[0].n_best = 5
    bert_base_squad2.model.prediction_heads[0].n_best_per_sample = 5
    bert_base_squad2.model.prediction_heads[0].duplicate_filtering = -1
    bert_base_squad2.model.prediction_heads[0].no_ans_boost = -100.0

    result = bert_base_squad2.inference_from_dicts(dicts=qa_input)
    offset_answer_starts = []
    offset_answer_ends = []
    for answer in result[0]["predictions"][0]["answers"]:
        offset_answer_starts.append(answer["offset_answer_start"])
        offset_answer_ends.append(answer["offset_answer_end"])

    assert len(offset_answer_starts) != len(set(offset_answer_starts))
    assert len(offset_answer_ends) != len(set(offset_answer_ends))


def test_range_duplicate_answer_filtering(bert_base_squad2):
    qa_input = [
        {
            "questions": ["“In what country lies the Normandy?”"],
            "text": """The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\")
                    raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia.
                    The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries. Weird things happen in Normandy, France.""",
        }
    ]

    bert_base_squad2.model.prediction_heads[0].n_best = 5
    bert_base_squad2.model.prediction_heads[0].n_best_per_sample = 5
    bert_base_squad2.model.prediction_heads[0].duplicate_filtering = 5

    result = bert_base_squad2.inference_from_dicts(dicts=qa_input)
    offset_answer_starts = []
    offset_answer_ends = []
    for answer in result[0]["predictions"][0]["answers"]:
        offset_answer_starts.append(answer["offset_answer_start"])
        offset_answer_ends.append(answer["offset_answer_end"])

    offset_answer_starts.sort()
    offset_answer_starts.remove(0)
    distances_answer_starts = [j - i for i, j in zip(offset_answer_starts[:-1], offset_answer_starts[1:])]
    assert all(
        distance > bert_base_squad2.model.prediction_heads[0].duplicate_filtering
        for distance in distances_answer_starts
    )

    offset_answer_ends.sort()
    offset_answer_ends.remove(0)
    distances_answer_ends = [j - i for i, j in zip(offset_answer_ends[:-1], offset_answer_ends[1:])]
    assert all(
        distance > bert_base_squad2.model.prediction_heads[0].duplicate_filtering for distance in distances_answer_ends
    )


def test_qa_confidence():
    inferencer = QAInferencer.load(
        "deepset/roberta-base-squad2", task_type="question_answering", batch_size=40, gpu=True
    )
    QA_input = [
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the greatest games ever created.",
        }
    ]
    result = inferencer.inference_from_dicts(dicts=QA_input, return_json=False)[0]
    assert np.isclose(result.prediction[0].confidence, 0.990427553653717)
    assert result.prediction[0].answer == "GameTrailers"
