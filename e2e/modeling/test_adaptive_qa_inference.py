import pytest

from haystack.modeling.infer import Inferencer


@pytest.mark.parametrize("multiprocessing_chunksize", [None, 2])
def test_qa_format_and_results(multiprocessing_chunksize):
    qa_inputs_dicts = [
        {
            "questions": ["In what country is Normandy"],
            "text": "The Normans are an ethnic group that arose in Normandy, a northern region "
            "of France, from contact between Viking settlers and indigenous Franks and Gallo-Romans",
        },
        {
            "questions": ["Who counted the game among the best ever made?"],
            "text": "Twilight Princess was released to universal critical acclaim and commercial success. It received "
            "perfect scores from major publications such as 1UP.com, Computer and Video Games, Electronic "
            "Gaming Monthly, Game Informer, GamesRadar, and GameSpy. On the review aggregators GameRankings "
            "and Metacritic, Twilight Princess has average scores of 95% and 95 for the Wii version and scores "
            "of 95% and 96 for the GameCube version. GameTrailers in their review called it one of the "
            "greatest games ever created.",
        },
    ]
    ground_truths = ["France", "GameTrailers"]

    adaptive_model_qa = Inferencer.load(
        "deepset/bert-medium-squad2-distilled", task_type="question_answering", batch_size=16, gpu=False
    )
    results = adaptive_model_qa.inference_from_dicts(
        dicts=qa_inputs_dicts, multiprocessing_chunksize=multiprocessing_chunksize
    )

    # sample results
    # [
    #     {
    #         "task": "qa",
    #         "predictions": [
    #             {
    #                 "question": "In what country is Normandy",
    #                 "question_id": "None",
    #                 "ground_truth": None,
    #                 "answers": [
    #                     {
    #                         "score": 1.1272038221359253,
    #                         "probability": -1,
    #                         "answer": "France",
    #                         "offset_answer_start": 54,
    #                         "offset_answer_end": 60,
    #                         "context": "The Normans gave their name to Normandy, a region in France.",
    #                         "offset_context_start": 0,
    #                         "offset_context_end": 60,
    #                         "document_id": None,
    #                     }
    #                 ]
    #             }
    #         ],
    #     }
    # ]
    predictions = list(results)[0]["predictions"]

    for prediction, ground_truth, qa_input_dict in zip(predictions, ground_truths, qa_inputs_dicts):
        assert prediction["question"] == qa_input_dict["questions"][0]
        answer = prediction["answers"][0]
        assert answer["answer"] in answer["context"]
        assert answer["answer"] == ground_truth
        assert {
            "answer",
            "score",
            "probability",
            "offset_answer_start",
            "offset_answer_end",
            "context",
            "offset_context_start",
            "offset_context_end",
            "document_id",
        } == answer.keys()
