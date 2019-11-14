from farm.infer import Inferencer
import numpy as np
from scipy.special import expit


class FARMReader:
    """
    Implementation of FARM Inferencer for Question Answering.

    The class loads a saved FARM adaptive model from a given directory and runs
    inference using `inference_from_dicts()` method.
    """

    def __init__(
        self,
        model_dir,
        context_size=30,
        no_answer_shift=-100,
        batch_size=16,
        use_gpu=True,
    ):
        """
        Load a saved FARM model in Inference mode.

        :param model_dir: directory path of the saved model
        """
        self.model = Inferencer.load(model_dir, batch_size=batch_size, gpu=use_gpu)
        self.model.model.prediction_heads[0].context_size = context_size
        self.model.model.prediction_heads[0].no_answer_shift = no_answer_shift

    def predict(self, input_dicts, top_k=None):
        """
        Run inference on the loaded model for the given input dicts.

        :param input_dicts: list of input dicts
        :param top_k: the maximum number of answers to return
        :return:
        """
        results = self.model.inference_from_dicts(
            dicts=input_dicts, rest_api_schema=True, use_multiprocessing=False
        )

        # The FARM Inferencer as of now do not support multi document QA.
        # The QA inference is done for each text independently and the
        # results are sorted descending by their `score`.

        all_predictions = []
        for res in results:
            all_predictions.extend(res["predictions"])

        all_answers = []
        for pred in all_predictions:
            answers = pred["answers"]
            for a in answers:
                # Two sets of offset fields are returned by FARM -- context level and document level.
                # For the API, only context level offsets are relevant.
                a["offset_start"] = a["offset_answer_start"] - a["offset_context_start"]
                a["offset_end"] = a["offset_context_end"] - a["offset_answer_end"]
            all_answers.extend(answers)

        # remove all null answers (where an answers in not found in the text)
        all_answers = [ans for ans in all_answers if ans["answer"]]

        scores = np.asarray([ans["score"] for ans in all_answers])
        probabilities = expit(scores / 8)
        for ans, prob in zip(all_answers, probabilities):
            ans["probability"] = prob

        # sort answers by their `probability`
        sorted_answers = sorted(
            all_answers, key=lambda k: k["probability"], reverse=True
        )

        # all predictions here are for the same questions, so the the metadata from
        # the first prediction in the list is taken.
        if all_predictions:
            resp = all_predictions[0]  # get the first prediction dict
            resp["answers"] = sorted_answers[:top_k]
        else:
            resp = []

        return {"results": [resp]}
