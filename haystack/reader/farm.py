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

    def predict(self, question, paragrahps, meta_data_paragraphs=None, top_k=None, max_processes=1):
        """
        Run inference on the loaded model for the given input dicts.

        TODO
        :param input_dicts: list of input dicts
        :param top_k: the maximum number of answers to return
        :param max_processes: max number of parallel processes
        :return:
        """

        # convert input to FARM format
        input_dicts = []

        if meta_data_paragraphs is None:
            meta_data_paragraphs = len(paragrahps) * [None]

        for paragraph, meta_data in zip(paragrahps, meta_data_paragraphs):
            cur = {"text": paragraph,
                   "questions": [question],
                   "document_id": meta_data["document_id"]
            }
            input_dicts.append(cur)

        # get answers from QA model: top 5 per input paragraph
        # TODO rename arg rest_api_schema?
        predictions = self.model.inference_from_dicts(
            dicts=input_dicts, rest_api_schema=True, max_processes=max_processes
        )

        # assemble answers from all the different paragraphs & format them
        answers = []
        for pred in predictions:
            for a in pred["predictions"][0]["answers"]:
                if a["answer"]: #skip "no answer"
                    cur = {"answer": a["answer"],
                           "score": a["score"],
                           "probability": expit(np.asarray([a["score"]]) / 8), #just a pseudo prob for now
                           "context": a["context"],
                           "offset_start": a["offset_answer_start"] - a["offset_context_start"],
                           "offset_end": a["offset_answer_start"] - a["offset_context_start"],
                           "document_id": a["document_id"]}
                    answers.append(cur)

        # sort answers by their `probability` and select top-k
        answers = sorted(
            answers, key=lambda k: k["probability"], reverse=True
        )
        answers = answers[:top_k]

        result = {"question": question,
                   "answers": answers}

        return result
