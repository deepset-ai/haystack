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
        n_best_per_passage=2
    ):
        """
        Load a saved FARM model in Inference mode.

        :param model_dir: directory path of the saved model
        """
        self.model = Inferencer.load(model_dir, batch_size=batch_size, gpu=use_gpu)
        self.model.model.prediction_heads[0].context_size = context_size
        self.model.model.prediction_heads[0].no_answer_shift = no_answer_shift
        self.model.model.prediction_heads[0].n_best = n_best_per_passage


    def predict(self, question, paragrahps, meta_data_paragraphs=None, top_k=None, max_processes=1):
        """
        Use loaded QA model to find answers for a question in the supplied paragraphs.

        Returns dictionaries containing answers sorted by (desc.) probability
        Example:
        {'question': 'Who is the father of Arya Stark?',
        'answers': [
                     {'answer': 'Eddard,',
                     'context': " She travels with her father, Eddard, to King's Landing when he is ",
                     'offset_answer_start': 147,
                     'offset_answer_end': 154,
                     'probability': 0.9787139466668613,
                     'score': None,
                     'document_id': None
                     },
                    ...
                   ]
        }

        :param question: question string
        :param paragraphs: list of strings in which to search for the answer
        :param meta_data_paragraphs: list of dicts containing meta data for the paragraphs.
                                     len(paragraphs) == len(meta_data_paragraphs)
        :param top_k: the maximum number of answers to return
        :param max_processes: max number of parallel processes
        :return: dict containing question and answers
        """

        if meta_data_paragraphs is None:
            meta_data_paragraphs = len(paragrahps) * [None]
        assert len(paragrahps) == len(meta_data_paragraphs)

        # convert input to FARM format
        input_dicts = []
        for paragraph, meta_data in zip(paragrahps, meta_data_paragraphs):
            cur = {"text": paragraph,
                   "questions": [question],
                   "document_id": meta_data["document_id"]
            }
            input_dicts.append(cur)

        # get answers from QA model (Top 5 per input paragraph)
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
                           "probability": float(expit(np.asarray([a["score"]]) / 8)), #just a pseudo prob for now
                           "context": a["context"],
                           "offset_start": a["offset_answer_start"] - a["offset_context_start"],
                           "offset_end": a["offset_answer_end"] - a["offset_context_start"],
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
