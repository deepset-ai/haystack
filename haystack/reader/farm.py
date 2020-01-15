from farm.infer import Inferencer
import numpy as np
from scipy.special import expit

import logging
import os
import pprint

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.utils import write_squad_predictions
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

logger = logging.getLogger(__name__)


class FARMReader:
    """
    Implementation of FARM Inferencer for Question Answering.

    The class loads a saved FARM adaptive model from a given directory and runs
    inference using `inference_from_dicts()` method.
    """

    def __init__(
        self,
        model_name_or_path,
        context_size=30,
        no_answer_shift=-100,
        batch_size=16,
        use_gpu=True,
        n_candidates_per_passage=2
    ):
        """
        Load a saved FARM model in Inference mode.

        :param model_name_or_path: directory of a saved model or the name of a public model:
                                   - 'bert-base-cased'
                                   - 'bert-base-cased-squad2'
                                   - 'distilbert'
                                   ....
                                   See XX for full list of available models.

        """
        #TODO enable loading of remote models
        #TODO conversion to QA model if model = base LM (e.g. bert-base-cased)

        #TODO potentially move the loading into a load() class method?

        self.inferencer = Inferencer.load(model_name_or_path, batch_size=batch_size, gpu=use_gpu)
        self.inferencer.model.prediction_heads[0].context_size = context_size
        self.inferencer.model.prediction_heads[0].no_answer_shift = no_answer_shift
        self.inferencer.model.prediction_heads[0].n_best = n_candidates_per_passage

    def train(self, data_dir, train_filename, dev_filename=None, test_file_name=None,
              use_gpu=True, batch_size=10, n_epochs=2, learning_rate=1e-5,
              max_seq_len=256, warmup_proportion=0.2, dev_split=0.1):
        """
        Fine-tune the loaded model using a
        :param data_dir:
        :param train_filename:
        :param dev_filename:
        :param test_file_name:
        :param use_gpu:
        :param batch_size:
        :param n_epochs:
        :param learning_rate:
        :param max_seq_len:
        :param warmup_proportion:
        :param dev_split:
        :return:
        """

        if dev_filename:
            dev_split = None

        set_all_seeds(seed=42)
        device, n_gpu = initialize_device_settings(use_cuda=use_gpu)
        evaluate_every = 100
        save_dir = f"../../saved_models/{self.inferencer.model.language_model.name}"

        # 1. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
        label_list = ["start_token", "end_token"]
        metric = "squad"
        processor = SquadProcessor(
            tokenizer=self.inferencer.processor.tokenizer,
            max_seq_len=max_seq_len,
            label_list=label_list,
            metric=metric,
            train_filename=train_filename,
            dev_filename=dev_filename,
            dev_split=dev_split,
            test_filename=test_file_name,
            data_dir=data_dir,
        )

        # 2. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them
        # and calculates a few descriptive statistics of our datasets
        data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

        # 3. Create an optimizer and pass the already initialized model
        model, optimizer, lr_schedule = initialize_optimizer(
            model=self.inferencer.model,
            learning_rate=learning_rate,
            schedule_opts={"name": "LinearWarmup", "warmup_proportion": warmup_proportion},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            device=device
        )
        # 4. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
        trainer = Trainer(
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device,
        )
        # 5. Let it grow!
        self.inferencer.model = trainer.train(model)
        self.save(save_dir)

    def save(self, directory):
        logger.info(f"Saving reader model to {directory}")
        self.inferencer.model.save(directory)
        self.inferencer.processor.save(directory)

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
        predictions = self.inferencer.inference_from_dicts(
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
