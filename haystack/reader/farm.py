import numpy as np
from scipy.special import expit
from pathlib import Path
import logging

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.infer import Inferencer
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

logger = logging.getLogger(__name__)


class FARMReader:
    """
    Transformer based model for extractive Question Answering using the FARM framework (https://github.com/deepset-ai/FARM).
    While the underlying model can vary (BERT, Roberta, DistilBERT ...) the interface remains the same.

    With a FARMReader, you can:
     - directly get predictions via predict()
     - fine-tune the model on QA data via train()
    """

    def __init__(
        self,
        model_name_or_path,
        context_window_size=30,
        no_ans_threshold=-100,
        batch_size=16,
        use_gpu=True,
        n_candidates_per_passage=2):
        """
        :param model_name_or_path: directory of a saved model or the name of a public model:
                                   - 'bert-base-cased'
                                   - 'deepset/bert-base-cased-squad2'
                                   - 'deepset/bert-base-cased-squad2'
                                   - 'distilbert-base-uncased-distilled-squad'
                                   ....
                                   See https://huggingface.co/models for full list of available models.
        :param context_window_size: The size, in characters, of the window around the answer span that is used when displaying the context around the answer.
        :param no_ans_threshold: How much greater the no_answer logit needs to be over the pos_answer in order to be chosen.
                                 The higher the value, the more `uncertain` answers are accepted
        :param batch_size: Number of samples the model receives in one batch for inference
        :param use_gpu: Whether to use GPU (if available)
        :param n_candidates_per_passage: How many candidate answers are extracted per text sequence that the model can process at once (depends on `max_seq_len`).
                                         Note: This is not the number of "final answers" you will receive
                                         (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        """


        self.inferencer = Inferencer.load(model_name_or_path, batch_size=batch_size, gpu=use_gpu, task_type="question_answering")
        self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        self.inferencer.model.prediction_heads[0].no_ans_threshold = no_ans_threshold
        self.no_ans_threshold = no_ans_threshold
        self.inferencer.model.prediction_heads[0].n_best = n_candidates_per_passage

    def train(self, data_dir, train_filename, dev_filename=None, test_file_name=None,
              use_gpu=True, batch_size=10, n_epochs=2, learning_rate=1e-5,
              max_seq_len=256, warmup_proportion=0.2, dev_split=0.1, evaluate_every=300, save_dir=None):
        """
        Fine-tune a model on a QA dataset. Options:
        - Take a plain language model (e.g. `bert-base-cased`) and train it for QA (e.g. on SQuAD data)
        - Take a QA model (e.g. `deepset/bert-base-cased-squad2`) and fine-tune it for your domain (e.g. using your labels collected via the haystack annotation tool)

        :param data_dir: Path to directory containing your training data in SQuAD style
        :param train_filename: filename of training data
        :param dev_filename: filename of dev / eval data
        :param test_file_name: filename of test data
        :param dev_split: Instead of specifying a dev_filename you can also specify a ratio (e.g. 0.1) here
                          that get's split off from training data for eval.
        :param use_gpu: Whether to use GPU (if available)
        :param batch_size: Number of samples the model receives in one batch for training
        :param n_epochs: number of iterations on the whole training data set
        :param learning_rate: learning rate of the optimizer
        :param max_seq_len: maximum text length (in tokens). Everything longer gets cut down.
        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.
                                  Until that point LR is increasing linearly. After that it's decreasing again linearly.
                                  Options for different schedules are available in FARM.
        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset
        :param save_dir: Path to store the final model
        :return: None
        """


        if dev_filename:
            dev_split = None

        set_all_seeds(seed=42)
        device, n_gpu = initialize_device_settings(use_cuda=use_gpu)

        if not save_dir:
            save_dir = f"../../saved_models/{self.inferencer.model.language_model.name}"
        save_dir = Path(save_dir)

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
            data_dir=Path(data_dir),
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
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=n_epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=evaluate_every,
            device=device,
        )
        # 5. Let it grow!
        self.inferencer.model = trainer.train()
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

        # get answers from QA model (Default: top 5 per input paragraph)
        predictions = self.inferencer.inference_from_dicts(
            dicts=input_dicts, rest_api_schema=True, max_processes=max_processes
        )

        # assemble answers from all the different paragraphs & format them
        # for the "no answer" option, we choose the no_answer score from the paragraph with the best "real answer"
        # the score of this "no answer" is then "boosted" with the no_ans_gap
        answers = []
        best_score_answer = 0
        for pred in predictions:
            for a in pred["predictions"][0]["answers"]:
                # skip "no answers" here
                if a["answer"]:
                    cur = {"answer": a["answer"],
                           "score": a["score"],
                           "probability": float(expit(np.asarray([a["score"]]) / 8)), #just a pseudo prob for now
                           "context": a["context"],
                           "offset_start": a["offset_answer_start"] - a["offset_context_start"],
                           "offset_end": a["offset_answer_end"] - a["offset_context_start"],
                           "document_id": a["document_id"]}
                    answers.append(cur)
                    # if cur answer is the best, we store the gap to "no answer" in this paragraph
                    if a["score"] > best_score_answer:
                        best_score_answer = a["score"]
                        no_ans_gap = pred["predictions"][0]["no_ans_gap"]
                        no_ans_score = (best_score_answer+no_ans_gap)-self.no_ans_threshold

        # add no answer option from the paragraph with the best answer
        cur = {"answer": "",
               "score": no_ans_score,
               "probability": float(expit(np.asarray(no_ans_score) / 8)),  # just a pseudo prob for now
               "context": "",
               "offset_start": -1,
               "offset_end": -1,
               "document_id": None}
        answers.append(cur)

        # sort answers by their `probability` and select top-k
        answers = sorted(
            answers, key=lambda k: k["probability"], reverse=True
        )
        answers = answers[:top_k]
        result = {"question": question,
                   "answers": answers}

        return result