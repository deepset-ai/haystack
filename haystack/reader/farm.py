import logging
from pathlib import Path

import numpy as np
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.infer import Inferencer
from farm.modeling.optimization import initialize_optimizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings
from scipy.special import expit

from haystack.database.base import Document

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
        context_window_size=150,
        batch_size=50,
        use_gpu=True,
        no_ans_boost=None,
        top_k_per_candidate=3,
        top_k_per_sample=1,
        num_processes=None,
        max_seq_len=256,
        doc_stride=128
    ):

        """
        :param model_name_or_path: directory of a saved model or the name of a public model:
                                   - 'bert-base-cased'
                                   - 'deepset/bert-base-cased-squad2'
                                   - 'deepset/bert-base-cased-squad2'
                                   - 'distilbert-base-uncased-distilled-squad'
                                   ....
                                   See https://huggingface.co/models for full list of available models.
        :param context_window_size: The size, in characters, of the window around the answer span that is used when displaying the context around the answer.
        :param batch_size: Number of samples the model receives in one batch for inference
                           Memory consumption is much lower in inference mode. Recommendation: increase the batch size to a value so only a single batch is used.
        :param use_gpu: Whether to use GPU (if available)
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
                             Possible values: None (default) = disable returning "no answer" predictions
                                              Negative = lower chance of "no answer" being predicted
                                              Positive = increase chance of "no answer"
        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
                                                   Note: - This is not the number of "final answers" you will receive
                                                   (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
                                                 - FARM includes no_answer in the sorted list of predictions
        :param top_k_per_sample: How many answers to extract from each small text passage that the model can
                                  process at once (one "candidate doc" is usually split into many smaller "passages").
                                  You usually want a very small value here, as it slows down inference and you
                                  don't gain much of quality by having multiple answers from one passage.
                                               Note: - This is not the number of "final answers" you will receive
                                               (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
                                             - FARM includes no_answer in the sorted list of predictions
        :param num_processes: the number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer determine optimum number. If you
                              want to debug the Language Model, you might need to disable multiprocessing!
        :type num_processes: int
        :param max_seq_len: max sequence length of one input text for the model
        :param doc_stride: length of striding window for splitting long texts (used if len(text) > max_seq_len)

        """

        if no_ans_boost is None:
            no_ans_boost = 0
            self.return_no_answers = False
        else:
            self.return_no_answers = True
        self.top_k_per_candidate = top_k_per_candidate
        self.inferencer = Inferencer.load(model_name_or_path, batch_size=batch_size, gpu=use_gpu,
                                          task_type="question_answering", max_seq_len=max_seq_len,
                                          doc_stride=doc_stride, num_processes=num_processes)
        self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        self.inferencer.model.prediction_heads[0].n_best = top_k_per_candidate + 1 # including possible no_answer
        try:
            self.inferencer.model.prediction_heads[0].n_best_per_sample = top_k_per_sample
        except:
            logger.warning("Could not set `top_k_per_sample` in FARM. Please update FARM version.")
        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu

    def train(self, data_dir, train_filename, dev_filename=None, test_file_name=None,
              use_gpu=None, batch_size=10, n_epochs=2, learning_rate=1e-5,
              max_seq_len=None, warmup_proportion=0.2, dev_split=0.1, evaluate_every=300, save_dir=None):
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

        # For these variables, by default, we use the value set when initializing the FARMReader.
        # These can also be manually set when train() is called if you want a different value at train vs inference
        if use_gpu is None:
            use_gpu = self.use_gpu
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

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

    def predict(self, question: str, documents: [Document], top_k: int = None):
        """
        Use loaded QA model to find answers for a question in the supplied list of Document.

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
                     'document_id': '1337'
                     },
                    ...
                   ]
        }

        :param question: question string
        :param documents: list of Document in which to search for the answer
        :param top_k: the maximum number of answers to return
        :return: dict containing question and answers
        """

        # convert input to FARM format
        input_dicts = []
        for doc in documents:
            cur = {
                "text": doc.text,
                "questions": [question],
                "document_id": doc.id
            }
            input_dicts.append(cur)

        # get answers from QA model
        predictions = self.inferencer.inference_from_dicts(
            dicts=input_dicts, rest_api_schema=True, multiprocessing_chunksize=1
        )
        # assemble answers from all the different documents & format them.
        # For the "no answer" option, we collect all no_ans_gaps and decide how likely
        # a no answer is based on all no_ans_gaps values across all documents
        answers = []
        no_ans_gaps = []
        best_score_answer = 0
        for pred in predictions:
            answers_per_document = []
            no_ans_gaps.append(pred["predictions"][0]["no_ans_gap"])
            for a in pred["predictions"][0]["answers"]:
                # skip "no answers" here
                if a["answer"]:
                    cur = {"answer": a["answer"],
                           "score": a["score"],
                           "probability": float(expit(np.asarray([a["score"]]) / 8)), #just a pseudo prob for now
                           "context": a["context"],
                           "offset_start": a["offset_answer_start"] - a["offset_context_start"],
                           "offset_end": a["offset_answer_end"] - a["offset_context_start"],
                           "offset_start_in_doc": a["offset_answer_start"],
                           "offset_end_in_doc": a["offset_answer_end"],
                           "document_id": a["document_id"]}
                    answers_per_document.append(cur)

                    if a["score"] > best_score_answer:
                        best_score_answer = a["score"]
            # only take n best candidates. Answers coming back from FARM are sorted with decreasing relevance.
            answers += answers_per_document[:self.top_k_per_candidate]

        # Calculate the score for predicting "no answer", relative to our best positive answer score
        no_ans_prediction, max_no_ans_gap = self._calc_no_answer(no_ans_gaps,best_score_answer)
        if self.return_no_answers:
            answers.append(no_ans_prediction)

        # sort answers by their `probability` and select top-k
        answers = sorted(
            answers, key=lambda k: k["probability"], reverse=True
        )
        answers = answers[:top_k]
        result = {"question": question,
                  "no_ans_gap": max_no_ans_gap,
                  "answers": answers}

        return result

    @staticmethod
    def _calc_no_answer(no_ans_gaps,best_score_answer):
        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to one specific document
        # - a "no answer" score is related to all input documents
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: a model switching from predicting an answer to "no answer" (or vice versa).
        # No_ans_gap coming from FARM mean how much no_ans_boost should change to switch predictions
        no_ans_gaps = np.array(no_ans_gaps)
        max_no_ans_gap = np.max(no_ans_gaps)
        if (np.sum(no_ans_gaps < 0) == len(no_ans_gaps)):  # all passages "no answer" as top score
            no_ans_score = best_score_answer - max_no_ans_gap  # max_no_ans_gap is negative, so it increases best pos score
        else:  # case: at least one passage predicts an answer (positive no_ans_gap)
            no_ans_score = best_score_answer - max_no_ans_gap

        no_ans_prediction = {"answer": None,
               "score": no_ans_score,
               "probability": float(expit(np.asarray(no_ans_score) / 8)),  # just a pseudo prob for now
               "context": None,
               "offset_start": 0,
               "offset_end": 0,
               "document_id": None}
        return no_ans_prediction, max_no_ans_gap

    def predict_on_texts(self, question: str, texts: [str], top_k=None):
        documents = []
        for i, text in enumerate(texts):
            documents.append(
                Document(
                    id=i,
                    text=text
                )
            )
        predictions = self.predict(question, documents, top_k)
        return predictions
