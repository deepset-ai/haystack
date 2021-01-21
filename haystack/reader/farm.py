import logging
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from collections import defaultdict
from time import perf_counter

import numpy as np
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.dataloader import NamedDataLoader
from farm.data_handler.inputs import QAInput, Question
from farm.infer import QAInferencer
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.predictions import QAPred, QACandidate
from farm.modeling.adaptive_model import BaseAdaptiveModel, AdaptiveModel
from farm.train import Trainer
from farm.eval import Evaluator
from farm.utils import set_all_seeds, initialize_device_settings
from scipy.special import expit
import shutil

from haystack import Document
from haystack.document_store.base import BaseDocumentStore
from haystack.reader.base import BaseReader

logger = logging.getLogger(__name__)


class FARMReader(BaseReader):
    """
    Transformer based model for extractive Question Answering using the FARM framework (https://github.com/deepset-ai/FARM).
    While the underlying model can vary (BERT, Roberta, DistilBERT, ...), the interface remains the same.

    |  With a FARMReader, you can:

     - directly get predictions via predict()
     - fine-tune the model on QA data via train()
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        context_window_size: int = 150,
        batch_size: int = 50,
        use_gpu: bool = True,
        no_ans_boost: float = 0.0,
        return_no_answer: bool = False,
        top_k_per_candidate: int = 3,
        top_k_per_sample: int = 1,
        num_processes: Optional[int] = None,
        max_seq_len: int = 256,
        doc_stride: int = 128,
    ):

        """
        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bert-base-cased',
        'deepset/bert-base-cased-squad2', 'deepset/bert-base-cased-squad2', 'distilbert-base-uncased-distilled-squad'.
        See https://huggingface.co/models for full list of available models.
        :param context_window_size: The size, in characters, of the window around the answer span that is used when
                                    displaying the context around the answer.
        :param batch_size: Number of samples the model receives in one batch for inference.
                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                           to a value so only a single batch is used.
        :param use_gpu: Whether to use GPU (if available)
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
        If set to 0 (default), the no_answer logit is not changed.
        If a negative number, there is a lower chance of "no_answer" being predicted.
        If a positive number, there is an increased chance of "no_answer"
        :param return_no_answer: Whether to include no_answer predictions in the results.
        :param top_k_per_candidate: How many answers to extract for each candidate doc that is coming from the retriever (might be a long text).
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        and that FARM includes no_answer in the sorted list of predictions.
        :param top_k_per_sample: How many answers to extract from each small text passage that the model can process at once
        (one "candidate doc" is usually split into many smaller "passages").
        You usually want a very small value here, as it slows down inference
        and you don't gain much of quality by having multiple answers from one passage.
        Note that this is not the number of "final answers" you will receive
        (see `top_k` in FARMReader.predict() or Finder.get_answers() for that)
        and that FARM includes no_answer in the sorted list of predictions.
        :param num_processes: The number of processes for `multiprocessing.Pool`. Set to value of 0 to disable
                              multiprocessing. Set to None to let Inferencer determine optimum number. If you
                              want to debug the Language Model, you might need to disable multiprocessing!
        :param max_seq_len: Max sequence length of one input text for the model
        :param doc_stride: Length of striding window for splitting long texts (used if ``len(text) > max_seq_len``)

        """

        self.return_no_answers = return_no_answer
        self.top_k_per_candidate = top_k_per_candidate
        self.inferencer = QAInferencer.load(model_name_or_path, batch_size=batch_size, gpu=use_gpu,
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

    def train(
        self,
        data_dir: str,
        train_filename: str,
        dev_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        batch_size: int = 10,
        n_epochs: int = 2,
        learning_rate: float = 1e-5,
        max_seq_len: Optional[int] = None,
        warmup_proportion: float = 0.2,
        dev_split: float = 0,
        evaluate_every: int = 300,
        save_dir: Optional[str] = None,
        num_processes: Optional[int] = None,
        use_amp: str = None,
    ):
        """
        Fine-tune a model on a QA dataset. Options:

        - Take a plain language model (e.g. `bert-base-cased`) and train it for QA (e.g. on SQuAD data)
        - Take a QA model (e.g. `deepset/bert-base-cased-squad2`) and fine-tune it for your domain (e.g. using your labels collected via the haystack annotation tool)

        :param data_dir: Path to directory containing your training data in SQuAD style
        :param train_filename: Filename of training data
        :param dev_filename: Filename of dev / eval data
        :param test_filename: Filename of test data
        :param dev_split: Instead of specifying a dev_filename, you can also specify a ratio (e.g. 0.1) here
                          that gets split off from training data for eval.
        :param use_gpu: Whether to use GPU (if available)
        :param batch_size: Number of samples the model receives in one batch for training
        :param n_epochs: Number of iterations on the whole training data set
        :param learning_rate: Learning rate of the optimizer
        :param max_seq_len: Maximum text length (in tokens). Everything longer gets cut down.
        :param warmup_proportion: Proportion of training steps until maximum learning rate is reached.
                                  Until that point LR is increasing linearly. After that it's decreasing again linearly.
                                  Options for different schedules are available in FARM.
        :param evaluate_every: Evaluate the model every X steps on the hold-out eval dataset
        :param save_dir: Path to store the final model
        :param num_processes: The number of processes for `multiprocessing.Pool` during preprocessing.
                              Set to value of 1 to disable multiprocessing. When set to 1, you cannot split away a dev set from train set.
                              Set to None to use all CPU cores minus one.
        :param use_amp: Optimization level of NVIDIA's automatic mixed precision (AMP). The higher the level, the faster the model.
                        Available options:
                        None (Don't use AMP)
                        "O0" (Normal FP32 training)
                        "O1" (Mixed Precision => Recommended)
                        "O2" (Almost FP16)
                        "O3" (Pure FP16).
                        See details on: https://nvidia.github.io/apex/amp.html
        :return: None
        """

        if dev_filename:
            dev_split = 0

        if num_processes is None:
            num_processes = multiprocessing.cpu_count() - 1 or 1

        set_all_seeds(seed=42)

        # For these variables, by default, we use the value set when initializing the FARMReader.
        # These can also be manually set when train() is called if you want a different value at train vs inference
        if use_gpu is None:
            use_gpu = self.use_gpu
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        device, n_gpu = initialize_device_settings(use_cuda=use_gpu,use_amp=use_amp)

        if not save_dir:
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
            test_filename=test_filename,
            data_dir=Path(data_dir),
        )

        # 2. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them
        # and calculates a few descriptive statistics of our datasets
        data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False, max_processes=num_processes)

        # Quick-fix until this is fixed upstream in FARM:
        # We must avoid applying DataParallel twice (once when loading the inferencer,
        # once when calling initalize_optimizer)
        self.inferencer.model.save("tmp_model")
        model = BaseAdaptiveModel.load(load_dir="tmp_model", device=device, strict=True)
        shutil.rmtree('tmp_model')

        # 3. Create an optimizer and pass the already initialized model
        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            # model=self.inferencer.model,
            learning_rate=learning_rate,
            schedule_opts={"name": "LinearWarmup", "warmup_proportion": warmup_proportion},
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=n_epochs,
            device=device,
            use_amp=use_amp,
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
            use_amp=use_amp,
        )


        # 5. Let it grow!
        self.inferencer.model = trainer.train()
        self.save(Path(save_dir))

    def update_parameters(
        self,
        context_window_size: Optional[int] = None,
        no_ans_boost: Optional[float] = None,
        return_no_answer: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
        doc_stride: Optional[int] = None,
    ):
        """
        Hot update parameters of a loaded Reader. It may not to be safe when processing concurrent requests.
        """
        if no_ans_boost is not None:
            self.inferencer.model.prediction_heads[0].no_ans_boost = no_ans_boost
        if return_no_answer is not None:
            self.return_no_answers = return_no_answer
        if doc_stride is not None:
            self.inferencer.processor.doc_stride = doc_stride
        if context_window_size is not None:
            self.inferencer.model.prediction_heads[0].context_window_size = context_window_size
        if max_seq_len is not None:
            self.inferencer.processor.max_seq_len = max_seq_len
            self.max_seq_len = max_seq_len

    def save(self, directory: Path):
        """
        Saves the Reader model so that it can be reused at a later point in time.

        :param directory: Directory where the Reader model should be saved
        """
        logger.info(f"Saving reader model to {directory}")
        self.inferencer.model.save(directory)
        self.inferencer.processor.save(directory)

    def predict_batch(self, query_doc_list: List[dict], top_k: int = None, batch_size: int = None):
        """
        Use loaded QA model to find answers for a list of queries in each query's supplied list of Document.

        Returns list of dictionaries containing answers sorted by (desc.) probability

        :param query_doc_list: List of dictionaries containing queries with their retrieved documents
        :param top_k: The maximum number of answers to return for each query
        :param batch_size: Number of samples the model receives in one batch for inference
        :return: List of dictionaries containing query and answers
        """

        # convert input to FARM format
        inputs = []
        number_of_docs = []
        labels = []

        # build input objects for inference_from_objects
        for query_with_docs in query_doc_list:
            documents = query_with_docs["docs"]
            query = query_with_docs["question"]
            labels.append(query)
            number_of_docs.append(len(documents))

            for doc in documents:
                cur = QAInput(doc_text=doc.text,
                              questions=Question(text=query.question,
                                                 uid=doc.id))
                inputs.append(cur)

        self.inferencer.batch_size = batch_size
        # make predictions on all document-query pairs
        predictions = self.inferencer.inference_from_objects(
            objects=inputs, return_json=False, multiprocessing_chunksize=1
        )

        # group predictions together
        grouped_predictions = []
        left_idx = 0
        right_idx = 0
        for number in number_of_docs:
            right_idx = left_idx + number
            grouped_predictions.append(predictions[left_idx:right_idx])
            left_idx = right_idx

        result = []
        for idx, group in enumerate(grouped_predictions):
            answers, max_no_ans_gap = self._extract_answers_of_predictions(group, top_k)
            query = group[0].question
            cur_label = labels[idx]
            result.append({
                "query": query,
                "no_ans_gap": max_no_ans_gap,
                "answers": answers,
                "label": cur_label
            })

        return result

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a query in the supplied list of Document.

        Returns dictionaries containing answers sorted by (desc.) probability.
        Example:
         ```python
            |{
            |    'query': 'Who is the father of Arya Stark?',
            |    'answers':[
            |                 {'answer': 'Eddard,',
            |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
            |                 'offset_answer_start': 147,
            |                 'offset_answer_end': 154,
            |                 'probability': 0.9787139466668613,
            |                 'score': None,
            |                 'document_id': '1337'
            |                 },...
            |              ]
            |}
         ```

        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """

        # convert input to FARM format
        inputs = []
        for doc in documents:
            cur = QAInput(doc_text=doc.text,
                          questions=Question(text=query,
                                             uid=doc.id))
            inputs.append(cur)

        # get answers from QA model
        # TODO: Need fix in FARM's `to_dict` function of `QAInput` class
        predictions = self.inferencer.inference_from_objects(
            objects=inputs, return_json=False, multiprocessing_chunksize=1
        )
        # assemble answers from all the different documents & format them.
        answers, max_no_ans_gap = self._extract_answers_of_predictions(predictions, top_k)
        result = {"query": query,
                  "no_ans_gap": max_no_ans_gap,
                  "answers": answers}

        return result

    def eval_on_file(self, data_dir: str, test_filename: str, device: str):
        """
        Performs evaluation on a SQuAD-formatted file.
        Returns a dict containing the following metrics:
            - "EM": exact match score
            - "f1": F1-Score
            - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

        :param data_dir: The directory in which the test set can be found
        :type data_dir: Path or str
        :param test_filename: The name of the file containing the test data in SQuAD format.
        :type test_filename: str
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
        :type device: str
        """
        eval_processor = SquadProcessor(
            tokenizer=self.inferencer.processor.tokenizer,
            max_seq_len=self.inferencer.processor.max_seq_len,
            label_list=self.inferencer.processor.tasks["question_answering"]["label_list"],
            metric=self.inferencer.processor.tasks["question_answering"]["metric"],
            train_filename=None,
            dev_filename=None,
            dev_split=0,
            test_filename=test_filename,
            data_dir=Path(data_dir),
        )

        data_silo = DataSilo(processor=eval_processor, batch_size=self.inferencer.batch_size, distributed=False)
        data_loader = data_silo.get_data_loader("test")

        evaluator = Evaluator(data_loader=data_loader, tasks=eval_processor.tasks, device=device)

        eval_results = evaluator.eval(self.inferencer.model)
        results = {
            "EM": eval_results[0]["EM"],
            "f1": eval_results[0]["f1"],
            "top_n_accuracy": eval_results[0]["top_n_accuracy"]
        }
        return results

    def eval(
            self,
            document_store: BaseDocumentStore,
            device: str,
            label_index: str = "label",
            doc_index: str = "eval_document",
            label_origin: str = "gold_label",
    ):
        """
        Performs evaluation on evaluation documents in the DocumentStore.
        Returns a dict containing the following metrics:
              - "EM": Proportion of exact matches of predicted answers with their corresponding correct answers
              - "f1": Average overlap between predicted answers and their corresponding correct answers
              - "top_n_accuracy": Proportion of predicted answers that overlap with correct answer

        :param document_store: DocumentStore containing the evaluation documents
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
        :param label_index: Index/Table name where labeled questions are stored
        :param doc_index: Index/Table name where documents that are used for evaluation are stored
        """

        if self.top_k_per_candidate != 4:
            logger.info(f"Performing Evaluation using top_k_per_candidate = {self.top_k_per_candidate} \n"
                        f"and consequently, QuestionAnsweringPredictionHead.n_best = {self.top_k_per_candidate + 1}. \n"
                        f"This deviates from FARM's default where QuestionAnsweringPredictionHead.n_best = 5")

        # extract all questions for evaluation
        filters = {"origin": [label_origin]}

        labels = document_store.get_all_labels(index=label_index, filters=filters)

        # Aggregate all answer labels per question
        aggregated_per_doc = defaultdict(list)
        for label in labels:
            if not label.document_id:
                logger.error(f"Label does not contain a document_id")
                continue
            aggregated_per_doc[label.document_id].append(label)

        # Create squad style dicts
        d: Dict[str, Any] = {}
        all_doc_ids = [x.id for x in document_store.get_all_documents(doc_index)]
        for doc_id in all_doc_ids:
            doc = document_store.get_document_by_id(doc_id, index=doc_index)
            if not doc:
                logger.error(f"Document with the ID '{doc_id}' is not present in the document store.")
                continue
            d[str(doc_id)] = {
                "context": doc.text
            }
            # get all questions / answers
            aggregated_per_question: Dict[str, Any] = defaultdict(list)
            if doc_id in aggregated_per_doc:
                for label in aggregated_per_doc[doc_id]:
                    # add to existing answers
                    if label.question in aggregated_per_question.keys():
                        if label.offset_start_in_doc == 0 and label.answer == "":
                            continue
                        else:
                            # Hack to fix problem where duplicate questions are merged by doc_store processing creating a QA example with 8 annotations > 6 annotation max
                            if len(aggregated_per_question[label.question]["answers"]) >= 6:
                                continue
                            aggregated_per_question[label.question]["answers"].append({
                                        "text": label.answer,
                                        "answer_start": label.offset_start_in_doc})
                            aggregated_per_question[label.question]["is_impossible"] = False
                    # create new one
                    else:
                        # We don't need to create an answer dict if is_impossible / no_answer
                        if label.offset_start_in_doc == 0 and label.answer == "":
                            aggregated_per_question[label.question] = {
                                "id": str(hash(str(doc_id) + label.question)),
                                "question": label.question,
                                "answers": [],
                                "is_impossible": True
                            }
                        else:
                            aggregated_per_question[label.question] = {
                                "id": str(hash(str(doc_id)+label.question)),
                                "question": label.question,
                                "answers": [{
                                        "text": label.answer,
                                        "answer_start": label.offset_start_in_doc}],
                                "is_impossible": False
                            }

            # Get rid of the question key again (after we aggregated we don't need it anymore)
            d[str(doc_id)]["qas"] = [v for v in aggregated_per_question.values()]

        # Convert input format for FARM
        farm_input = [v for v in d.values()]
        n_queries = len([y for x in farm_input for y in x["qas"]])

        # Create DataLoader that can be passed to the Evaluator
        tic = perf_counter()
        indices = range(len(farm_input))
        dataset, tensor_names, problematic_ids = self.inferencer.processor.dataset_from_dicts(farm_input, indices=indices)
        data_loader = NamedDataLoader(dataset=dataset, batch_size=self.inferencer.batch_size, tensor_names=tensor_names)

        evaluator = Evaluator(data_loader=data_loader, tasks=self.inferencer.processor.tasks, device=device)

        eval_results = evaluator.eval(self.inferencer.model)
        toc = perf_counter()
        reader_time = toc - tic
        results = {
            "EM": eval_results[0]["EM"],
            "f1": eval_results[0]["f1"],
            "top_n_accuracy": eval_results[0]["top_n_accuracy"],
            "top_n": self.inferencer.model.prediction_heads[0].n_best,
            "reader_time": reader_time,
            "seconds_per_query": reader_time / n_queries
        }
        return results

    def _extract_answers_of_predictions(self, predictions: List[QAPred], top_k: Optional[int] = None):
        # Assemble answers from all the different documents and format them.
        # For the 'no answer' option, we collect all no_ans_gaps and decide how likely
        # a no answer is based on all no_ans_gaps values across all documents
        answers = []
        no_ans_gaps = []
        best_score_answer = 0

        for pred in predictions:
            answers_per_document = []
            no_ans_gaps.append(pred.no_answer_gap)
            for ans in pred.prediction:
                # skip 'no answers' here
                if self._check_no_answer(ans):
                    pass
                else:
                    cur = {
                        "answer": ans.answer,
                        "score": ans.score,
                        # just a pseudo prob for now
                        "probability": self._get_pseudo_prob(ans.score),
                        "context": ans.context_window,
                        "offset_start": ans.offset_answer_start - ans.offset_context_window_start,
                        "offset_end": ans.offset_answer_end - ans.offset_context_window_start,
                        "offset_start_in_doc": ans.offset_answer_start,
                        "offset_end_in_doc": ans.offset_answer_end,
                        "document_id": pred.id
                    }
                    answers_per_document.append(cur)

                    if ans.score > best_score_answer:
                        best_score_answer = ans.score

            # Only take n best candidates. Answers coming back from FARM are sorted with decreasing relevance
            answers += answers_per_document[:self.top_k_per_candidate]

        # calculate the score for predicting 'no answer', relative to our best positive answer score
        no_ans_prediction, max_no_ans_gap = self._calc_no_answer(no_ans_gaps, best_score_answer)
        if self.return_no_answers:
            answers.append(no_ans_prediction)

        # sort answers by score and select top-k
        answers = sorted(answers, key=lambda k: k["score"], reverse=True)
        answers = answers[:top_k]

        return answers, max_no_ans_gap

    @staticmethod
    def _get_pseudo_prob(score: float):
        return float(expit(np.asarray(score) / 8))

    @staticmethod
    def _check_no_answer(c: QACandidate):
        # check for correct value in "answer"
        if c.offset_answer_start == 0 and c.offset_answer_end == 0:
            if c.answer != "no_answer":
                logger.error("Invalid 'no_answer': Got a prediction for position 0, but answer string is not 'no_answer'")
        if c.answer == "no_answer":
            return True
        else:
            return False

    def predict_on_texts(self, question: str, texts: List[str], top_k: Optional[int] = None):
        """
        Use loaded QA model to find answers for a question in the supplied list of Document.
        Returns dictionaries containing answers sorted by (desc.) probability.
        Example:
         ```python
            |{
            |    'question': 'Who is the father of Arya Stark?',
            |    'answers':[
            |                 {'answer': 'Eddard,',
            |                 'context': " She travels with her father, Eddard, to King's Landing when he is ",
            |                 'offset_answer_start': 147,
            |                 'offset_answer_end': 154,
            |                 'probability': 0.9787139466668613,
            |                 'score': None,
            |                 'document_id': '1337'
            |                 },...
            |              ]
            |}
         ```

        :param question: Question string
        :param documents: List of documents as string type
        :param top_k: The maximum number of answers to return
        :return: Dict containing question and answers
        """
        documents = []
        for text in texts:
            documents.append(
                Document(
                    text=text
                )
            )
        predictions = self.predict(question, documents, top_k)
        return predictions

    @classmethod
    def convert_to_onnx(
            cls,
            model_name: str,
            output_path: Path,
            convert_to_float16: bool = False,
            quantize: bool = False,
            task_type: str = "question_answering",
            opset_version: int = 11
    ):
        """
        Convert a PyTorch BERT model to ONNX format and write to ./onnx-export dir. The converted ONNX model
        can be loaded with in the `FARMReader` using the export path as `model_name_or_path` param.

        Usage:

            `from haystack.reader.farm import FARMReader
            from pathlib import Path
            onnx_model_path = Path("roberta-onnx-model")
            FARMReader.convert_to_onnx(model_name="deepset/bert-base-cased-squad2", output_path=onnx_model_path)
            reader = FARMReader(onnx_model_path)`

        :param model_name: transformers model name
        :param output_path: Path to output the converted model
        :param convert_to_float16: Many models use float32 precision by default. With the half precision of float16,
                                   inference is faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs,
                                   float32 could still be be more performant.
        :param quantize: convert floating point number to integers
        :param task_type: Type of task for the model. Available options: "question_answering" or "embeddings".
        :param opset_version: ONNX opset version
        """
        AdaptiveModel.convert_to_onnx(
            model_name=model_name,
            output_path=output_path,
            task_type=task_type,
            convert_to_float16=convert_to_float16,
            quantize=quantize,
            opset_version=opset_version
        )

