import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import AutoModelForQuestionAnswering
from scipy.special import expit

from haystack.modeling.data_handler.samples import SampleBasket
from haystack.modeling.model.predictions import QACandidate, QAPred
from haystack.modeling.utils import try_get, all_gather_list


logger = logging.getLogger(__name__)


class PredictionHead(nn.Module):
    """
    Takes word embeddings from a language model and generates logits for a given task. Can also convert logits
    to loss and and logits to predictions.
    """

    subclasses = {}  # type: Dict

    def __init_subclass__(cls, **kwargs):
        """This automatically keeps track of all available subclasses.
        Enables generic load() for all specific PredictionHead implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def create(cls, prediction_head_name: str, layer_dims: List[int], class_weights=Optional[List[float]]):
        """
        Create subclass of Prediction Head.

        :param prediction_head_name: Classname (exact string!) of prediction head we want to create
        :param layer_dims: describing the feed forward block structure, e.g. [768,2]
        :param class_weights: The loss weighting to be assigned to certain label classes during training.
           Used to correct cases where there is a strong class imbalance.
        :return: Prediction Head of class prediction_head_name
        """
        # TODO make we want to make this more generic.
        #  1. Class weights is not relevant for all heads.
        #  2. Layer weights impose FF structure, maybe we want sth else later
        # Solution: We could again use **kwargs
        return cls.subclasses[prediction_head_name](layer_dims=layer_dims, class_weights=class_weights)

    def save_config(self, save_dir: Union[str, Path], head_num: int = 0):
        """
        Saves the config as a json file.

        :param save_dir: Path to save config to
        :param head_num: Which head to save
        """
        # updating config in case the parameters have been changed
        self.generate_config()
        output_config_file = Path(save_dir) / f"prediction_head_{head_num}_config.json"
        with open(output_config_file, "w") as file:
            json.dump(self.config, file)

    def save(self, save_dir: Union[str, Path], head_num: int = 0):
        """
        Saves the prediction head state dict.

        :param save_dir: path to save prediction head to
        :param head_num: which head to save
        """
        output_model_file = Path(save_dir) / f"prediction_head_{head_num}.bin"
        torch.save(self.state_dict(), output_model_file)
        self.save_config(save_dir, head_num)

    def generate_config(self):
        """
        Generates config file from Class parameters (only for sensible config parameters).
        """
        config = {}
        for key, value in self.__dict__.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            if _is_json(value) and key[0] != "_":
                config[key] = value
            if self.task_name == "text_similarity" and key == "similarity_function":
                config["similarity_function"] = value
        config["name"] = self.__class__.__name__
        config.pop("config", None)
        self.config = config

    @classmethod
    def load(cls, config_file: str, strict: bool = True, load_weights: bool = True):
        """
        Loads a Prediction Head. Infers the class of prediction head from config_file.

        :param config_file: location where corresponding config is stored
        :param strict: whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
                       Set to `False` for backwards compatibility with PHs saved with older version of Haystack.
        :param load_weights: whether to load weights of the prediction head
        :return: PredictionHead
        :rtype: PredictionHead[T]
        """
        with open(config_file) as f:
            config = json.load(f)
        prediction_head = cls.subclasses[config["name"]](**config)
        if load_weights:
            model_file = cls._get_model_file(config_file=config_file)
            logger.info("Loading prediction head from %s", model_file)
            prediction_head.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")), strict=strict)
        return prediction_head

    def logits_to_loss(self, logits, labels):
        """
        Implement this function in your special Prediction Head.
        Should combine logits and labels with a loss fct to a per sample loss.

        :param logits: logits, can vary in shape and type, depending on task
        :param labels: labels, can vary in shape and type, depending on task
        :return: per sample loss as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def logits_to_preds(self, logits, span_mask, start_of_word, seq_2_start_t, max_answer_length, **kwargs):
        """
        Implement this function in your special Prediction Head.
        Should combine turn logits into predictions.

        :param logits: logits, can vary in shape and type, depending on task
        :return: predictions as a torch.tensor of shape [batch_size]
        """
        raise NotImplementedError()

    def prepare_labels(self, **kwargs):
        """
        Some prediction heads need additional label conversion.

        :param kwargs: placeholder for passing generic parameters
        :return: labels in the right format
        :rtype: object
        """
        # TODO maybe just return **kwargs to not force people to implement this
        raise NotImplementedError()

    def resize_input(self, input_dim):
        """
        This function compares the output dimensionality of the language model against the input dimensionality
        of the prediction head. If there is a mismatch, the prediction head will be resized to fit.
        """
        # Note on pylint disable
        # self.feed_forward's existence seems to be a condition for its own initialization
        # within this class, which is clearly wrong. The only way this code could ever be called is
        # thanks to subclasses initializing self.feed_forward somewhere else; however, this is a
        # very implicit requirement for subclasses, and in general bad design. FIXME when possible.
        if "feed_forward" not in dir(self):
            return
        else:
            old_dims = self.feed_forward.layer_dims  # pylint: disable=access-member-before-definition
            if input_dim == old_dims[0]:
                return
            new_dims = [input_dim] + old_dims[1:]
            logger.info(
                "Resizing input dimensions of %s (%s) from %s to %s to match language model",
                type(self).__name__,
                self.task_name,
                old_dims,
                new_dims,
            )
            self.feed_forward = FeedForwardBlock(new_dims)
            self.layer_dims[0] = input_dim
            self.feed_forward.layer_dims[0] = input_dim

    @classmethod
    def _get_model_file(cls, config_file: Union[str, Path]):
        if "config.json" in str(config_file) and "prediction_head" in str(config_file):
            head_num = int("".join([char for char in os.path.basename(config_file) if char.isdigit()]))
            model_file = Path(os.path.dirname(config_file)) / f"prediction_head_{head_num}.bin"
        else:
            raise ValueError(f"This doesn't seem to be a proper prediction_head config file: '{config_file}'")
        return model_file

    def _set_name(self, name):
        self.task_name = name


class FeedForwardBlock(nn.Module):
    """
    A feed forward neural network of variable depth and width.
    """

    def __init__(self, layer_dims: List[int], **kwargs):
        # Todo: Consider having just one input argument
        super(FeedForwardBlock, self).__init__()
        self.layer_dims = layer_dims
        # If read from config the input will be string
        n_layers = len(layer_dims) - 1
        layers_all = []
        # TODO: IS this needed?
        self.output_size = layer_dims[-1]

        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X: torch.Tensor):
        logits = self.feed_forward(X)
        return logits


class QuestionAnsweringHead(PredictionHead):
    """
    A question answering head predicts the start and end of the answer on token level.

    In addition, it gives a score for the prediction so that multiple answers can be ranked.
    There are three different kinds of scores available:
    1) (standard) score: the sum of the logits of the start and end index. This score is unbounded because the logits are unbounded.
    It is the default for ranking answers.
    2) confidence score: also based on the logits of the start and end index but scales them to the interval 0 to 1 and incorporates no_answer.
    It can be used for ranking by setting use_confidence_scores_for_ranking to True
    3) calibrated confidence score: same as 2) but divides the logits by a learned temperature_for_confidence parameter
    so that the confidence scores are closer to the model's achieved accuracy. It can be used for ranking by setting
    use_confidence_scores_for_ranking to True and temperature_for_confidence!=1.0. See examples/question_answering_confidence.py for more details.
    """

    def __init__(
        self,
        layer_dims: Optional[List[int]] = None,
        task_name: str = "question_answering",
        no_ans_boost: float = 0.0,
        context_window_size: int = 100,
        n_best: int = 5,
        n_best_per_sample: Optional[int] = None,
        duplicate_filtering: int = -1,
        temperature_for_confidence: float = 1.0,
        use_confidence_scores_for_ranking: bool = True,
        use_no_answer_legacy_confidence: bool = False,
        **kwargs,
    ):
        """
        :param layer_dims: dimensions of Feed Forward block, e.g. [768,2] used by default, for adjusting to BERT embedding. Output should be always 2
        :param kwargs: placeholder for passing generic parameters
        :param no_ans_boost: How much the no_answer logit is boosted/increased.
                             The higher the value, the more likely a "no answer possible given the input text" is returned by the model
        :param context_window_size: The size, in characters, of the window around the answer span that is used when displaying the context around the answer.
        :param n_best: The number of positive answer spans for each document.
        :param n_best_per_sample: num candidate answer spans to consider from each passage. Each passage also returns "no answer" info.
                                  This is decoupled from n_best on document level, since predictions on passage level are very similar.
                                  It should have a low value
        :param duplicate_filtering: Answers are filtered based on their position. Both start and end position of the answers are considered.
                                    The higher the value, answers that are more apart are filtered out. 0 corresponds to exact duplicates. -1 turns off duplicate removal.
        :param temperature_for_confidence: The divisor that is used to scale logits to calibrate confidence scores
        :param use_confidence_scores_for_ranking: Whether to sort answers by confidence score (normalized between 0 and 1)(default) or by standard score (unbounded).
        :param use_no_answer_legacy_confidence: Whether to use the legacy confidence definition for no_answer: difference between the best overall answer confidence and the no_answer gap confidence.
                                                Otherwise we use the no_answer score normalized to a range of [0,1] by an expit function (default).
        """
        if layer_dims is None:
            layer_dims = [768, 2]
        super(QuestionAnsweringHead, self).__init__()
        if len(kwargs) > 0:
            logger.warning(
                "Some unused parameters are passed to the QuestionAnsweringHead. Might not be a problem. Params: %s",
                json.dumps(kwargs),
            )
        self.layer_dims = layer_dims
        assert self.layer_dims[-1] == 2
        self.feed_forward = FeedForwardBlock(self.layer_dims)
        logger.debug("Prediction head initialized with size %s", self.layer_dims)
        self.num_labels = self.layer_dims[-1]
        self.ph_output_type = "per_token_squad"
        self.model_type = "span_classification"  # predicts start and end token of answer
        self.task_name = task_name
        self.no_ans_boost = no_ans_boost
        self.context_window_size = context_window_size
        self.n_best = n_best
        if n_best_per_sample:
            self.n_best_per_sample = n_best_per_sample
        else:
            # increasing n_best_per_sample to n_best ensures that there are n_best predictions in total
            # otherwise this might not be the case for very short documents with only one "sample"
            self.n_best_per_sample = n_best
        self.duplicate_filtering = duplicate_filtering
        self.generate_config()
        self.temperature_for_confidence = nn.Parameter(torch.ones(1) * temperature_for_confidence)
        self.use_confidence_scores_for_ranking = use_confidence_scores_for_ranking
        self.use_no_answer_legacy_confidence = use_no_answer_legacy_confidence

    @classmethod
    def load(  # type: ignore
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        revision: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        """
        Load a prediction head from a saved Haystack or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a Haystack prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - bert-large-uncased-whole-word-masking-finetuned-squad
                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        """
        if (
            os.path.exists(pretrained_model_name_or_path)
            and "config.json" in str(pretrained_model_name_or_path)
            and "prediction_head" in str(pretrained_model_name_or_path)
        ):
            # a) Haystack style
            super(QuestionAnsweringHead, cls).load(str(pretrained_model_name_or_path))
        else:
            # b) transformers style
            # load all weights from model
            full_qa_model = AutoModelForQuestionAnswering.from_pretrained(
                pretrained_model_name_or_path, revision=revision, use_auth_token=use_auth_token, **kwargs
            )
            # init empty head
            head = cls(layer_dims=[full_qa_model.config.hidden_size, 2], task_name="question_answering")
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_qa_model.qa_outputs.state_dict())
            del full_qa_model

        return head

    def forward(self, X: torch.Tensor):
        """
        One forward pass through the prediction head model, starting with language model output on token level.
        """
        logits = self.feed_forward(X)
        return self.temperature_scale(logits)

    def logits_to_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Combine predictions and labels to a per sample loss.
        """
        # todo explain how we only use first answer for train
        # labels.shape =  [batch_size, n_max_answers, 2]. n_max_answers is by default 6 since this is the
        # most that occurs in the SQuAD dev set. The 2 in the final dimension corresponds to [start, end]
        start_position = labels[:, 0, 0]
        end_position = labels[:, 0, 1]

        # logits is of shape [batch_size, max_seq_len, 2]. Like above, the final dimension corresponds to [start, end]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Squeeze final singleton dimensions
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)

        ignored_index = start_logits.size(1)
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        # Workaround for pytorch bug in version 1.10.0 with non-continguous tensors
        # Fix expected in 1.10.1 based on https://github.com/pytorch/pytorch/pull/64954
        start_logits = start_logits.contiguous()
        start_position = start_position.contiguous()
        end_logits = end_logits.contiguous()
        end_position = end_position.contiguous()

        loss_fct = CrossEntropyLoss(reduction="none")
        start_loss = loss_fct(start_logits, start_position)
        end_loss = loss_fct(end_logits, end_position)
        per_sample_loss = (start_loss + end_loss) / 2
        return per_sample_loss

    def temperature_scale(self, logits: torch.Tensor):
        return torch.div(logits, self.temperature_for_confidence)

    def calibrate_conf(self, logits, label_all):
        """
        Learning a temperature parameter to apply temperature scaling to calibrate confidence scores
        """
        logits = torch.cat(logits, dim=0)

        # To handle no_answer labels correctly (-1,-1), we set their start_position to 0. The logit at index 0 also refers to no_answer
        # TODO some language models do not have the CLS token at position 0. For these models, we need to map start_position==-1 to the index of CLS token
        start_position = [label[0][0] if label[0][0] >= 0 else 0 for label in label_all]
        end_position = [label[0][1] if label[0][1] >= 0 else 0 for label in label_all]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_position = torch.tensor(start_position)
        if len(start_position.size()) > 1:
            start_position = start_position.squeeze(-1)
        end_position = torch.tensor(end_position)
        if len(end_position.size()) > 1:
            end_position = end_position.squeeze(-1)

        ignored_index = start_logits.size(1) - 1
        start_position.clamp_(0, ignored_index)
        end_position.clamp_(0, ignored_index)

        nll_criterion = CrossEntropyLoss()

        optimizer = optim.LBFGS([self.temperature_for_confidence], lr=0.01, max_iter=50)

        def eval_start_end_logits():
            loss = nll_criterion(
                self.temperature_scale(start_logits), start_position.to(device=start_logits.device)
            ) + nll_criterion(self.temperature_scale(end_logits), end_position.to(device=end_logits.device))
            loss.backward()
            return loss

        optimizer.step(eval_start_end_logits)

    def logits_to_preds(
        self,
        logits: torch.Tensor,
        span_mask: torch.Tensor,
        start_of_word: torch.Tensor,
        seq_2_start_t: torch.Tensor,
        max_answer_length: int = 1000,
        **kwargs,
    ):
        """
        Get the predicted index of start and end token of the answer. Note that the output is at token level
        and not word level. Note also that these logits correspond to the tokens of a sample
        (i.e. special tokens, question tokens, passage_tokens)
        """

        # Will be populated with the top-n predictions of each sample in the batch
        # shape = batch_size x ~top_n
        # Note that ~top_n = n   if no_answer is     within the top_n predictions
        #           ~top_n = n+1 if no_answer is not within the top_n predictions
        all_top_n = []

        # logits is of shape [batch_size, max_seq_len, 2]. The final dimension corresponds to [start, end]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Calculate a few useful variables
        batch_size = start_logits.size()[0]
        max_seq_len = start_logits.shape[1]  # target dim

        # get scores for all combinations of start and end logits => candidate answers
        start_matrix = start_logits.unsqueeze(2).expand(-1, -1, max_seq_len)
        end_matrix = end_logits.unsqueeze(1).expand(-1, max_seq_len, -1)
        start_end_matrix = start_matrix + end_matrix

        # disqualify answers where end < start
        # (set the lower triangular matrix to low value, excluding diagonal)
        indices = torch.tril_indices(max_seq_len, max_seq_len, offset=-1, device=start_end_matrix.device)
        start_end_matrix[:, indices[0][:], indices[1][:]] = -888

        # disqualify answers where answer span is greater than max_answer_length
        # (set the upper triangular matrix to low value, excluding diagonal)
        indices_long_span = torch.triu_indices(
            max_seq_len, max_seq_len, offset=max_answer_length, device=start_end_matrix.device
        )
        start_end_matrix[:, indices_long_span[0][:], indices_long_span[1][:]] = -777

        # disqualify answers where start=0, but end != 0
        start_end_matrix[:, 0, 1:] = -666

        # Turn 1d span_mask vectors into 2d span_mask along 2 different axes
        # span mask has:
        #   0 for every position that is never a valid start or end index (question tokens, mid and end special tokens, padding)
        #   1 everywhere else
        span_mask_start = span_mask.unsqueeze(2).expand(-1, -1, max_seq_len)
        span_mask_end = span_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        span_mask_2d = span_mask_start + span_mask_end
        # disqualify spans where either start or end is on an invalid token
        invalid_indices = torch.nonzero((span_mask_2d != 2), as_tuple=True)
        start_end_matrix[invalid_indices[0][:], invalid_indices[1][:], invalid_indices[2][:]] = -999

        # Sort the candidate answers by their score. Sorting happens on the flattened matrix.
        # flat_sorted_indices.shape: (batch_size, max_seq_len^2, 1)
        flat_scores = start_end_matrix.view(batch_size, -1)
        flat_sorted_indices_2d = flat_scores.sort(descending=True)[1]
        flat_sorted_indices = flat_sorted_indices_2d.unsqueeze(2)

        # The returned indices are then converted back to the original dimensionality of the matrix.
        # sorted_candidates.shape : (batch_size, max_seq_len^2, 2)
        start_indices = torch.div(flat_sorted_indices, max_seq_len, rounding_mode="trunc")
        end_indices = flat_sorted_indices % max_seq_len
        sorted_candidates = torch.cat((start_indices, end_indices), dim=2)

        # Get the n_best candidate answers for each sample
        sorted_candidates = sorted_candidates.cpu().numpy()
        start_end_matrix = start_end_matrix.cpu().numpy()
        for sample_idx in range(batch_size):
            sample_top_n = self.get_top_candidates(
                sorted_candidates[sample_idx],
                start_end_matrix[sample_idx],
                sample_idx,
                start_matrix=start_matrix[sample_idx],
                end_matrix=end_matrix[sample_idx],
            )
            all_top_n.append(sample_top_n)

        return all_top_n

    def get_top_candidates(self, sorted_candidates, start_end_matrix, sample_idx: int, start_matrix, end_matrix):
        """
        Returns top candidate answers as a list of Span objects. Operates on a matrix of summed start and end logits.
        This matrix corresponds to a single sample (includes special tokens, question tokens, passage tokens).
        This method always returns a list of len n_best + 1 (it is comprised of the n_best positive answers along with the one no_answer)
        """
        # Initialize some variables
        top_candidates: List[QACandidate] = []
        n_candidates = sorted_candidates.shape[0]
        start_idx_candidates = set()
        end_idx_candidates = set()

        start_matrix_softmax_start = torch.softmax(start_matrix[:, 0], dim=-1).cpu().numpy()
        end_matrix_softmax_end = torch.softmax(end_matrix[0, :], dim=-1).cpu().numpy()
        # Iterate over all candidates and break when we have all our n_best candidates
        for candidate_idx in range(n_candidates):
            # Retrieve candidate's indices
            start_idx = sorted_candidates[candidate_idx, 0]
            end_idx = sorted_candidates[candidate_idx, 1]
            # Ignore no_answer scores which will be extracted later in this method
            if start_idx == 0 and end_idx == 0:
                continue
            if self.duplicate_filtering > -1 and (start_idx in start_idx_candidates or end_idx in end_idx_candidates):
                continue
            score = start_end_matrix[start_idx, end_idx]
            confidence = (
                (start_matrix_softmax_start[start_idx] + end_matrix_softmax_end[end_idx]) / 2
                if score > -500
                else np.exp(score / 10)  # disqualify answers according to scores in logits_to_preds()
            )
            top_candidates.append(
                QACandidate(
                    offset_answer_start=start_idx,
                    offset_answer_end=end_idx,
                    score=score,
                    answer_type="span",
                    offset_unit="token",
                    aggregation_level="passage",
                    passage_id=str(sample_idx),
                    confidence=confidence,
                )
            )
            if self.duplicate_filtering > -1:
                for i in range(0, self.duplicate_filtering + 1):
                    start_idx_candidates.add(start_idx + i)
                    start_idx_candidates.add(start_idx - i)
                    end_idx_candidates.add(end_idx + i)
                    end_idx_candidates.add(end_idx - i)

            # Only check if we have enough candidates after adding new candidate to the list
            if len(top_candidates) == self.n_best_per_sample:
                break

        no_answer_score = start_end_matrix[0, 0]
        no_answer_confidence = (start_matrix_softmax_start[0] + end_matrix_softmax_end[0]) / 2
        top_candidates.append(
            QACandidate(
                offset_answer_start=0,
                offset_answer_end=0,
                score=no_answer_score,
                answer_type="no_answer",
                offset_unit="token",
                aggregation_level="passage",
                passage_id=None,
                confidence=no_answer_confidence,
            )
        )

        return top_candidates

    def formatted_preds(
        self, preds: List[QACandidate], baskets: List[SampleBasket], logits: Optional[torch.Tensor] = None, **kwargs
    ):
        """
        Takes a list of passage level predictions, each corresponding to one sample, and converts them into document level
        predictions. Leverages information in the SampleBaskets. Assumes that we are being passed predictions from
        ALL samples in the one SampleBasket i.e. all passages of a document. Logits should be None, because we have
        already converted the logits to predictions before calling formatted_preds.
        (see Inferencer._get_predictions_and_aggregate()).
        """
        # Unpack some useful variables
        # passage_start_t is the token index of the passage relative to the document (usually a multiple of doc_stride)
        # seq_2_start_t is the token index of the first token in passage relative to the input sequence (i.e. number of
        # special tokens and question tokens that come before the passage tokens)
        if logits or preds is None:
            logger.error(
                "QuestionAnsweringHead.formatted_preds() expects preds as input and logits to be None \
                            but was passed something different"
            )
        samples = [s for b in baskets for s in b.samples]  # type: ignore
        ids = [s.id for s in samples]
        passage_start_t = [s.features[0]["passage_start_t"] for s in samples]  # type: ignore
        seq_2_start_t = [s.features[0]["seq_2_start_t"] for s in samples]  # type: ignore

        # Aggregate passage level predictions to create document level predictions.
        # This method assumes that all passages of each document are contained in preds
        # i.e. that there are no incomplete documents. The output of this step
        # are prediction spans
        preds_d = self.aggregate_preds(preds, passage_start_t, ids, seq_2_start_t)

        # Separate top_preds list from the no_ans_gap float.
        top_preds, no_ans_gaps = zip(*preds_d)

        # Takes document level prediction spans and returns string predictions
        doc_preds = self.to_qa_preds(top_preds, no_ans_gaps, baskets)

        return doc_preds

    def to_qa_preds(self, top_preds, no_ans_gaps, baskets):
        """
        Groups Span objects together in a QAPred object
        """
        ret = []

        # Iterate over each set of document level prediction
        for pred_d, no_ans_gap, basket in zip(top_preds, no_ans_gaps, baskets):
            # Unpack document offsets, clear text and id
            token_offsets = basket.raw["document_offsets"]
            pred_id = basket.id_external if basket.id_external else basket.id_internal

            # These options reflect the different input dicts that can be assigned to the basket
            # before any kind of normalization or preprocessing can happen
            question_names = ["question_text", "qas", "questions"]
            doc_names = ["document_text", "context", "text"]

            document_text = try_get(doc_names, basket.raw)
            question = self.get_question(question_names, basket.raw)
            ground_truth = self.get_ground_truth(basket)

            curr_doc_pred = QAPred(
                id=pred_id,
                prediction=pred_d,
                context=document_text,
                question=question,
                token_offsets=token_offsets,
                context_window_size=self.context_window_size,
                aggregation_level="document",
                ground_truth_answer=ground_truth,
                no_answer_gap=no_ans_gap,
            )

            ret.append(curr_doc_pred)
        return ret

    @staticmethod
    def get_ground_truth(basket: SampleBasket):
        if "answers" in basket.raw:
            return basket.raw["answers"]
        elif "annotations" in basket.raw:
            return basket.raw["annotations"]
        else:
            return None

    @staticmethod
    def get_question(question_names: List[str], raw_dict: Dict):
        # For NQ style dicts
        qa_name = None
        if "qas" in raw_dict:
            qa_name = "qas"
        elif "question" in raw_dict:
            qa_name = "question"
        if qa_name:
            if type(raw_dict[qa_name][0]) == dict:
                return raw_dict[qa_name][0]["question"]
        return try_get(question_names, raw_dict)

    def aggregate_preds(self, preds, passage_start_t, ids, seq_2_start_t=None, labels=None):
        """
        Aggregate passage level predictions to create document level predictions.
        This method assumes that all passages of each document are contained in preds
        i.e. that there are no incomplete documents. The output of this step
        are prediction spans. No answer is represented by a (-1, -1) span on the document level
        """
        # Initialize some variables
        n_samples = len(preds)
        all_basket_preds = {}
        all_basket_labels = {}

        # Iterate over the preds of each sample - remove final number which is the sample id and not needed for aggregation
        for sample_idx in range(n_samples):
            basket_id = ids[sample_idx]
            basket_id = basket_id.split("-")[:-1]
            basket_id = "-".join(basket_id)

            # curr_passage_start_t is the token offset of the current passage
            # It will always be a multiple of doc_stride
            curr_passage_start_t = passage_start_t[sample_idx]

            # This is to account for the fact that all model input sequences start with some special tokens
            # and also the question tokens before passage tokens.
            if seq_2_start_t:
                cur_seq_2_start_t = seq_2_start_t[sample_idx]
                curr_passage_start_t -= cur_seq_2_start_t

            # Converts the passage level predictions+labels to document level predictions+labels. Note
            # that on the passage level a no answer is (0,0) but at document level it is (-1,-1) since (0,0)
            # would refer to the first token of the document
            pred_d = self.pred_to_doc_idxs(preds[sample_idx], curr_passage_start_t)
            if labels:
                label_d = self.label_to_doc_idxs(labels[sample_idx], curr_passage_start_t)

            # Initialize the basket_id as a key in the all_basket_preds and all_basket_labels dictionaries
            if basket_id not in all_basket_preds:
                all_basket_preds[basket_id] = []
                all_basket_labels[basket_id] = []

            # Add predictions and labels to dictionary grouped by their basket_ids
            all_basket_preds[basket_id].append(pred_d)
            if labels:
                all_basket_labels[basket_id].append(label_d)

        # Pick n-best predictions and remove repeated labels
        all_basket_preds = {k: self.reduce_preds(v) for k, v in all_basket_preds.items()}
        if labels:
            all_basket_labels = {k: self.reduce_labels(v) for k, v in all_basket_labels.items()}

        # Return aggregated predictions in order as a list of lists
        keys = [k for k in all_basket_preds]
        aggregated_preds = [all_basket_preds[k] for k in keys]
        if labels:
            labels = [all_basket_labels[k] for k in keys]
            return aggregated_preds, labels
        else:
            return aggregated_preds

    @staticmethod
    def reduce_labels(labels):
        """
        Removes repeat answers. Represents a no answer label as (-1,-1)
        """
        positive_answers = [(start, end) for x in labels for start, end in x if not (start == -1 and end == -1)]
        if not positive_answers:
            return [(-1, -1)]
        else:
            return list(set(positive_answers))

    def reduce_preds(self, preds):
        """
        This function contains the logic for choosing the best answers from each passage. In the end, it
        returns the n_best predictions on the document level.
        """
        # Initialize variables
        passage_no_answer = []
        passage_best_score = []
        passage_best_confidence = []
        no_answer_scores = []
        no_answer_confidences = []
        n_samples = len(preds)

        # Iterate over the top predictions for each sample
        for sample_idx, sample_preds in enumerate(preds):
            best_pred = sample_preds[0]
            best_pred_score = best_pred.score
            best_pred_confidence = best_pred.confidence
            no_answer_score, no_answer_confidence = self.get_no_answer_score_and_confidence(sample_preds)
            no_answer_score += self.no_ans_boost
            # TODO we might want to apply some kind of a no_ans_boost to no_answer_confidence too
            no_answer = no_answer_score > best_pred_score
            passage_no_answer.append(no_answer)
            no_answer_scores.append(no_answer_score)
            no_answer_confidences.append(no_answer_confidence)
            passage_best_score.append(best_pred_score)
            passage_best_confidence.append(best_pred_confidence)

        # Get all predictions in flattened list and sort by score
        pos_answers_flat = []
        for sample_idx, passage_preds in enumerate(preds):
            for qa_candidate in passage_preds:
                if not (qa_candidate.offset_answer_start == -1 and qa_candidate.offset_answer_end == -1):
                    pos_answers_flat.append(
                        QACandidate(
                            offset_answer_start=qa_candidate.offset_answer_start,
                            offset_answer_end=qa_candidate.offset_answer_end,
                            score=qa_candidate.score,
                            answer_type=qa_candidate.answer_type,
                            offset_unit="token",
                            aggregation_level="document",
                            passage_id=str(sample_idx),
                            n_passages_in_doc=n_samples,
                            confidence=qa_candidate.confidence,
                        )
                    )

        # TODO add switch for more variation in answers, e.g. if varied_ans then never return overlapping answers
        pos_answer_dedup = self.deduplicate(pos_answers_flat)

        # This is how much no_ans_boost needs to change to turn a no_answer to a positive answer (or vice versa)
        no_ans_gap = -min(nas - pbs for nas, pbs in zip(no_answer_scores, passage_best_score))
        no_ans_gap_confidence = -min(nas - pbs for nas, pbs in zip(no_answer_confidences, passage_best_confidence))

        # "no answer" scores and positive answers scores are difficult to compare, because
        # + a positive answer score is related to a specific text qa_candidate
        # - a "no answer" score is related to all input texts
        # Thus we compute the "no answer" score relative to the best possible answer and adjust it by
        # the most significant difference between scores.
        # Most significant difference: change top prediction from "no answer" to answer (or vice versa)
        best_overall_positive_score = max(x.score for x in pos_answer_dedup)
        best_overall_positive_confidence = max(x.confidence for x in pos_answer_dedup)
        no_answer_pred = QACandidate(
            offset_answer_start=-1,
            offset_answer_end=-1,
            score=best_overall_positive_score - no_ans_gap,
            answer_type="no_answer",
            offset_unit="token",
            aggregation_level="document",
            passage_id=None,
            n_passages_in_doc=n_samples,
            confidence=best_overall_positive_confidence - no_ans_gap_confidence
            if self.use_no_answer_legacy_confidence
            else float(expit(np.asarray(best_overall_positive_score - no_ans_gap) / 8)),
        )

        # Add no answer to positive answers, sort the order and return the n_best
        n_preds = [no_answer_pred] + pos_answer_dedup
        n_preds_sorted = sorted(
            n_preds, key=lambda x: x.confidence if self.use_confidence_scores_for_ranking else x.score, reverse=True
        )
        n_preds_reduced = n_preds_sorted[: self.n_best]
        return n_preds_reduced, no_ans_gap

    @staticmethod
    def deduplicate(flat_pos_answers):
        # Remove duplicate spans that might be twice predicted in two different passages
        seen = {}
        for qa_answer in flat_pos_answers:
            if (qa_answer.offset_answer_start, qa_answer.offset_answer_end) not in seen:
                seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)] = qa_answer
            else:
                seen_score = seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)].score
                if qa_answer.score > seen_score:
                    seen[(qa_answer.offset_answer_start, qa_answer.offset_answer_end)] = qa_answer
        return list(seen.values())

    @staticmethod
    def get_no_answer_score_and_confidence(preds):
        for qa_answer in preds:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            score = qa_answer.score
            confidence = qa_answer.confidence
            if start == -1 and end == -1:
                return score, confidence
        raise Exception

    @staticmethod
    def pred_to_doc_idxs(pred, passage_start_t):
        """
        Converts the passage level predictions to document level predictions. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) qa_answer but will instead be represented by (-1, -1)
        """
        new_pred = []
        for qa_answer in pred:
            start = qa_answer.offset_answer_start
            end = qa_answer.offset_answer_end
            if start == 0:
                start = -1
            else:
                start += passage_start_t
                if start < 0:
                    logger.error("Start token index < 0 (document level)")
            if end == 0:
                end = -1
            else:
                end += passage_start_t
                if end < 0:
                    logger.error("End token index < 0 (document level)")
            qa_answer.to_doc_level(start, end)
            new_pred.append(qa_answer)
        return new_pred

    @staticmethod
    def label_to_doc_idxs(label, passage_start_t):
        """
        Converts the passage level labels to document level labels. Note that on the doc level we
        don't have special tokens or question tokens. This means that a no answer
        cannot be prepresented by a (0,0) span but will instead be represented by (-1, -1)
        """
        new_label = []
        for start, end in label:
            # If there is a valid label
            if start > 0 or end > 0:
                new_label.append((start + passage_start_t, end + passage_start_t))
            # If the label is a no answer, we represent this as a (-1, -1) span
            # since there is no CLS token on the document level
            if start == 0 and end == 0:
                new_label.append((-1, -1))
        return new_label

    def prepare_labels(self, labels, start_of_word, **kwargs):
        return labels


class TextSimilarityHead(PredictionHead):
    """
    Trains a head on predicting the similarity of two texts like in Dense Passage Retrieval.
    """

    def __init__(self, similarity_function: str = "dot_product", global_loss_buffer_size: int = 150000, **kwargs):
        """
        Init the TextSimilarityHead.

        :param similarity_function: Function to calculate similarity between queries and passage embeddings.
                                    Choose either "dot_product" (Default) or "cosine".
        :param global_loss_buffer_size: Buffer size for all_gather() in DDP.
                                        Increase if errors like "encoded data exceeds max_size ..." come up
        :param kwargs:
        """
        super(TextSimilarityHead, self).__init__()

        self.similarity_function = similarity_function
        self.loss_fct = NLLLoss(reduction="mean")
        self.task_name = "text_similarity"
        self.model_type = "text_similarity"
        self.ph_output_type = "per_sequence"
        self.global_loss_buffer_size = global_loss_buffer_size
        self.generate_config()

    @classmethod
    def dot_product_scores(cls, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates dot product similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                        of dimension n1 x D,
                        where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                        of dimension n2 x D,
                        where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                        and D is embedding size

        :return: dot_product: similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        dot_product = torch.matmul(query_vectors, torch.transpose(passage_vectors, 0, 1))
        return dot_product

    @classmethod
    def cosine_scores(cls, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates cosine similarity scores for two 2-dimensional tensors

        :param query_vectors: tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is (batch_size * num_positives) + (batch_size * num_hard_negatives)
                          and D is embedding size

        :return: cosine similarity score of each query with each context/passage (dimension: n1xn2)
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        cosine_similarities = []
        passages_per_batch = passage_vectors.shape[0]
        for query_vector in query_vectors:
            query_vector_repeated = query_vector.repeat(passages_per_batch, 1)
            current_cosine_similarities = nn.functional.cosine_similarity(query_vector_repeated, passage_vectors, dim=1)
            cosine_similarities.append(current_cosine_similarities)
        return torch.stack(cosine_similarities)

    def get_similarity_function(self):
        """
        Returns the type of similarity function used to compare queries and passages/contexts
        """
        if "dot_product" in self.similarity_function:
            return TextSimilarityHead.dot_product_scores
        elif "cosine" in self.similarity_function:
            return TextSimilarityHead.cosine_scores
        else:
            raise AttributeError(
                f"The similarity function can only be 'dot_product' or 'cosine', not '{self.similarity_function}'"
            )

    def forward(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Only packs the embeddings from both language models into a tuple. No further modification.
        The similarity calculation is handled later to enable distributed training (DDP)
        while keeping the support for in-batch negatives.
        (Gather all embeddings from nodes => then do similarity scores + loss)

        :param query_vectors: Tensor of query embeddings from BiAdaptive model
                          of dimension n1 x D,
                          where n1 is the number of queries/batch size and D is embedding size
        :param passage_vectors: Tensor of context/passage embeddings from BiAdaptive model
                          of dimension n2 x D,
                          where n2 is the number of queries/batch size and D is embedding size
        """
        return query_vectors, passage_vectors

    def _embeddings_to_scores(self, query_vectors: torch.Tensor, passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates similarity scores between all given query_vectors and passage_vectors

        :param query_vectors: Tensor of queries encoded by the query encoder model
        :param passage_vectors: Tensor of passages encoded by the passage encoder model
        :return: Tensor of log softmax similarity scores of each query with each passage (dimension: n1xn2)
        """
        sim_func = self.get_similarity_function()
        scores = sim_func(query_vectors, passage_vectors)

        if len(query_vectors.size()) > 1:
            q_num = query_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = nn.functional.log_softmax(scores, dim=1)
        return softmax_scores

    def logits_to_loss(self, logits: Tuple[torch.Tensor, torch.Tensor], label_ids, **kwargs):  # type: ignore
        """
        Computes the loss (Default: NLLLoss) by applying a similarity function (Default: dot product) to the input
        tuple of (query_vectors, passage_vectors) and afterwards applying the loss function on similarity scores.

        :param logits: Tuple of Tensors (query_embedding, passage_embedding) as returned from forward()

        :return: negative log likelihood loss from similarity scores
        """
        # Check if DDP is initialized
        try:
            if torch.distributed.is_available():
                rank = torch.distributed.get_rank()
            else:
                rank = -1
        except (AssertionError, RuntimeError):
            rank = -1

        # Prepare predicted scores
        query_vectors, passage_vectors = logits

        # Prepare Labels
        positive_idx_per_question = torch.nonzero((label_ids.view(-1) == 1), as_tuple=False)

        # Gather global embeddings from all distributed nodes (DDP)
        if rank != -1:
            q_vector_to_send = torch.empty_like(query_vectors).cpu().copy_(query_vectors).detach_()
            p_vector_to_send = torch.empty_like(passage_vectors).cpu().copy_(passage_vectors).detach_()

            global_question_passage_vectors = all_gather_list(
                [q_vector_to_send, p_vector_to_send, positive_idx_per_question], max_size=self.global_loss_buffer_size
            )

            global_query_vectors = []
            global_passage_vectors = []
            global_positive_idx_per_question = []
            total_passages = 0
            for i, item in enumerate(global_question_passage_vectors):
                q_vector, p_vectors, positive_idx = item

                if i != rank:
                    global_query_vectors.append(q_vector.to(query_vectors.device))
                    global_passage_vectors.append(p_vectors.to(passage_vectors.device))
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx])
                else:
                    global_query_vectors.append(query_vectors)
                    global_passage_vectors.append(passage_vectors)
                    global_positive_idx_per_question.extend([v + total_passages for v in positive_idx_per_question])
                total_passages += p_vectors.size(0)

            global_query_vectors = torch.cat(global_query_vectors, dim=0)  # type: ignore
            global_passage_vectors = torch.cat(global_passage_vectors, dim=0)  # type: ignore
            global_positive_idx_per_question = torch.LongTensor(global_positive_idx_per_question)  # type: ignore
        else:
            global_query_vectors = query_vectors  # type: ignore
            global_passage_vectors = passage_vectors  # type: ignore
            global_positive_idx_per_question = positive_idx_per_question  # type: ignore

        # Get similarity scores
        softmax_scores = self._embeddings_to_scores(global_query_vectors, global_passage_vectors)  # type: ignore
        targets = global_positive_idx_per_question.squeeze(-1).to(softmax_scores.device)  # type: ignore

        # Calculate loss
        loss = self.loss_fct(softmax_scores, targets)
        return loss

    def logits_to_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:  # type: ignore
        """
        Returns predicted ranks(similarity) of passages/context for each query

        :param logits: tensor of log softmax similarity scores of each query with each context/passage (dimension: n1xn2)

        :return: predicted ranks of passages for each query
        """
        query_vectors, passage_vectors = logits
        softmax_scores = self._embeddings_to_scores(query_vectors, passage_vectors)
        _, sorted_scores = torch.sort(softmax_scores, dim=1, descending=True)
        return sorted_scores

    def prepare_labels(self, label_ids, **kwargs) -> torch.Tensor:  # type: ignore
        """
        Returns a tensor with passage labels(0:hard_negative/1:positive) for each query

        :return: passage labels(0:hard_negative/1:positive) for each query
        """
        labels = torch.zeros(label_ids.size(0), label_ids.numel())

        positive_indices = torch.nonzero(label_ids.view(-1) == 1, as_tuple=False)

        for i, indx in enumerate(positive_indices):
            labels[i, indx.item()] = 1
        return labels

    def formatted_preds(self, logits: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        raise NotImplementedError("formatted_preds is not supported in TextSimilarityHead yet!")


def _is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False
