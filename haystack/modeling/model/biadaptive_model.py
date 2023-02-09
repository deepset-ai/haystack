from typing import List, Optional, Callable, Union, Dict

import os
import logging
from pathlib import Path

import torch
from torch import nn
from transformers import DPRContextEncoder, DPRQuestionEncoder, AutoModel

from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.model.language_model import get_language_model, LanguageModel
from haystack.modeling.model.prediction_head import PredictionHead, TextSimilarityHead
from haystack.utils.experiment_tracking import Tracker as tracker


logger = logging.getLogger(__name__)


def loss_per_head_sum(
    loss_per_head: List[torch.Tensor], global_step: Optional[int] = None, batch: Optional[Dict] = None
):
    """
    Sums up the loss of each prediction head.

    :param loss_per_head: List of losses.
    """
    return sum(loss_per_head)


class BiAdaptiveModel(nn.Module):
    """
    PyTorch implementation containing all the modelling needed for your NLP task.
    Combines 2 language models for representation of 2 sequences and a prediction head.
    Allows for gradient flow back to the 2 language model components.
    """

    def __init__(
        self,
        language_model1: LanguageModel,
        language_model2: LanguageModel,
        prediction_heads: List[PredictionHead],
        embeds_dropout_prob: float = 0.1,
        device: torch.device = torch.device("cuda"),
        lm1_output_types: Optional[Union[str, List[str]]] = None,
        lm2_output_types: Optional[Union[str, List[str]]] = None,
        loss_aggregation_fn: Optional[Callable] = None,
    ):
        """
        :param language_model1: Any model that turns token ids into vector representations.
        :param language_model2: Any model that turns token ids into vector representations.
        :param prediction_heads: A list of models that take 2 sequence embeddings and return logits for a given task.
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by any of the 2
                                    language models will be zeroed.
        :param lm1_output_types: How to extract the embeddings from the final layer of the first language model. When set
                                 to "per_token", one embedding will be extracted per input token. If set to
                                 "per_sequence" (default), a single embedding will be extracted to represent the full
                                 input sequence. Can either be a single string, or a list of strings,
                                 one for each prediction head.
        :param lm2_output_types: How to extract the embeddings from the final layer of the second language model. When set
                                 to "per_token", one embedding will be extracted per input token. If set to
                                 "per_sequence" (default), a single embedding will be extracted to represent the full
                                 input sequence. Can either be a single string, or a list of strings,
                                 one for each prediction head.
        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
                                    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
                                    Output: aggregated loss (tensor)
                                    Default is a simple sum:
                                    `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
                                    However, you can pass more complex functions that depend on the
                                    current step (e.g. for round-robin style multitask learning) or the actual
                                    content of the batch (e.g. certain labels)
                                    Note: The loss at this stage is per sample, i.e one tensor of
                                    shape (batchsize) per prediction head.
        """
        if lm1_output_types is None:
            lm1_output_types = ["per_sequence"]
        if lm2_output_types is None:
            lm2_output_types = ["per_sequence"]
        super(BiAdaptiveModel, self).__init__()

        self.device = device
        self.language_model1 = language_model1.to(device)
        self.lm1_output_dims = language_model1.output_dims
        self.language_model2 = language_model2.to(device)
        self.lm2_output_dims = language_model2.output_dims
        self.dropout1 = nn.Dropout(embeds_dropout_prob)
        self.dropout2 = nn.Dropout(embeds_dropout_prob)
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.lm1_output_types = [lm1_output_types] if isinstance(lm1_output_types, str) else lm1_output_types
        self.lm2_output_types = [lm2_output_types] if isinstance(lm2_output_types, str) else lm2_output_types
        self.log_params()
        # default loss aggregation function is a simple sum (without using any of the optional params)
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def save(self, save_dir: Union[str, Path], lm1_name: str = "lm1", lm2_name: str = "lm2"):
        """
        Saves the 2 language model weights and respective config_files in directories lm1 and lm2 within save_dir.

        :param save_dir: Path | str to save the BiAdaptiveModel to.
        """
        os.makedirs(save_dir, exist_ok=True)
        for name, model in zip([lm1_name, lm2_name], [self.language_model1, self.language_model2]):
            model_save_dir = Path.joinpath(Path(save_dir), Path(name))
            os.makedirs(model_save_dir, exist_ok=True)
            model.save(model_save_dir)

        for i, ph in enumerate(self.prediction_heads):
            logger.info("prediction_head saving")
            ph.save(save_dir, i)

    @classmethod
    def load(
        cls,
        load_dir: Path,
        device: torch.device,
        strict: bool = False,
        lm1_name: str = "lm1",
        lm2_name: str = "lm2",
        processor: Optional[Processor] = None,
    ):
        """
        Loads a BiAdaptiveModel from a directory. The directory must contain:

        * directory "lm1_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * directory "lm2_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Token
        * special_tokens_map.json

        :param load_dir: Location where adaptive model is stored.
        :param device: To which device we want to sent the model, either torch.device("cpu") or torch.device("cuda").
        :param lm1_name: The name to assign to the first loaded language model (for encoding queries).
        :param lm2_name: The name to assign to the second loaded language model (for encoding context/passages).
        :param strict: Whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
        :param processor: Processor to populate prediction head with information coming from tasks.
        """
        # Language Model
        if lm1_name:
            language_model1 = get_language_model(os.path.join(load_dir, lm1_name))
        else:
            language_model1 = get_language_model(load_dir)
        if lm2_name:
            language_model2 = get_language_model(os.path.join(load_dir, lm2_name))
        else:
            language_model2 = get_language_model(load_dir)

        # Prediction heads
        ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict, load_weights=False)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        model = cls(language_model1, language_model2, prediction_heads, 0.1, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)  # type: ignore

        return model

    def logits_to_loss_per_head(self, logits: torch.Tensor, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: The per sample per prediciton head loss whose first two dimensions have length n_pred_heads, batch_size.
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            # check if PredictionHead connected to Processor
            assert hasattr(head, "label_tensor_name"), (
                f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to the Adaptive Model?"
            )
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head, **kwargs))
        return all_losses

    def logits_to_loss(self, logits: torch.Tensor, global_step: Optional[int] = None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: Logits, can vary in shape and type, depending on task.
        :param global_step: Number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: torch.Tensor that is the per sample loss (len: batch_size).
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def logits_to_preds(self, logits: torch.Tensor, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: A list of all predictions from all prediction heads.
        """
        all_preds = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits: torch.Tensor, **kwargs):
        """
        Format predictions to strings for inference output

        :param logits: Model logits.
        :param kwargs: Placeholder for passing generic parameters
        :return: Predictions in the right format.
        """
        n_heads = len(self.prediction_heads)

        if n_heads == 1:
            preds_final = []
            # This try catch is to deal with the fact that sometimes we collect preds before passing it to
            # formatted_preds (see Inferencer._get_predictions_and_aggregate()) and sometimes we don't
            # (see Inferencer._get_predictions())
            try:
                preds = kwargs["preds"]
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs["preds"] = preds_flat
            except KeyError:
                kwargs["preds"] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            # TODO This is very messy - we need better definition of what the output should look like
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and "predictions" in preds:
                preds_final.append(preds)
        return preds_final

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :return: Labels in the right format.
        """
        all_labels = []
        # for head, label_map_one_head in zip(self.prediction_heads):
        #     labels = head.prepare_labels(label_map=label_map_one_head, **kwargs)
        #     all_labels.append(labels)
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_segment_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        passage_input_ids: Optional[torch.Tensor] = None,
        passage_segment_ids: Optional[torch.Tensor] = None,
        passage_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Push data through the whole model and returns logits. The data will propagate through
        the first language model and second language model based on the tensor names and both the
        encodings through each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to both the language models and prediction head(s).
        :return: All logits as torch.tensor or multiple tensors.
        """

        # Run forward pass of both language models
        pooled_output = self.forward_lm(
            query_input_ids=query_input_ids,
            query_segment_ids=query_segment_ids,
            query_attention_mask=query_attention_mask,
            passage_input_ids=passage_input_ids,
            passage_segment_ids=passage_segment_ids,
            passage_attention_mask=passage_attention_mask,
        )

        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm1_out, lm2_out in zip(self.prediction_heads, self.lm1_output_types, self.lm2_output_types):
                # Choose relevant vectors from LM as output and perform dropout
                if pooled_output[0] is not None:
                    if lm1_out == "per_sequence" or lm1_out == "per_sequence_continuous":
                        output1 = self.dropout1(pooled_output[0])
                    else:
                        raise ValueError(
                            "Unknown extraction strategy from BiAdaptive language_model1: {}".format(lm1_out)
                        )
                else:
                    output1 = None

                if pooled_output[1] is not None:
                    if lm2_out == "per_sequence" or lm2_out == "per_sequence_continuous":
                        output2 = self.dropout2(pooled_output[1])
                    else:
                        raise ValueError(
                            "Unknown extraction strategy from BiAdaptive language_model2: {}".format(lm2_out)
                        )
                else:
                    output2 = None

                embedding1, embedding2 = head(output1, output2)
                all_logits.append(tuple([embedding1, embedding2]))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append((pooled_output))

        return all_logits

    def forward_lm(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_segment_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        passage_input_ids: Optional[torch.Tensor] = None,
        passage_segment_ids: Optional[torch.Tensor] = None,
        passage_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the BiAdaptive model.

        :param kwargs: Holds all arguments that need to be passed to the language models.
        :return: 2 tensors of pooled_output from the 2 language models.
        """
        pooled_output = [None, None]

        if query_input_ids is not None and query_segment_ids is not None and query_attention_mask is not None:
            pooled_output1, _ = self.language_model1(
                input_ids=query_input_ids, segment_ids=query_segment_ids, attention_mask=query_attention_mask
            )
            pooled_output[0] = pooled_output1

        if passage_input_ids is not None and passage_segment_ids is not None and passage_attention_mask is not None:
            max_seq_len = passage_input_ids.shape[-1]
            passage_input_ids = passage_input_ids.view(-1, max_seq_len)
            passage_attention_mask = passage_attention_mask.view(-1, max_seq_len)
            passage_segment_ids = passage_segment_ids.view(-1, max_seq_len)

            pooled_output2, _ = self.language_model2(
                input_ids=passage_input_ids, segment_ids=passage_segment_ids, attention_mask=passage_attention_mask
            )
            pooled_output[1] = pooled_output2

        return tuple(pooled_output)

    def log_params(self):
        """
        Logs paramteres to generic logger MlLogger
        """
        params = {
            "lm1_type": self.language_model1.__class__.__name__,
            "lm1_name": self.language_model1.name,
            "lm1_output_types": ",".join(self.lm1_output_types),
            "lm2_type": self.language_model2.__class__.__name__,
            "lm2_name": self.language_model2.name,
            "lm2_output_types": ",".join(self.lm2_output_types),
            "prediction_heads": ",".join([head.__class__.__name__ for head in self.prediction_heads]),
        }
        try:
            tracker.track_params(params)
        except Exception as e:
            logger.warning("ML logging didn't work: %s", e)

    def verify_vocab_size(self, vocab_size1: int, vocab_size2: int):
        """
        Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()
        """

        model1_vocab_len = self.language_model1.model.resize_token_embeddings(new_num_tokens=None).num_embeddings  # type: ignore [union-attr,operator]

        msg = (
            f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model1_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )
        assert vocab_size1 == model1_vocab_len, msg

        model2_vocab_len = self.language_model2.model.resize_token_embeddings(new_num_tokens=None).num_embeddings  # type: ignore [union-attr,operator]

        msg = (
            f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model2_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )
        assert vocab_size2 == model2_vocab_len, msg

    def connect_heads_with_processor(self, tasks: Dict, require_labels: bool = True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """

        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            head.metric = tasks[head.task_name]["metric"]

    def get_language(self):
        return self.language_model1.language, self.language_model2.language

    @classmethod
    def _get_prediction_head_files(cls, load_dir: Union[str, Path]):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        config_files = [load_dir / f for f in files if "config.json" in f and "prediction_head" in f]
        # sort them to get correct order in case of multiple prediction heads
        config_files.sort()
        return config_files

    def convert_to_transformers(self):
        if len(self.prediction_heads) != 1:
            raise ValueError(
                f"Currently conversion only works for models with a SINGLE prediction head. "
                f"Your model has {len(self.prediction_heads)}"
            )

        if self.prediction_heads[0].model_type == "text_similarity":
            # init model
            if "dpr" in self.language_model1.model.config.model_type:
                transformers_model1 = DPRQuestionEncoder(config=self.language_model1.model.config)
            else:
                transformers_model1 = AutoModel.from_config(config=self.language_model1.model.config)
            if "dpr" in self.language_model2.model.config.model_type:
                transformers_model2 = DPRContextEncoder(config=self.language_model2.model.config)
            else:
                transformers_model2 = AutoModel.from_config(config=self.language_model2.model.config)

            # transfer weights for language model + prediction head
            setattr(transformers_model1, transformers_model1.base_model_prefix, self.language_model1.model)
            setattr(transformers_model2, transformers_model2.base_model_prefix, self.language_model2.model)
            logger.warning("No prediction head weights are required for DPR")

        else:
            raise NotImplementedError(
                f"Haystack -> Transformers conversion is not supported yet for"
                f" prediction heads of type {self.prediction_heads[0].model_type}"
            )
        pass

        return transformers_model1, transformers_model2

    @classmethod
    def convert_from_transformers(
        cls,
        model_name_or_path1: Union[str, Path],
        model_name_or_path2: Union[str, Path],
        device: torch.device,
        task_type: str = "text_similarity",
        processor: Optional[Processor] = None,
        similarity_function: str = "dot_product",
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in Haystack (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path1: local path of a saved model or name of a public one for Question Encoder
                                              Exemplary public names:
                                              - facebook/dpr-question_encoder-single-nq-base
                                              - deepset/bert-large-uncased-whole-word-masking-squad2
        :param model_name_or_path2: local path of a saved model or name of a public one for Context/Passage Encoder
                                      Exemplary public names:
                                      - facebook/dpr-ctx_encoder-single-nq-base
                                      - deepset/bert-large-uncased-whole-word-masking-squad2
        :param device: On which hardware the conversion is going to run on. Either torch.device("cpu") or torch.device("cuda")
        :param task_type: 'text_similarity' More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :return: AdaptiveModel
        """
        lm1 = get_language_model(pretrained_model_name_or_path=model_name_or_path1, use_auth_token=use_auth_token)
        lm2 = get_language_model(pretrained_model_name_or_path=model_name_or_path2, use_auth_token=use_auth_token)
        prediction_head = TextSimilarityHead(similarity_function=similarity_function)
        # TODO Infer type of head automatically from config
        if task_type == "text_similarity":
            bi_adaptive_model = cls(
                language_model1=lm1,
                language_model2=lm2,
                prediction_heads=[prediction_head],
                embeds_dropout_prob=0.1,
                lm1_output_types=["per_sequence"],
                lm2_output_types=["per_sequence"],
                device=device,
            )
        else:
            raise NotImplementedError(
                f"Huggingface's transformer models of type {task_type} are not supported yet for BiAdaptive Models"
            )

        if processor:
            bi_adaptive_model.connect_heads_with_processor(processor.tasks)  # type: ignore

        return bi_adaptive_model
