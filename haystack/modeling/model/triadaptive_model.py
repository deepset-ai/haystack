import logging
import os
from pathlib import Path
from typing import List, Optional, Callable, Union, Dict

import torch
from torch import nn

from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.model.language_model import get_language_model, LanguageModel
from haystack.modeling.model.prediction_head import PredictionHead
from haystack.utils.experiment_tracking import Tracker as tracker

logger = logging.getLogger(__name__)


def loss_per_head_sum(
    loss_per_head: List[torch.Tensor], global_step: Optional[int] = None, batch: Optional[Dict] = None
):
    """
    Input: loss_per_head (list of tensors), global_step (int), batch (dict)
    Output: aggregated loss (tensor)
    """
    return sum(loss_per_head)


class TriAdaptiveModel(nn.Module):
    """PyTorch implementation containing all the modelling needed for
    your NLP task. Combines 3 language models for representation of 3
    sequences and a prediction head. Allows for gradient flow back to
    the 3 language model components.

    The TriAdaptiveModel is currently highly tailored to the use case of joint
    retrieval of text passages and tables using one encoder as question encoder
    (language_model1), one encoder as text passage encoder (language_model2), and
    one encoder as table encoder (language_model3).
    """

    def __init__(
        self,
        language_model1: LanguageModel,
        language_model2: LanguageModel,
        language_model3: LanguageModel,
        prediction_heads: List[PredictionHead],
        embeds_dropout_prob: float = 0.1,
        device: torch.device = torch.device("cuda"),
        lm1_output_types: Union[str, List[str]] = ["per_sequence"],
        lm2_output_types: Union[str, List[str]] = ["per_sequence"],
        lm3_output_types: Union[str, List[str]] = ["per_sequence"],
        loss_aggregation_fn: Optional[Callable] = None,
    ):
        """
        :param language_model1: Any model that turns token ids into vector representations.
        :param language_model2: Any model that turns token ids into vector representations.
        :param language_model3: Any model that turns token ids into vector representations.
        :param prediction_heads: A list of models that take 3 sequence embeddings and return logits for a given task.
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by any of the three
           language model will be zeroed.
        :param lm1_output_types: How to extract the embeddings from the final layer of the first language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :param lm2_output_types: How to extract the embeddings from the final layer of the second language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
                                input sequence. Can either be a single string, or a list of strings,
                                one for each prediction head.
        :param lm3_output_types: How to extract the embeddings from the final layer of the third language model. When set
                                to "per_token", one embedding will be extracted per input token. If set to
                                "per_sequence", a single embedding will be extracted to represent the full
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

        super(TriAdaptiveModel, self).__init__()
        self.device = device
        self.language_model1 = language_model1.to(device)
        self.lm1_output_dims = language_model1.output_dims
        self.language_model2 = language_model2.to(device)
        self.lm2_output_dims = language_model2.output_dims
        self.language_model3 = language_model3.to(device)
        self.lm3_output_dims = language_model3.output_dims
        self.dropout1 = nn.Dropout(embeds_dropout_prob)
        self.dropout2 = nn.Dropout(embeds_dropout_prob)
        self.dropout3 = nn.Dropout(embeds_dropout_prob)
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.lm1_output_types = [lm1_output_types] if isinstance(lm1_output_types, str) else lm1_output_types
        self.lm2_output_types = [lm2_output_types] if isinstance(lm2_output_types, str) else lm2_output_types
        self.lm3_output_types = [lm3_output_types] if isinstance(lm3_output_types, str) else lm3_output_types
        self.log_params()
        # default loss aggregation function is a simple sum (without using any of the optional params)
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def save(self, save_dir: Path, lm1_name: str = "lm1", lm2_name: str = "lm2", lm3_name: str = "lm3"):
        """
        Saves the 3 language model weights and respective config_files in directories lm1 and lm2 within save_dir.

        :param save_dir: Path to save the TriAdaptiveModel to.
        """
        os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(Path.joinpath(save_dir, Path(lm1_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm1_name)))
        if not os.path.exists(Path.joinpath(save_dir, Path(lm2_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm2_name)))
        if not os.path.exists(Path.joinpath(save_dir, Path(lm3_name))):
            os.makedirs(Path.joinpath(save_dir, Path(lm3_name)))
        self.language_model1.save(Path.joinpath(save_dir, Path(lm1_name)))
        self.language_model2.save(Path.joinpath(save_dir, Path(lm2_name)))
        self.language_model3.save(Path.joinpath(save_dir, Path(lm3_name)))
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
        lm3_name: str = "lm3",
        processor: Optional[Processor] = None,
    ):
        """
        Loads a TriAdaptiveModel from a directory. The directory must contain:

        * directory "lm1_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * directory "lm2_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * directory "lm3_name" with following files:
            -> language_model.bin
            -> language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Token
        * special_tokens_map.json

        :param load_dir: Location where the TriAdaptiveModel is stored.
        :param device: To which device we want to sent the model, either torch.device("cpu") or torch.device("cuda").
        :param lm1_name: The name to assign to the first loaded language model (for encoding queries).
        :param lm2_name: The name to assign to the second loaded language model (for encoding context/passages).
        :param lm3_name: The name to assign to the second loaded language model (for encoding tables).
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
        if lm3_name:
            language_model3 = get_language_model(os.path.join(load_dir, lm3_name))
        else:
            language_model3 = get_language_model(load_dir)

        # Prediction heads
        ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict, load_weights=False)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        model = cls(language_model1, language_model2, language_model3, prediction_heads, 0.1, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)

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

        :param logits: logits, can vary in shape and type, depending on task.
        :param global_step: number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: loss: torch.Tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :return: Labels in the right format.
        """
        all_labels = []
        for head in self.prediction_heads:
            labels = head.prepare_labels(**kwargs)
            all_labels.append(labels)
        return all_labels

    def forward(self, **kwargs):
        """
        Push data through the whole model and returns logits. The data will propagate through
        the first language model, second and third language model based on the tensor names and both the
        encodings through each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to both the language models and prediction head(s).
        :return: All logits as torch.Tensor or multiple tensors.
        """

        # Run forward pass of the three language models
        pooled_output = self.forward_lm(**kwargs)

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
                            "Unknown extraction strategy from TriAdaptive language_model1: {}".format(lm1_out)
                        )
                else:
                    output1 = None

                if pooled_output[1] is not None:
                    if lm2_out == "per_sequence" or lm2_out == "per_sequence_continuous":
                        output2 = self.dropout2(pooled_output[1])
                    else:
                        raise ValueError(
                            "Unknown extraction strategy from TriAdaptive language_model2: {}".format(lm2_out)
                        )
                else:
                    output2 = None

                embedding1, embedding2 = head(output1, output2)
                all_logits.append(tuple([embedding1, embedding2]))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append((pooled_output))

        return all_logits

    def forward_lm(self, **kwargs):
        """
        Forward pass for the TriAdaptive model.

        :param kwargs: Holds all arguments that need to be passed to the language models.
        :return: Two tensors of pooled_output from the three language models
        """
        pooled_output = [None, None]
        # Forward pass for the queries
        if "query_input_ids" in kwargs.keys():
            pooled_output1, _ = self.language_model1(
                input_ids=kwargs.get("query_input_ids"),
                segment_ids=kwargs.get("query_segment_ids"),
                attention_mask=kwargs.get("query_attention_mask"),
                output_hidden_states=False,
                output_attentions=False,
            )
            pooled_output[0] = pooled_output1

        # Forward pass for text passages and tables
        if "passage_input_ids" in kwargs.keys():
            table_mask = torch.flatten(kwargs["is_table"]) == True

            # Current batch consists of only tables
            if all(table_mask):
                pooled_output2, _ = self.language_model3(
                    passage_input_ids=kwargs["passage_input_ids"],
                    passage_segment_ids=kwargs["table_segment_ids"],
                    passage_attention_mask=kwargs["passage_attention_mask"],
                    output_hidden_states=False,
                    output_attentions=False,
                )
                pooled_output[1] = pooled_output2

            # Current batch consists of tables and texts
            elif any(table_mask):

                # Make input two-dimensional
                max_seq_len = kwargs["passage_input_ids"].shape[-1]
                passage_input_ids = kwargs["passage_input_ids"].view(-1, max_seq_len)
                passage_attention_mask = kwargs["passage_attention_mask"].view(-1, max_seq_len)
                passage_segment_ids = kwargs["passage_segment_ids"].view(-1, max_seq_len)

                table_segment_ids = kwargs["table_segment_ids"].view(-1, max_seq_len)
                table_input_ids = passage_input_ids[table_mask]
                table_segment_ids = table_segment_ids[table_mask]
                table_attention_mask = passage_attention_mask[table_mask]

                pooled_output_tables, _ = self.language_model3(
                    input_ids=table_input_ids,
                    segment_ids=table_segment_ids,
                    attention_mask=table_attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                )

                text_input_ids = passage_input_ids[~table_mask]
                text_segment_ids = passage_segment_ids[~table_mask]
                text_attention_mask = passage_attention_mask[~table_mask]

                pooled_output_text, _ = self.language_model2(
                    input_ids=text_input_ids,
                    segment_ids=text_segment_ids,
                    attention_mask=text_attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                )

                last_table_idx = 0
                last_text_idx = 0
                combined_outputs = []
                for mask in table_mask:
                    if mask:
                        combined_outputs.append(pooled_output_tables[last_table_idx])
                        last_table_idx += 1
                    else:
                        combined_outputs.append(pooled_output_text[last_text_idx])
                        last_text_idx += 1
                combined_outputs = torch.stack(combined_outputs)

                embedding_size = pooled_output_tables.shape[-1]
                assert (
                    pooled_output_tables.shape[-1] == pooled_output_text.shape[-1]
                ), "Passage embedding model and table embedding model use different embedding sizes"
                pooled_output_combined = combined_outputs.view(-1, embedding_size)
                pooled_output[1] = pooled_output_combined

            # Current batch consists of only texts
            else:
                # Make input two-dimensional
                max_seq_len = kwargs["passage_input_ids"].shape[-1]
                input_ids = kwargs["passage_input_ids"].view(-1, max_seq_len)
                attention_mask = kwargs["passage_attention_mask"].view(-1, max_seq_len)
                segment_ids = kwargs["passage_segment_ids"].view(-1, max_seq_len)

                pooled_output2, _ = self.language_model2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    output_hidden_states=False,
                    output_attentions=False,
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
            "lm3_type": self.language_model3.__class__.__name__,
            "lm3_name": self.language_model3.name,
            "lm3_output_types": ",".join(self.lm3_output_types),
            "prediction_heads": ",".join([head.__class__.__name__ for head in self.prediction_heads]),
        }
        try:
            tracker.track_params(params)
        except Exception as e:
            logger.warning("ML logging didn't work: %s", e)

    def verify_vocab_size(self, vocab_size1: int, vocab_size2: int, vocab_size3: int):
        """Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()"""

        model1_vocab_len = self.language_model1.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = (
            f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model1_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )
        assert vocab_size1 == model1_vocab_len, msg

        model2_vocab_len = self.language_model2.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = (
            f"Vocab size of tokenizer {vocab_size1} doesn't match with model {model2_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )
        assert vocab_size2 == model2_vocab_len, msg

        model3_vocab_len = self.language_model3.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = (
            f"Vocab size of tokenizer {vocab_size3} doesn't match with model {model3_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )

        assert vocab_size3 == model1_vocab_len, msg

    def get_language(self):
        return self.language_model1.language, self.language_model2.language, self.language_model3.language

    def connect_heads_with_processor(self, tasks: Dict, require_labels: bool = True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task
                     (e.g. label_list, metric, tensor name).
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels.
        """

        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            num_labels = len(label_list)
            head.metric = tasks[head.task_name]["metric"]

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
