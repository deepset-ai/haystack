import copy
import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Iterable, Dict, Union, List, Optional, Callable

import numpy
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForQuestionAnswering
from transformers.convert_graph_to_onnx import convert, quantize as quantize_model

from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.model.language_model import (
    get_language_model,
    LanguageModel,
    _get_model_type,
    capitalize_model_type,
)
from haystack.modeling.model.prediction_head import PredictionHead, QuestionAnsweringHead
from haystack.utils.experiment_tracking import Tracker as tracker


logger = logging.getLogger(__name__)


class BaseAdaptiveModel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """

    language_model: LanguageModel
    subclasses = {}  # type: Dict

    def __init_subclass__(cls, **kwargs):
        """
        This automatically keeps track of all available subclasses.
        Enables generic load() for all specific AdaptiveModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads: Union[List[PredictionHead], nn.ModuleList]):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: Arguments to pass for loading the model.
        :return: Instance of a model.
        """
        if (Path(kwargs["load_dir"]) / "model.onnx").is_file():
            model = cls.subclasses["ONNXAdaptiveModel"].load(**kwargs)
        else:
            model = cls.subclasses["AdaptiveModel"].load(**kwargs)
        return model

    def logits_to_preds(self, logits: torch.Tensor, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: Logits that can vary in shape and type, depending on task.
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
        Format predictions for inference.

        :param logits: Model logits.
        :return: Predictions in the right format.
        """
        n_heads = len(self.prediction_heads)

        if n_heads == 0:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)

        elif n_heads == 1:
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

    def connect_heads_with_processor(self, tasks: Dict, require_labels: bool = True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and
                      the values are the details of the task (e.g. label_list, metric,
                      tensor name).
        :param require_labels: If True, an error will be thrown when a task is
                               not supplied with labels.
        :return: None
        """
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]["label_tensor_name"]
            label_list = tasks[head.task_name]["label_list"]
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            head.metric = tasks[head.task_name]["metric"]

    @classmethod
    def _get_prediction_head_files(cls, load_dir: Union[str, Path], strict: bool = True):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [load_dir / f for f in files if ".bin" in f and "prediction_head" in f]
        config_files = [load_dir / f for f in files if "config.json" in f and "prediction_head" in f]
        # sort them to get correct order in case of multiple prediction heads
        model_files.sort()
        config_files.sort()

        if strict:
            error_str = (
                f"There is a mismatch in number of model files ({len(model_files)}) and config files ({len(config_files)})."
                "This might be because the Language Model Prediction Head "
                "does not currently support saving and loading"
            )
            assert len(model_files) == len(config_files), error_str
        logger.info("Found files for loading %s prediction heads", len(model_files))

        return model_files, config_files


def loss_per_head_sum(loss_per_head: Iterable, global_step: Optional[int] = None, batch: Optional[Dict] = None):
    """
    Sums up the loss of each prediction head.

    :param loss_per_head: List of losses.
    """
    return sum(loss_per_head)


class AdaptiveModel(nn.Module, BaseAdaptiveModel):
    """
    PyTorch implementation containing all the modelling needed for your NLP task. Combines a language
    model and a prediction head. Allows for gradient flow back to the language model component.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        prediction_heads: List[PredictionHead],
        embeds_dropout_prob: float,
        lm_output_types: Union[str, List[str]],
        device: torch.device,
        loss_aggregation_fn: Optional[Callable] = None,
    ):
        """
        :param language_model: Any model that turns token ids into vector representations.
        :param prediction_heads: A list of models that take embeddings and return logits for a given task.
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by the
                                    language model will be zeroed.
        :param lm_output_types: How to extract the embeddings from the final layer of the language model. When set
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
        super(AdaptiveModel, self).__init__()  # type: ignore
        self.device = device
        self.language_model = language_model.to(device)
        self.lm_output_dims = language_model.output_dims
        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])
        self.fit_heads_to_lm()
        self.dropout = nn.Dropout(embeds_dropout_prob)
        self.lm_output_types = [lm_output_types] if isinstance(lm_output_types, str) else lm_output_types
        self.log_params()
        # default loss aggregation function is a simple sum (without using any of the optional params)
        if not loss_aggregation_fn:
            loss_aggregation_fn = loss_per_head_sum
        self.loss_aggregation_fn = loss_aggregation_fn

    def fit_heads_to_lm(self):
        """
        This iterates over each prediction head and ensures that its input
        dimensionality matches the output dimensionality of the language model.
        If it doesn't, it is resized so it does fit.
        """
        for ph in self.prediction_heads:
            ph.resize_input(self.lm_output_dims)
            ph.to(self.device)

    def bypass_ph(self):
        """
        Replaces methods in the prediction heads with dummy functions.
        Used for benchmarking where we want to isolate the LanguageModel run time
        from the PredictionHead run time.
        """
        # TODO convert inner functions into lambdas

        def fake_forward(x):
            """
            Slices lm vector outputs of shape (batch_size, max_seq_len, dims) --> (batch_size, max_seq_len, 2)
            """
            return x.narrow(2, 0, 2)

        def fake_logits_to_preds(logits, **kwargs):
            batch_size = logits.shape[0]
            return [None, None] * batch_size

        def fake_formatted_preds(**kwargs):
            return None

        for ph in self.prediction_heads:
            ph.forward = fake_forward
            ph.logits_to_preds = fake_logits_to_preds
            ph.formatted_preds = fake_formatted_preds

    def save(self, save_dir: Union[str, Path]):
        """
        Saves the language model and prediction heads. This will generate a config file
        and model weights for each.

        :param save_dir: Path to save the AdaptiveModel to.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.language_model.save(save_dir)
        for i, ph in enumerate(self.prediction_heads):
            ph.save(save_dir, i)
            # Need to save config and pipeline

    @classmethod
    def load(  # type: ignore
        cls,
        load_dir: Union[str, Path],
        device: Union[str, torch.device],
        strict: bool = True,
        processor: Optional[Processor] = None,
    ):
        """
        Loads an AdaptiveModel from a directory. The directory must contain:

        * language_model.bin
        * language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Tokens

        :param load_dir: Location where the AdaptiveModel is stored.
        :param device: To which device we want to sent the model, either torch.device("cpu") or torch.device("cuda").
        :param strict: Whether to strictly enforce that the keys loaded from saved model match the ones in
                       the PredictionHead (see torch.nn.module.load_state_dict()).
        :param processor: Processor to populate prediction head with information coming from tasks.
        """
        device = torch.device(device)
        language_model = get_language_model(load_dir)

        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = PredictionHead.load(config_file, strict=strict)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        model = cls(language_model, prediction_heads, 0.1, ph_output_type, device)
        if processor:
            model.connect_heads_with_processor(processor.tasks)

        return model

    @classmethod
    def convert_from_transformers(
        cls,
        model_name_or_path,
        device: Union[str, torch.device],
        revision: Optional[str] = None,
        task_type: str = "question_answering",
        processor: Optional[Processor] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        **kwargs,
    ) -> "AdaptiveModel":
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in Haystack (e.g. take a squad QA model and fine-tune on your own data)
         - compare models without switching frameworks
         - use model directly for inference

        :param model_name_or_path: local path of a saved model or name of a public one.
                                              Exemplary public names:
                                              - distilbert-base-uncased-distilled-squad
                                              - deepset/bert-large-uncased-whole-word-masking-squad2

                                              See https://huggingface.co/models for full list
        :param device: torch.device("cpu") or torch.device("cuda")
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
                         Right now accepts only 'question_answering'.
        :param processor: populates prediction head with information coming from tasks.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :return: AdaptiveModel
        """

        lm = get_language_model(
            model_name_or_path, revision=revision, use_auth_token=use_auth_token, model_kwargs=kwargs
        )
        if task_type is None:
            # Infer task type from config
            architecture = lm.model.config.architectures[0]
            if "QuestionAnswering" in architecture:
                task_type = "question_answering"
            else:
                logger.error(
                    "Could not infer task type from model config. Please provide task type manually. "
                    "('question_answering' or 'embeddings')"
                )

        if task_type == "question_answering":
            ph = QuestionAnsweringHead.load(
                model_name_or_path, revision=revision, use_auth_token=use_auth_token, **kwargs
            )
            adaptive_model = cls(
                language_model=lm,
                prediction_heads=[ph],
                embeds_dropout_prob=0.1,
                lm_output_types="per_token",
                device=device,
            )
        elif task_type == "embeddings":
            adaptive_model = cls(
                language_model=lm,
                prediction_heads=[],
                embeds_dropout_prob=0.1,
                lm_output_types=["per_token", "per_sequence"],
                device=device,
            )

        if processor:
            adaptive_model.connect_heads_with_processor(processor.tasks)

        return adaptive_model

    def convert_to_transformers(self):
        """
        Convert an adaptive model to huggingface's transformers format. Returns a list containing one model for each
        prediction head.

        :return: List of huggingface transformers models.
        """
        converted_models = []

        # convert model for each prediction head
        for prediction_head in self.prediction_heads:
            if len(prediction_head.layer_dims) != 2:
                logger.error(
                    f"Currently conversion only works for PredictionHeads that are a single layer Feed Forward NN with dimensions [LM_output_dim, number_classes].\n"
                    f"            Your PredictionHead has {str(prediction_head.layer_dims)} dimensions."
                )
                continue
            if prediction_head.model_type == "span_classification":
                transformers_model = self._convert_to_transformers_qa(prediction_head)
                converted_models.append(transformers_model)
            else:
                logger.error(
                    f"Haystack -> Transformers conversion is not supported yet for"
                    f" prediction heads of type {prediction_head.model_type}"
                )

        return converted_models

    def _convert_to_transformers_qa(self, prediction_head):
        # TODO add more infos to config

        # remove pooling layer
        self.language_model.model.pooler = None
        # init model
        transformers_model = AutoModelForQuestionAnswering.from_config(self.language_model.model.config)
        # transfer weights for language model + prediction head
        setattr(transformers_model, transformers_model.base_model_prefix, self.language_model.model)
        transformers_model.qa_outputs.load_state_dict(prediction_head.feed_forward.feed_forward[0].state_dict())

        return transformers_model

    def logits_to_loss_per_head(self, logits: torch.Tensor, **kwargs):
        """
        Collect losses from each prediction head.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: The per sample per prediciton head loss whose first two dimensions
                 have length n_pred_heads, batch_size.
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
        :return: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]
        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def prepare_labels(self, **kwargs):
        """
        Label conversion to original label space, per prediction head.

        :param label_maps: dictionary for mapping ids to label strings
        :type label_maps: dict[int:str]
        :return: labels in the right format
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
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """
        Push data through the whole model and returns logits. The data will
        propagate through the language model and each of the attached prediction heads.

        :param input_ids: The IDs of each token in the input sequence. It's a tensor of shape [batch_size, max_seq_len].
        :param segment_ids: The ID of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and the tokens in the second sentence are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len].
        :param padding_mask: A mask that assigns 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len].
        :param output_hidden_states: Whether to output hidden states
        :param output_attentions: Whether to output attentions
        :return: All logits as torch.tensor or multiple tensors.
        """
        # Run forward pass of language model
        output_tuple = self.language_model.forward(
            input_ids=input_ids,
            segment_ids=segment_ids,
            attention_mask=padding_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if output_hidden_states and output_attentions:
            sequence_output, pooled_output, hidden_states, attentions = output_tuple
        elif output_hidden_states:
            sequence_output, pooled_output, hidden_states = output_tuple
        elif output_attentions:
            sequence_output, pooled_output, attentions = output_tuple
        else:
            sequence_output, pooled_output = output_tuple
        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        if len(self.prediction_heads) > 0:
            for head, lm_out in zip(self.prediction_heads, self.lm_output_types):
                # Choose relevant vectors from LM as output and perform dropout
                if lm_out == "per_token":
                    output = self.dropout(sequence_output)
                elif lm_out == "per_sequence" or lm_out == "per_sequence_continuous":
                    output = self.dropout(pooled_output)
                elif (
                    lm_out == "per_token_squad"
                ):  # we need a per_token_squad because of variable metric computation later on...
                    output = self.dropout(sequence_output)
                else:
                    raise ValueError("Unknown extraction strategy from language model: {}".format(lm_out))

                # Do the actual forward pass of a single head
                all_logits.append(head(output))
        else:
            # just return LM output (e.g. useful for extracting embeddings at inference time)
            all_logits.append((sequence_output, pooled_output))

        if output_hidden_states and output_attentions:
            return all_logits, hidden_states, attentions
        if output_hidden_states:
            return all_logits, hidden_states
        if output_attentions:
            return all_logits, attentions
        return all_logits

    def forward_lm(self, **kwargs):
        """
        Forward pass for the language model.

        :return: Tuple containing list of embeddings for each token and
                 embedding for whole sequence.
        """
        # Check if we have to extract from a special layer of the LM (default = last layer)
        try:
            extraction_layer = self.language_model.extraction_layer
        except:
            extraction_layer = -1

        # Run forward pass of language model
        if extraction_layer == -1:
            sequence_output, pooled_output = self.language_model(
                **kwargs, return_dict=False, output_all_encoded_layers=False
            )
        else:
            # get output from an earlier layer
            self.language_model.enable_hidden_states_output()
            sequence_output, pooled_output, all_hidden_states = self.language_model(**kwargs, return_dict=False)
            sequence_output = all_hidden_states[extraction_layer]
            pooled_output = None  # not available in earlier layers
            self.language_model.disable_hidden_states_output()
        return sequence_output, pooled_output

    def log_params(self):
        """
        Logs parameteres to generic logger MlLogger
        """
        params = {
            "lm_type": self.language_model.__class__.__name__,
            "lm_name": self.language_model.name,
            "prediction_heads": ",".join([head.__class__.__name__ for head in self.prediction_heads]),
            "lm_output_types": ",".join(self.lm_output_types),
        }
        try:
            tracker.track_params(params)
        except Exception as e:
            logger.warning("ML logging didn't work: %s", e)

    def verify_vocab_size(self, vocab_size: int):
        """
        Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()
        """
        model_vocab_len = self.language_model.model.resize_token_embeddings(new_num_tokens=None).num_embeddings

        msg = (
            f"Vocab size of tokenizer {vocab_size} doesn't match with model {model_vocab_len}. "
            "If you added a custom vocabulary to the tokenizer, "
            "make sure to supply 'n_added_tokens' to get_language_model() and BertStyleLM.load()"
        )
        assert vocab_size == model_vocab_len, msg

        for head in self.prediction_heads:
            if head.model_type == "language_modelling":
                ph_decoder_len = head.decoder.weight.shape[0]
                assert vocab_size == ph_decoder_len, msg

    def get_language(self):
        return self.language_model.language

    @classmethod
    def convert_to_onnx(
        cls,
        model_name: str,
        output_path: Path,
        task_type: str,
        convert_to_float16: bool = False,
        quantize: bool = False,
        opset_version: int = 11,
        use_auth_token: Optional[Union[str, bool]] = None,
    ):
        """
        Convert a PyTorch model from transformers hub to an ONNX Model.

        :param model_name: Transformers model name.
        :param output_path: Output Path to write the converted model to.
        :param task_type: Type of task for the model. Available options: "question_answering"
        :param convert_to_float16: By default, the model uses float32 precision. With half precision of float16, inference
                                   should be faster on Nvidia GPUs with Tensor core like T4 or V100. On older GPUs, float32
                                   might be more performant.
        :param quantize: Convert floating point number to integers
        :param opset_version: ONNX opset version.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :return: None.
        """
        model_type = capitalize_model_type(_get_model_type(model_name))  # type: ignore
        if model_type not in ["Bert", "Roberta", "XLMRoberta"]:
            raise Exception("The current ONNX conversion only support 'BERT', 'RoBERTa', and 'XLMRoberta' models.")

        task_type_to_pipeline_map = {"question_answering": "question-answering"}

        convert(
            pipeline_name=task_type_to_pipeline_map[task_type],
            framework="pt",
            model=model_name,
            output=output_path / "model.onnx",
            opset=opset_version,
            use_external_format=True if model_type == "XLMRoberta" else False,
            use_auth_token=use_auth_token,
        )

        # save processor & model config files that are needed when loading the model with the Haystack.basics Inferencer
        processor = Processor.convert_from_transformers(
            tokenizer_name_or_path=model_name,
            task_type=task_type,
            max_seq_len=256,
            doc_stride=128,
            use_fast=True,
            use_auth_token=use_auth_token,
        )
        processor.save(output_path)
        model = AdaptiveModel.convert_from_transformers(
            model_name, device=torch.device("cpu"), task_type=task_type, use_auth_token=use_auth_token
        )
        model.save(output_path)
        os.remove(output_path / "language_model.bin")  # remove the actual PyTorch model(only configs are required)

        onnx_model_config = {
            "task_type": task_type,
            "onnx_opset_version": opset_version,
            "language_model_class": model_type,
            "language": model.language_model.language,
        }
        with open(output_path / "onnx_model_config.json", "w") as f:
            json.dump(onnx_model_config, f)

        if convert_to_float16:
            from onnxruntime_tools import optimizer

            config = AutoConfig.from_pretrained(model_name, use_auth_token=use_auth_token)
            optimized_model = optimizer.optimize_model(
                input=str(output_path / "model.onnx"),
                model_type="bert",
                num_heads=config.num_hidden_layers,
                hidden_size=config.hidden_size,
            )
            optimized_model.convert_model_float32_to_float16()
            optimized_model.save_model_to_file("model.onnx")

        if quantize:
            quantize_model(output_path / "model.onnx")


class ONNXAdaptiveModel(BaseAdaptiveModel):
    """
    Implementation of ONNX Runtime for Inference of ONNX Models.

    Existing PyTorch based Haystack.basics AdaptiveModel can be converted to ONNX format using AdaptiveModel.convert_to_onnx().
    The conversion is currently only implemented for Question Answering Models.

    For inference, this class is compatible with the Haystack.basics Inferencer.
    """

    # TODO validate usefulness
    def __init__(
        self,
        onnx_session,  # TODO
        language_model_class: str,
        language: str,
        prediction_heads: List[PredictionHead],
        device: torch.device,
    ):
        """
        :param onnx_session: ? # TODO
        :param language_model_class: Class of LanguageModel
        :param language: Language the model is trained for.
        :param prediction_heads: A list of models that take embeddings and return logits for a given task.
        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").
        """
        import onnxruntime

        super().__init__(prediction_heads)

        if str(device) == "cuda" and onnxruntime.get_device() != "GPU":
            raise Exception(
                f"Device {device} not available for Inference. For CPU, run pip install onnxruntime and"
                f"for GPU run pip install onnxruntime-gpu"
            )
        self.onnx_session = onnx_session
        self.language_model_class = language_model_class
        self.language = language
        self.prediction_heads = prediction_heads
        self.device = device

    @classmethod
    def load(cls, load_dir: Union[str, Path], device: Union[str, torch.device], **kwargs):  # type: ignore
        """
        Loads an ONNXAdaptiveModel from a directory.

        :param load_dir: Location where the ONNXAdaptiveModel is stored.
        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").
        """
        device = torch.device(device)
        load_dir = Path(load_dir)
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()

        providers = kwargs.get(
            "providers", ["CPUExecutionProvider"] if device.type == "cpu" else ["CUDAExecutionProvider"]
        )
        onnx_session = onnxruntime.InferenceSession(str(load_dir / "model.onnx"), sess_options, providers=providers)

        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir, strict=False)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            # ONNX Model doesn't need have a separate neural network for PredictionHead. It only uses the
            # instance methods of PredictionHead class, so, we load with the load_weights param as False.
            head = PredictionHead.load(config_file, load_weights=False)
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        with open(load_dir / "onnx_model_config.json") as f:
            model_config = json.load(f)
            language_model_class = model_config["language_model_class"]
            language = model_config["language"]

        return cls(onnx_session, language_model_class, language, prediction_heads, device)

    def forward(self, **kwargs):
        """
        Perform forward pass on the model and return the logits.

        :param kwargs: All arguments that need to be passed on to the model.
        :return: All logits as torch.tensor or multiple tensors.
        """
        with torch.inference_mode():
            if self.language_model_class == "Bert":
                input_to_onnx = {
                    "input_ids": numpy.ascontiguousarray(kwargs["input_ids"].cpu().numpy()),
                    "attention_mask": numpy.ascontiguousarray(kwargs["padding_mask"].cpu().numpy()),
                    "token_type_ids": numpy.ascontiguousarray(kwargs["segment_ids"].cpu().numpy()),
                }
            elif self.language_model_class in ["Roberta", "XLMRoberta"]:
                input_to_onnx = {
                    "input_ids": numpy.ascontiguousarray(kwargs["input_ids"].cpu().numpy()),
                    "attention_mask": numpy.ascontiguousarray(kwargs["padding_mask"].cpu().numpy()),
                }
            res = self.onnx_session.run(None, input_to_onnx)
            res = numpy.stack(res).transpose(1, 2, 0)
            logits = [torch.Tensor(res).to(self.device)]

        return logits

    def eval(self):
        """
        Stub to make ONNXAdaptiveModel compatible with the PyTorch AdaptiveModel.
        """
        return True

    def get_language(self):
        """
        Get the language(s) the model was trained for.
        :return: str
        """
        return self.language


class ONNXWrapper(AdaptiveModel):
    """
    Wrapper Class for converting PyTorch models to ONNX.

    As of torch v1.4.0, torch.onnx.export only support passing positional arguments
    to the forward pass of the model. However, the AdaptiveModel's forward takes keyword arguments.
    This class circumvents the issue by converting positional arguments to keyword arguments.
    """

    @classmethod
    def load_from_adaptive_model(cls, adaptive_model: AdaptiveModel):
        model = copy.deepcopy(adaptive_model)
        model.__class__ = ONNXWrapper
        return model

    def forward(self, *batch):
        return super().forward(input_ids=batch[0], padding_mask=batch[1], segment_ids=batch[2])
