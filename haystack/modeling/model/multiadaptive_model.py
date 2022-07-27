from typing import List, Dict, Union, Optional, Literal

import logging
from pathlib import Path


import torch
from haystack.errors import ModelingError


logger = logging.getLogger(__name__)


import logging
import os
from pathlib import Path
from typing import List, Optional, Callable, Union, Dict, Tuple

import torch
from torch import nn

from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.prediction_head import PredictionHead
from haystack.utils.experiment_tracking import Tracker as tracker
from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


ModelTypes = Union[Literal["query"], ContentTypes]


class MultiAdaptiveModel(nn.Module):
    """
    PyTorch implementation containing all the modeling needed for
    your NLP task. Combines N language models for representation of N
    sequences and a prediction head. Allows for gradient flow back to
    the language model components.

    The MultiAdaptiveModel is currently tailored to the use case of
    multimodal retrieval using one encoder as question encoder and the
    others as encoders for each different data type (text, tables, images,
    audio, etc...)
    """

    def __init__(
        self,
        query_model: LanguageModel,
        context_models: Dict[ContentTypes, LanguageModel],
        prediction_heads: List[PredictionHead],
        embeds_dropout_prob: float = 0.1,
        device: torch.device = torch.device("cuda"),
        query_output_types: List[str] = ["per_sequence"],
        context_output_types: List[str] = ["per_sequence"],
        loss_aggregation_fn: Optional[Callable] = sum,
    ):
        """
        :param query_model: Any model that turns token ids into vector representations.
        :param context_models: Dict of models that turns token ids into vector representations,
            indexed by the content type they should be used for (text, image, etc).
        :param prediction_heads: A list of models that take as many sequence embeddings as
            `len(context_models) + 1` and return logits for a given task.
        :param embeds_dropout_prob: The probability that a value in the embeddings returned by any of the three
            language model will be zeroed.
        :param query_output_types: How to extract the embeddings from the final layer of the first language model.
            When set to "per_token", one embedding will be extracted per input token.
            When set to "per_sequence", a single embedding will be extracted to represent the full input sequence.
            Can either be a single string, or a list of strings, one for each prediction head.
        :param context_output_types: How to extract the embeddings from the final layer of the first language model.
            When set to "per_token", one embedding will be extracted per input token.
            When set to "per_sequence", a single embedding will be extracted to represent the full input sequence.
            Can either be a single string, or a list of strings, one for each prediction head.
        :param device: The device on which this model will operate. Either torch.device("cpu") or torch.device("cuda").
        :param loss_aggregation_fn: Function to aggregate the loss of multiple prediction heads.
            Input: loss_per_head (list of tensors), global_step (int), batch (dict)
            Output: aggregated loss (tensor)
            Default is a simple sum: `lambda loss_per_head, global_step=None, batch=None: sum(tensors)`
            However, you can pass more complex functions that depend on the current step (e.g. for round-robin
            style multitask learning) or the content of the batch (e.g. certain labels)
            Note: The loss at this stage is per sample, i.e one tensor of shape (batchsize) per prediction head.
        """
        super().__init__()

        self.device = device
        self.loss_aggregation_fn = loss_aggregation_fn
        self.supported_content_types = [key for key in context_models]

        self.models = {
            content_type: model.to(device)
            for content_type, model in context_models.items()
        }
        self.models["query"] = query_model.to(device)

        self.query_dropout = nn.Dropout(embeds_dropout_prob)
        self.context_dropout = nn.Dropout(embeds_dropout_prob)
        self.query_output_types = query_output_types
        self.context_output_types = context_output_types

        self.prediction_heads = nn.ModuleList([ph.to(device) for ph in prediction_heads])

        self.log_params()


    def save(self, save_dir: Path, query_model_name: Optional[str] = None, context_model_names: Optional[Dict[str, str]] = None):
        """
        Saves the language model weights and respective config_files in named directories within save_dir.
        Saves the prediction heads as well
        """
        context_model_names = context_model_names or {key: key for key in self.supported_content_types}
        if list(context_model_names.keys()) != self.supported_content_types:
            logging.warning(
                "The list of names does not correspond to the list of models initialized. "
                "Unmatched models will be named after the content type they support. "
                f"(context models: {', '.join([key for key in self.models.keys() if key != 'query'])} - "
                f"names received: {context_model_names})"
            )
            context_model_names = {key: context_model_names.get(key, key) for key in self.supported_content_types}

        model_names = context_model_names.update({"query": query_model_name or "query"})

        os.makedirs(save_dir, exist_ok=True)
        for content_type, name in model_names.items():
            if not os.path.exists(save_dir / name):
                os.makedirs(save_dir / name)
            self.models[content_type].save(save_dir / name)

        for i, ph in enumerate(self.prediction_heads):
            ph.save(save_dir, i)


    def logits_to_loss(self, logits: torch.Tensor, global_step: Optional[int] = None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: logits, can vary in shape and type, depending on task.
        :param global_step: number of current training step.
        :param kwargs: Placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :return: loss: torch.Tensor that is the per sample loss (len: batch_size)
        """
        all_losses = []
        for head, logits_for_one_head in zip(self.prediction_heads, logits):
            # check if PredictionHead connected to Processor
            if not hasattr(head, "label_tensor_name"):
                raise ModelingError(
                f"Label_tensor_names are missing inside the {head.task_name} Prediction Head. Did you connect the model"
                " with the processor through either 'model.connect_heads_with_processor(processor.tasks)'"
                " or by passing the processor to MultiAdaptiveModel?"
            )
            all_losses.append(head.logits_to_loss(logits=logits_for_one_head))#, **kwargs))

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


    def forward(self, inputs_by_model: Dict[ModelTypes, Dict[str, torch.Tensor]]):
        """
        Push data through the whole model and returns logits. The data will propagate through
        the first language model, second and third language model based on the tensor names and both the
        encodings through each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to both the language models and prediction head(s).
        :return: All logits as torch.Tensor or multiple tensors.
        """
        pooled_outputs = self.forward_lm(inputs_by_model)

        # just return LM output (e.g. useful for extracting embeddings at inference time)
        if len(self.prediction_heads) == 0:
            return [((pooled_outputs["query"], pooled_outputs["context"]))]

        # Run forward pass of (multiple) prediction heads using the output from above
        all_logits = []
        for head, query_output_type, context_output_type in zip(self.prediction_heads, self.query_output_types, self.context_output_types):
            outputs = (None, None)

            # Choose relevant vectors from LM as output and perform dropout
            if pooled_outputs["query"]:
                if not query_output_type in ["per_sequence", "per_sequence_continuous"]:
                    raise ModelingError(
                        f"Unknown extraction strategy from query language model: {query_output_type}"
                    )
                outputs[0] = self.query_dropout(pooled_outputs["query"])

            if pooled_outputs["context"]:
                if not context_output_type in ["per_sequence", "per_sequence_continuous"]:
                    raise ModelingError(
                        f"Unknown extraction strategy from context language models: {context_output_type}"
                    )
                outputs[1] = self.context_dropout(pooled_outputs["context"])

            all_logits.append(head(*outputs))

        return all_logits


    def forward_lm(self, inputs_by_model: Dict[ModelTypes, Dict[str, torch.Tensor]]) -> Dict[str, Tuple[torch.Tensor]]:
        """
        Forward pass for the MultiAdaptiveModel.

        :param inputs_by_model: Arguments that need to be passed to the language models, by content_type.
        :return: Two tensors of pooled_output, one from the query, and one from the context models (merged together)
        """
        # Query forward pass
        # Note: **inputs might be unavoidable here. Different model types take different input vectors.
        query_pooled_outputs, _ = self.models["query"](inputs_by_model["query"])

        # Contexts forward pass
        contexts_pooled_outputs = {}
        for key, inputs in inputs_by_model.keys():

            model = self.models.get(key)
            if not model:
                logging.error(
                    f"Some input tensor were passed for models handling {key} data, "
                    "but no such model was initialized. They will be ignored."
                    f"Initialized models: {', '.join(self.supported_content_types)}"
                )
            else:
                # Note: **inputs might be unavoidable here. Different model types take different input vectors.
                contexts_pooled_outputs[key], _ = self.models[key](**inputs)

        # Combine the outputs
        embedding_sizes = [pooled_output.shape[-1] for pooled_output in contexts_pooled_outputs.values()]
        if not all(embedding_size == embedding_sizes[0] for embedding_size in embedding_sizes):
            raise ModelingError(
                "Some of the models are using a different embedding size. They should all match. "
                f"Embedding sizes by model: "
                f"{ {name: output.shape[-1] for name, output in contexts_pooled_outputs.items()} }")

        combined_contexts_pooled_outputs = torch.stack(contexts_pooled_outputs.values())
        combined_contexts_pooled_outputs = combined_contexts_pooled_outputs.view(-1, embedding_sizes[0])

        return {"query": query_pooled_outputs, "context": combined_contexts_pooled_outputs}


    def log_params(self):
        """
        Logs paramteres to generic logger MlLogger
        """
        params = {}
        for name, model in self.models.items():
            params[f"{name}_model_class"] = model.__class__.__name__
            params[f"{name}_model_name"] = model.name,
            params[f"{name}_model_output_types"] = ",".join(self.query_output_types if name == "query" else self.context_output_types)

        params["prediction_heads"] = ",".join([head.__class__.__name__ for head in self.prediction_heads])
        try:
            tracker.track_params(params)
        except Exception as e:
            logger.warning(f"ML logging failed: {e}")


    def verify_vocab_size(self, vocab_sizes: Dict[ContentTypes, int]):
        """
        Verifies that the model fits to the tokenizer vocabulary.
        They could diverge in case of custom vocabulary added via tokenizer.add_tokens()
        """
        for name, vocab_size in vocab_sizes.items():
            vocab_len = self.models[name].model.resize_token_embeddings(new_num_tokens=None).num_embeddings
            if not vocab_size == vocab_len:
                raise ModelingError(
                    f"The vocabulary size of the tokenizer ({vocab_size}) doesn't match with "
                    f"the vocabulary size of the model ({vocab_len}). "
                    "If you added custom vocabulary to the tokenizer, "
                    "make sure to supply 'n_added_tokens' to `get_language_model()`"
                )

    def get_language(self) -> Dict[ContentTypes, str]:
        return {name: model.language for name, model in self.models}


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
                raise ModelingError(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]["label_list"]
            head.label_list = label_list
            head.metric = tasks[head.task_name]["metric"]


    def logits_to_preds(self, logits: torch.Tensor, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: Logits, can vary in shape and type, depending on task.
        :return: A list of all predictions from all prediction heads.
        """
        return [head.logits_to_preds(logits=logits, **kwargs) for head, logits in zip(self.prediction_heads, logits)]
