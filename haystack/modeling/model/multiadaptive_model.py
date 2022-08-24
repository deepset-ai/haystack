from typing import Dict

import logging
import torch
from torch import nn

from haystack.modeling.model.multimodal_language_model import MultiModalLanguageModel
from haystack.errors import ModelingError
from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


class MultiAdaptiveModel(nn.Module):
    """
    PyTorch implementation containing all the modeling needed for
    your NLP task. Combines N language models for representation of N
    sequences.

    The MultiAdaptiveModel is currently tailored to the use case of
    multimodal retrieval using one encoder as question encoder and the
    others as encoders for each different data type (text, tables, images,
    audio, etc...)
    """

    # TODO If we implement multimodal training, this is where it should go.

    def __init__(
        self, models: Dict[ContentTypes, MultiModalLanguageModel], device: torch.device = torch.device("cuda")
    ):
        """
        :param models: Any model that turns token ids into vector representations.
        :param device: The device on which this model will operate. Like torch.device("cpu"), torch.device("cuda"), etc.
        """
        super().__init__()

        self.device = device
        self.supported_content_types = [key for key in models]
        self.models = {content_type: model.to(device) for content_type, model in models.items()}

    def forward(self, inputs_by_model: Dict[ContentTypes, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Push data through the whole model and returns logits. The data will propagate through the model
        corresponding to their data type.

        :param inputs_by_model: Holds all tensors that need to be passed to the language models, by type.
        :return: All logits as torch.Tensor.
        """
        # Get output for each model
        outputs_by_type: Dict[ContentTypes, torch.Tensor] = {}
        for key, inputs in inputs_by_model.items():

            model = self.models.get(key)
            if not model:
                raise ModelingError(
                    f"Some input tensor were passed for models handling {key} data, "
                    "but no such model was initialized. They will be ignored."
                    f"Initialized models: {', '.join(self.supported_content_types)}"
                )

            # Note: **inputs is unavoidable here. Different model types take different input vectors.
            # Validation of the inputs occurrs in the forward() method.
            outputs_by_type[key] = self.models[key].forward(**inputs)

        # Check the output sizes
        embedding_sizes = [output.shape[-1] for output in outputs_by_type.values()]
        if not all(embedding_size == embedding_sizes[0] for embedding_size in embedding_sizes):
            raise ModelingError(
                "Some of the models are using a different embedding size. They should all match. "
                f"Embedding sizes by model: "
                f"{ {name: output.shape[-1] for name, output in outputs_by_type.items()} }"
            )

        # Combine the outputs in a single matrix
        outputs = torch.stack(list(outputs_by_type.values()))
        outputs = outputs.view(-1, embedding_sizes[0])

        return outputs
