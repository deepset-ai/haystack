# this code is taken and adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines.py

import csv
import json
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import chain
from os.path import abspath, exists
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from psutil import cpu_count
from transformers.configuration_auto import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.convert_graph_to_onnx import convert_pytorch, convert_tensorflow, infer_shapes
from transformers.data import SquadExample, squad_convert_examples_to_features
from transformers.file_utils import add_end_docstrings, is_tf_available, is_torch_available
from transformers.modelcard import ModelCard
from transformers.tokenization_auto import AutoTokenizer
from transformers.tokenization_bert import BasicTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.utils import logging


if is_tf_available():
    import tensorflow as tf
    from transformers.modeling_tf_auto import (
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
        TFAutoModelWithLMHead,
    )

if is_torch_available():
    import torch
    from transformers.modeling_auto import (
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

if TYPE_CHECKING:
    from transformers.modeling_tf_utils import TFPreTrainedModel
    from transformers.modeling_utils import PreTrainedModel


ONNX_CACHE_DIR = Path(os.path.dirname(__file__)).parent.joinpath(".onnx")


logger = logging.get_logger(__name__)


# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:

    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


def get_framework(model=None):
    """
    Select framework (TensorFlow or PyTorch) to use.

    Args:
        model (:obj:`str`, :class:`~transformers.PreTrainedModel` or :class:`~transformers.TFPreTrainedModel`, `optional`):
            If both frameworks are installed, picks the one corresponding to the model passed (either a model class or
            the model name). If no specific model is provided, defaults to using PyTorch.
    """
    if is_tf_available() and is_torch_available() and model is not None and not isinstance(model, str):
        # Both framework are available but the user supplied a model class instance.
        # Try to guess which framework to use from the model classname
        framework = "tf" if model.__class__.__name__.startswith("TF") else "pt"
    elif not is_tf_available() and not is_torch_available():
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )
    else:
        # framework = 'tf' if is_tf_available() else 'pt'
        framework = "pt" if is_torch_available() else "tf"
    return framework


class PipelineException(Exception):
    """
    Raised by a :class:`~transformers.Pipeline` when handling __call__.

    Args:
        task (:obj:`str`): The task of the pipeline.
        model (:obj:`str`): The model used by the pipeline.
        reason (:obj:`str`): The error message to display.
    """

    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each :class:`~transformers.pipelines.Pipeline`.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultArgumentHandler(ArgumentHandler):
    """
    Default argument parser handling parameters for each :class:`~transformers.pipelines.Pipeline`.
    """

    @staticmethod
    def handle_kwargs(kwargs: Dict) -> List:
        if len(kwargs) == 1:
            output = list(kwargs.values())
        else:
            output = list(chain(kwargs.values()))

        return DefaultArgumentHandler.handle_args(output)

    @staticmethod
    def handle_args(args: Sequence[Any]) -> List[str]:

        # Only one argument, let's do case by case
        if len(args) == 1:
            if isinstance(args[0], str):
                return [args[0]]
            elif not isinstance(args[0], list):
                return list(args)
            else:
                return args[0]

        # Multiple arguments (x1, x2, ...)
        elif len(args) > 1:
            if all([isinstance(arg, str) for arg in args]):
                return list(args)

            # If not instance of list, then it should instance of iterable
            elif isinstance(args, Iterable):
                return list(chain.from_iterable(chain(args)))
            else:
                raise ValueError(
                    "Invalid input type {}. Pipeline supports Union[str, Iterable[str]]".format(type(args))
                )
        else:
            return []

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0 and len(args) > 0:
            raise ValueError("Pipeline cannot handle mixed args and kwargs")

        if len(kwargs) > 0:
            return DefaultArgumentHandler.handle_kwargs(kwargs)
        else:
            return DefaultArgumentHandler.handle_args(args)


class PipelineDataFormat:
    """
    Base class for all the pipeline supported data format both for reading and writing.
    Supported data formats currently includes:
    - JSON
    - CSV
    - stdin/stdout (pipe)

    :obj:`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets
    columns to pipelines keyword arguments through the :obj:`dataset_kwarg_1=dataset_column_1` format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    SUPPORTED_FORMATS = ["json", "csv", "pipe"]

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite: bool = False,
    ):
        self.output_path = output_path
        self.input_path = input_path
        self.column = column.split(",") if column is not None else [""]
        self.is_multi_columns = len(self.column) > 1

        if self.is_multi_columns:
            self.column = [tuple(c.split("=")) if "=" in c else (c, c) for c in self.column]

        if output_path is not None and not overwrite:
            if exists(abspath(self.output_path)):
                raise OSError("{} already exists on disk".format(self.output_path))

        if input_path is not None:
            if not exists(abspath(self.input_path)):
                raise OSError("{} doesnt exist on disk".format(self.input_path))

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, data: Union[dict, List[dict]]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.
        """
        raise NotImplementedError()

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        """
        Save the provided data object as a pickle-formatted binary data on the disk.

        Args:
            data (:obj:`dict` or list of :obj:`dict`): The data to store.

        Returns:
            :obj:`str`: Path where the data has been saved.
        """
        path, _ = os.path.splitext(self.output_path)
        binary_path = os.path.extsep.join((path, "pickle"))

        with open(binary_path, "wb+") as f_output:
            pickle.dump(data, f_output)

        return binary_path

    @staticmethod
    def from_str(
        format: str,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ) -> "PipelineDataFormat":
        """
        Creates an instance of the right subclass of :class:`~transformers.pipelines.PipelineDataFormat` depending
        on :obj:`format`.

        Args:
            format: (:obj:`str`):
                The format of the desired pipeline. Acceptable values are :obj:`"json"`, :obj:`"csv"` or :obj:`"pipe"`.
            output_path (:obj:`str`, `optional`):
                Where to save the outgoing data.
            input_path (:obj:`str`, `optional`):
                Where to look for the input data.
            column (:obj:`str`, `optional`):
                The column to read.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to overwrite the :obj:`output_path`.

        Returns:
            :class:`~transformers.pipelines.PipelineDataFormat`: The proper data format.
        """
        if format == "json":
            return JsonPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "csv":
            return CsvPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        elif format == "pipe":
            return PipedPipelineDataFormat(output_path, input_path, column, overwrite=overwrite)
        else:
            raise KeyError("Unknown reader {} (Available reader are json/csv/pipe)".format(format))


class CsvPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using CSV data format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

    def __iter__(self):
        with open(self.input_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.is_multi_columns:
                    yield {k: row[c] for k, c in self.column}
                else:
                    yield row[self.column[0]]

    def save(self, data: List[dict]):
        """
        Save the provided data object with the representation for the current
        :class:`~transformers.pipelines.PipelineDataFormat`.

        Args:
            data (:obj:`List[dict]`): The data to store.
        """
        with open(self.output_path, "w") as f:
            if len(data) > 0:
                writer = csv.DictWriter(f, list(data[0].keys()))
                writer.writeheader()
                writer.writerows(data)


class JsonPipelineDataFormat(PipelineDataFormat):
    """
    Support for pipelines using JSON file format.

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    def __init__(
        self,
        output_path: Optional[str],
        input_path: Optional[str],
        column: Optional[str],
        overwrite=False,
    ):
        super().__init__(output_path, input_path, column, overwrite=overwrite)

        with open(input_path, "r") as f:
            self._entries = json.load(f)

    def __iter__(self):
        for entry in self._entries:
            if self.is_multi_columns:
                yield {k: entry[c] for k, c in self.column}
            else:
                yield entry[self.column[0]]

    def save(self, data: dict):
        """
        Save the provided data object in a json file.

        Args:
            data (:obj:`dict`): The data to store.
        """
        with open(self.output_path, "w") as f:
            json.dump(data, f)


class PipedPipelineDataFormat(PipelineDataFormat):
    """
    Read data from piped input to the python process.
    For multi columns data, columns should separated by \t

    If columns are provided, then the output will be a dictionary with {column_x: value_x}

    Args:
        output_path (:obj:`str`, `optional`): Where to save the outgoing data.
        input_path (:obj:`str`, `optional`): Where to look for the input data.
        column (:obj:`str`, `optional`): The column to read.
        overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to overwrite the :obj:`output_path`.
    """

    def __iter__(self):
        for line in sys.stdin:
            # Split for multi-columns
            if "\t" in line:

                line = line.split("\t")
                if self.column:
                    # Dictionary to map arguments
                    yield {kwargs: l for (kwargs, _), l in zip(self.column, line)}
                else:
                    yield tuple(line)

            # No dictionary to map arguments
            else:
                yield line

    def save(self, data: dict):
        """
        Print the data.

        Args:
            data (:obj:`dict`): The data to store.
        """
        print(data)

    def save_binary(self, data: Union[dict, List[dict]]) -> str:
        if self.output_path is None:
            raise KeyError(
                "When using piped input on pipeline outputting large object requires an output file path. "
                "Please provide such output path through --output argument."
            )

        return super().save_binary(data)


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()


PIPELINE_INIT_ARGS = r"""
    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no
            model is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model
            on the associated CUDA device id.
        binary_output (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Flag indicating if the output the pipeline should happen in a binary format (i.e., pickle) or as raw text.
"""


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Pipeline(_ScikitCompat):
    """
    The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
    different pipelines.

    Base class implementing pipelined operations.
    Pipeline workflow is defined as a sequence of the following operations:

        Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

    Pipeline supports running on CPU or GPU through the device argument or using onnx runtime (see below).

    Some pipeline, like for instance :class:`~transformers.FeatureExtractionPipeline` (:obj:`'feature-extraction'` )
    output large tensor object as nested-lists. In order to avoid dumping such large structure as textual data we
    provide the :obj:`binary_output` constructor argument. If set to :obj:`True`, the output will be stored in the
    pickle format.
    """

    default_input_names = None

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel", str],
        tokenizer: PreTrainedTokenizer,
        config: PretrainedConfig,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        onnx: bool = True,
        graph_path: Optional[Path] = None,
    ):

        if framework is None:
            framework = get_framework(model)

        self.onnx = onnx
        self.graph_path = graph_path
        self.task = task
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.framework = framework
        self.device = device if framework == "tf" else torch.device("cpu" if device < 0 else "cuda:{}".format(device))
        self.binary_output = binary_output
        self._args_parser = args_parser or DefaultArgumentHandler()

        # Special handling
        if self.framework == "pt" and self.device.type == "cuda" and (not onnx):
            self.model = self.model.to(self.device)

        # Export the graph
        if onnx:
            input_names_path = graph_path.parent.joinpath(f"{os.path.basename(graph_path)}.input_names.json")
            if not graph_path.exists() or not input_names_path.exists():
                self._export_onnx_graph(input_names_path)

            logger.info(f"loading onnx graph from {self.graph_path.as_posix()}")
            self.onnx_model = create_model_for_provider(str(graph_path), "CPUExecutionProvider")
            self.input_names = json.load(open(input_names_path))
            self.framework = "np"
            self._warup_onnx_graph()

        # TODO: handle this
        # Update config with task specific parameters
        # task_specific_params = self.model.config.task_specific_params
        # if task_specific_params is not None and task in task_specific_params:
        #     self.model.config.update(task_specific_params.get(task))

    def save_pretrained(self, save_directory: str):
        """
        Save the pipeline's model and tokenizer.

        Args:
            save_directory (:obj:`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        if self.modelcard is not None:
            self.modelcard.save_pretrained(save_directory)

    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples::

            # Explicitly ask for tensor allocation on CUDA device :0
            pipe = pipeline(..., device=0)
            with pipe.device_placement():
                # Every framework specific tensor allocation will be done on the request device
                output = pipe(...)
        """
        if self.framework == "tf":
            with tf.device("/CPU:0" if self.device == -1 else "/device:GPU:{}".format(self.device)):
                yield
        else:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

            yield

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be :obj:`torch.Tensor`): The tensors to place on :obj:`self.device`.

        Return:
            :obj:`Dict[str, torch.Tensor]`: The same as :obj:`inputs` but on the proper device.
        """
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def check_model_type(self, supported_models: Union[List[str], dict]):
        """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (:obj:`List[str]` or :obj:`dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
        if not isinstance(supported_models, list):  # Create from a model mapping
            supported_models = [item[1].__name__ for item in supported_models.items()]
        if self.model.__class__.__name__ not in supported_models:
            raise PipelineException(
                self.task,
                self.model.base_model_prefix,
                f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}",
            )

    def _parse_and_tokenize(self, *args, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
        )

        return inputs

    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        if self.onnx:
            return self._forward_onnx(inputs)
        else:
            return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching.
        Args:
            inputs: dict holding all the keyworded arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array.
        Returns:
            Numpy array
        """
        # Encode for forward
        with self.device_placement():
            if self.framework == "tf":
                # TODO trace model
                predictions = self.model(inputs.data, training=False)[0]
            else:
                with torch.no_grad():
                    inputs = self.ensure_tensor_on_device(**inputs)
                    predictions = self.model(**inputs)[0].cpu()

        if return_tensors:
            return predictions
        else:
            return predictions.numpy()

    def _export_onnx_graph(self, input_names_path: Path):
        # if graph exists, but we are here then it means something went wrong in previous load
        # so delete old graph
        if self.graph_path.exists():
            self.graph_path.unlink()
        if input_names_path.exists():
            input_names_path.unlink()

        # create parent dir
        if not self.graph_path.parent.exists():
            os.makedirs(self.graph_path.parent.as_posix())
       
        logger.info(f"Saving onnx graph at { self.graph_path.as_posix()}")

        if self.framework == "pt":
            convert_pytorch(self, opset=11, output=self.graph_path, use_external_format=False)
        else:
            convert_tensorflow(self, opset=11, output=self.graph_path)

        # save input names
        self.input_names = infer_shapes(self, "pt")[0]
        with open(input_names_path, "w") as f:
            json.dump(self.input_names, f)

    def _forward_onnx(self, inputs, return_tensors=False):
        # inputs_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items() if k in self.input_names}
        inputs_onnx = {k: v for k, v in inputs.items() if k in self.input_names}
        predictions = self.onnx_model.run(None, inputs_onnx)
        return predictions

    def _warup_onnx_graph(self, n=10):
        model_inputs = self.tokenizer("My name is Bert", return_tensors="np")
        for _ in range(n):
            self._forward_onnx(model_inputs)


# Can't use @add_end_docstrings(PIPELINE_INIT_ARGS) here because this one does not accept `binary_output`
class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline using no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    This feature extraction pipeline can currently be loaded from :func:`~transformers.pipeline` using the task
    identifier: :obj:`"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    `huggingface.co/models <https://huggingface.co/models>`__.

    Arguments:
        model (:obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            :class:`~transformers.PreTrainedModel` for PyTorch and :class:`~transformers.TFPreTrainedModel` for
            TensorFlow.
        tokenizer (:obj:`~transformers.PreTrainedTokenizer`):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            :class:`~transformers.PreTrainedTokenizer`.
        modelcard (:obj:`str` or :class:`~transformers.ModelCard`, `optional`):
            Model card attributed to the model for this pipeline.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no
            model is provided.
        task (:obj:`str`, defaults to :obj:`""`):
            A task-identifier for the pipeline.
        args_parser (:class:`~transformers.pipelines.ArgumentHandler`, `optional`):
            Reference to the object in charge of parsing supplied pipeline parameters.
        device (:obj:`int`, `optional`, defaults to -1):
            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model
            on the associated CUDA device id.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        config: PretrainedConfig,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            config=config,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (:obj:`str` or :obj:`List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of :obj:`float`: The features computed by the model.
        """
        output = super().__call__(*args, **kwargs)
        if self.onnx:
            return output[0].tolist()
        return output.tolist()


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return all prediction scores or just the one of the predicted class.
    """,
)
class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any :obj:`ModelForSequenceClassification`. See the
    `sequence classification examples <../task_summary.html#sequence-classification>`__ for more information.

    This text classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"sentiment-analysis"` (for classifying sequences according to positive or negative
    sentiments).

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=text-classification>`__.
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        # self.check_model_type(
        #     TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        #     if self.framework == "tf"
        #     else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        # )

        self.return_all_scores = return_all_scores

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several textts (or one list of prompts) to classify.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the
            following keys:

            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.

            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [{"label": self.config.id2label[i], "score": score.item()} for i, score in enumerate(item)]
                for item in scores
            ]
        else:
            return [{"label": self.config.id2label[item.argmax()], "score": item.max().item()} for item in scores]


class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """

    def _parse_labels(self, labels):
        if isinstance(labels, str):
            labels = [label.strip() for label in labels.split(",")]
        return labels

    def __call__(self, sequences, labels, hypothesis_template):
        if len(labels) == 0 or len(sequences) == 0:
            raise ValueError("You must include at least one label and at least one sequence.")
        if hypothesis_template.format(labels[0]) == hypothesis_template:
            raise ValueError(
                (
                    'The provided hypothesis_template "{}" was not able to be formatted with the target labels. '
                    "Make sure the passed template includes formatting syntax such as {{}} where the label should go."
                ).format(hypothesis_template)
            )

        if isinstance(sequences, str):
            sequences = [sequences]
        labels = self._parse_labels(labels)

        sequence_pairs = []
        for sequence in sequences:
            sequence_pairs.extend([[sequence, hypothesis_template.format(label)] for label in labels])

        return sequence_pairs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotClassificationPipeline(Pipeline):
    """
    NLI-based zero-shot classification pipeline using a :obj:`ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for `entailment` is taken as the logit for the
    candidate label being valid. Any NLI model can be used as long as the first output logit corresponds to
    `contradiction` and the last to `entailment`.

    This NLI pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?search=nli>`__.
    """

    def __init__(self, args_parser=ZeroShotClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, args_parser=args_parser, **kwargs)

    def _parse_and_tokenize(self, *args, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        inputs = self._args_parser(*args, **kwargs)
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors=self.framework,
            padding=padding,
            truncation="only_first",
        )

        return inputs

    def __call__(self, sequences, candidate_labels, hypothesis_template="This example is {}.", multi_class=False):
        """
        Classify the sequence(s) given as inputs.

        Args:
            sequences (:obj:`str` or :obj:`List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (:obj:`str` or :obj:`List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (:obj:`str`, `optional`, defaults to :obj:`"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {}
                or similar syntax for the candidate label to be inserted into the template. For example, the default
                template is :obj:`"This example is {}."` With the candidate label :obj:`"sports"`, this would be fed
                into the model like :obj:`"<cls> sequence to classify <sep> This example is sports . <sep>"`. The
                default template works well in many cases, but it may be worthwhile to experiment with different
                templates depending on the task setting.
            multi_class (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not multiple candidate labels can be true. If :obj:`False`, the scores are normalized
                such that the sum of the label likelihoods for each sequence is 1. If :obj:`True`, the labels are
                considered independent and probabilities are normalized for each candidate by doing a softmax of
                the entailment score vs. the contradiction score.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the
            following keys:

            - **sequence** (:obj:`str`) -- The sequence for which this is the output.
            - **labels** (:obj:`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (:obj:`List[float]`) -- The probabilities for each of the labels.
        """
        outputs = super().__call__(sequences, candidate_labels, hypothesis_template)
        if self.onnx:
            outputs = outputs[0]
        num_sequences = 1 if isinstance(sequences, str) else len(sequences)
        candidate_labels = self._args_parser._parse_labels(candidate_labels)
        reshaped_outputs = outputs.reshape((num_sequences, len(candidate_labels), -1))

        if len(candidate_labels) == 1:
            multi_class = True

        if not multi_class:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., -1]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)
        else:
            # softmax over the entailment vs. contradiction dim for each label independently
            entail_contr_logits = reshaped_outputs[..., [0, -1]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]

        result = []
        for iseq in range(num_sequences):
            top_inds = list(reversed(scores[iseq].argsort()))
            result.append(
                {
                    "sequence": sequences if isinstance(sequences, str) else sequences[iseq],
                    "labels": [candidate_labels[i] for i in top_inds],
                    "scores": scores[iseq][top_inds].tolist(),
                }
            )

        if len(result) == 1:
            return result[0]
        return result


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        ignore_labels (:obj:`List[str]`, defaults to :obj:`["O"]`):
            A list of labels to ignore.
        grouped_entities (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to group the tokens corresponding to the same entity together in the predictions or not.
    """,
)
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the
    `named entity recognition examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=token-classification>`__.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        config: PretrainedConfig,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        onnx: bool = True,
        graph_path: Optional[str] = None,
        ignore_labels=["O"],
        task: str = "",
        grouped_entities: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
            config=config,
            onnx=onnx,
            graph_path=graph_path,
        )

        # self.check_model_type(
        #     TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        #     if self.framework == "tf"
        #     else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        # )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities

    def __call__(self, *args, **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with
            :obj:`grouped_entities=True`) with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word.
            - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
              corresponding token in the sentence.
        """
        inputs = self._args_parser(*args, **kwargs)
        answers = []
        for sentence in inputs:
            if self.onnx:
                tokens = self.tokenizer(
                    sentence,
                    return_attention_mask=True,
                    return_tensors=self.framework,
                    truncation=True,
                )
                entities = self._forward_onnx(tokens)[0]
                entities = entities.squeeze(0)
                input_ids = tokens["input_ids"][0]
            else:
                tokens = self.tokenizer(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    truncation=True,
                )
                # Manage correct placement of the tensors
                with self.device_placement():
                    # Forward
                    if self.framework == "tf":
                        entities = self.model(tokens.data)[0][0].numpy()
                        input_ids = tokens["input_ids"].numpy()[0]
                    else:
                        with torch.no_grad():
                            tokens = self.ensure_tensor_on_device(**tokens)
                            entities = self.model(**tokens)[0][0].cpu().numpy()
                            input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            entities = []
            # Filter to labels not in `self.ignore_labels`
            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if self.config.id2label[label_idx] not in self.ignore_labels
            ]

            for idx, label_idx in filtered_labels_idx:

                entity = {
                    "word": self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                    "score": score[idx][label_idx].item(),
                    "entity": self.config.id2label[label_idx],
                    "index": idx,
                }

                entities += [entity]

            # Append grouped entities
            if self.grouped_entities:
                answers += [self.group_entities(entities)]
            # Append ungrouped entities
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"]
        scores = np.mean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:
            is_last_idx = entity["index"] == last_idx
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            if (
                entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ):
                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups


NerPipeline = TokenClassificationPipeline


class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped
    to internal :class:`~transformers.SquadExample`.

    QuestionAnsweringArgumentHandler manages all the possible to create a :class:`~transformers.SquadExample` from
    the command-line supplied arguments.
    """

    def __call__(self, *args, **kwargs):
        # Position args, handling is sensibly the same as X and data, so forwarding to avoid duplicating
        if args is not None and len(args) > 0:
            if len(args) == 1:
                kwargs["X"] = args[0]
            else:
                kwargs["X"] = list(args)

        # Generic compatibility with sklearn and Keras
        # Batched data
        if "X" in kwargs or "data" in kwargs:
            inputs = kwargs["X"] if "X" in kwargs else kwargs["data"]

            if isinstance(inputs, dict):
                inputs = [inputs]
            else:
                # Copy to avoid overriding arguments
                inputs = [i for i in inputs]

            for i, item in enumerate(inputs):
                if isinstance(item, dict):
                    if any(k not in item for k in ["question", "context"]):
                        raise KeyError("You need to provide a dictionary with keys {question:..., context:...}")

                    inputs[i] = QuestionAnsweringPipeline.create_sample(**item)

                elif not isinstance(item, SquadExample):
                    raise ValueError(
                        "{} argument needs to be of type (list[SquadExample | dict], SquadExample, dict)".format(
                            "X" if "X" in kwargs else "data"
                        )
                    )

            # Tabular input
        elif "question" in kwargs and "context" in kwargs:
            if isinstance(kwargs["question"], str):
                kwargs["question"] = [kwargs["question"]]

            if isinstance(kwargs["context"], str):
                kwargs["context"] = [kwargs["context"]]

            inputs = [
                QuestionAnsweringPipeline.create_sample(q, c) for q, c in zip(kwargs["question"], kwargs["context"])
            ]
        else:
            raise ValueError("Unknown arguments {}".format(kwargs))

        if not isinstance(inputs, list):
            inputs = [inputs]

        return inputs


@add_end_docstrings(PIPELINE_INIT_ARGS)
class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using any :obj:`ModelForQuestionAnswering`. See the
    `question answering examples <../task_summary.html#question-answering>`__ for more information.

    This question answering pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task.
    See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=question-answering>`__.
    """

    default_input_names = "question,context"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        device: int = -1,
        task: str = "",
        **kwargs
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=QuestionAnsweringArgumentHandler(),
            device=device,
            task=task,
            **kwargs,
        )

        # self.check_model_type(
        #     TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING if self.framework == "tf" else MODEL_FOR_QUESTION_ANSWERING_MAPPING
        # )

    @staticmethod
    def create_sample(
        question: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally.
        This helper method encapsulate all the logic for converting question(s) and context(s) to
        :class:`~transformers.SquadExample`.

        We currently support extractive question answering.

        Arguments:
            question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
            context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of :class:`~transformers.SquadExample`: The corresponding
            :class:`~transformers.SquadExample` grouping question and context.
        """
        if isinstance(question, list):
            return [SquadExample(None, q, c, None, None, None) for q, c in zip(question, context)]
        else:
            return SquadExample(None, question, context, None, None, None)

    def __call__(self, *args, **kwargs):
        """
        Answer the question(s) given as inputs by using the context(s).

        Args:
            args (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`):
                One or several :class:`~transformers.SquadExample` containing the question and context.
            X (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context
                (will be treated the same way as if passed as the first positional argument).
            data (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context
                (will be treated the same way as if passed as the first positional argument).
            question (:obj:`str` or :obj:`List[str]`):
                One or several question(s) (must be used in conjunction with the :obj:`context` argument).
            context (:obj:`str` or :obj:`List[str]`):
                One or several context(s) associated with the qustion(s) (must be used in conjunction with the
                :obj:`question` argument).
            topk (:obj:`int`, `optional`, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood).
            doc_stride (:obj:`int`, `optional`, defaults to 128):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            max_answer_len (:obj:`int`, `optional`, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (:obj:`int`, `optional`, defaults to 384):
                The maximum length of the total sentence (context + question) after tokenization. The context will be
                split in several chunks (using :obj:`doc_stride`) if needed.
            max_question_len (:obj:`int`, `optional`, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not we accept impossible as an answer.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the
            following keys:

            - **score** (:obj:`float`) -- The probability associated to the answer.
            - **start** (:obj:`int`) -- The start index of the answer (in the tokenized version of the input).
            - **end** (:obj:`int`) -- The end index of the answer (in the tokenized version of the input).
            - **answer** (:obj:`str`) -- The answer to the question.
        """
        # Set defaults values
        kwargs.setdefault("topk", 1)
        kwargs.setdefault("doc_stride", 128)
        kwargs.setdefault("max_answer_len", 15)
        kwargs.setdefault("max_seq_len", 384)
        kwargs.setdefault("max_question_len", 64)
        kwargs.setdefault("handle_impossible_answer", False)

        if kwargs["topk"] < 1:
            raise ValueError("topk parameter should be >= 1 (got {})".format(kwargs["topk"]))

        if kwargs["max_answer_len"] < 1:
            raise ValueError("max_answer_len parameter should be >= 1 (got {})".format(kwargs["max_answer_len"]))

        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        features_list = [
            squad_convert_examples_to_features(
                examples=[example],
                tokenizer=self.tokenizer,
                max_seq_length=kwargs["max_seq_len"],
                doc_stride=kwargs["doc_stride"],
                max_query_length=kwargs["max_question_len"],
                padding_strategy=PaddingStrategy.DO_NOT_PAD.value,
                is_training=False,
                tqdm_enabled=False,
            )
            for example in examples
        ]
        all_answers = []
        for features, example in zip(features_list, examples):
            model_input_names = self.tokenizer.model_input_names + ["input_ids"]
            fw_args = {k: [feature.__dict__[k] for feature in features] for k in model_input_names}

            # Manage tensor allocation on correct device
            if self.onnx:
                # fw_args = {k: torch.tensor(v) for (k, v) in fw_args.items()}
                fw_args = {k: np.array(v) for (k, v) in fw_args.items()}
                start, end = self._forward_onnx(fw_args)[:2]
            else:
                with self.device_placement():
                    if self.framework == "tf":
                        fw_args = {k: tf.constant(v) for (k, v) in fw_args.items()}
                        start, end = self.model(fw_args)[:2]
                        start, end = start.numpy(), end.numpy()
                    else:
                        with torch.no_grad():
                            # Retrieve the score for the context tokens only (removing question tokens)
                            fw_args = {k: torch.tensor(v, device=self.device) for (k, v) in fw_args.items()}
                            start, end = self.model(**fw_args)[:2]
                            start, end = start.cpu().numpy(), end.cpu().numpy()

            min_null_score = 1000000  # large and positive
            answers = []
            for (feature, start_, end_) in zip(features, start, end):
                # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
                undesired_tokens = np.abs(np.array(feature.p_mask) - 1) & feature.attention_mask

                # Generate mask
                undesired_tokens_mask = undesired_tokens == 0.0

                # Make sure non-context indexes in the tensor cannot contribute to the softmax
                start_ = np.where(undesired_tokens_mask, -10000.0, start_)
                end_ = np.where(undesired_tokens_mask, -10000.0, end_)

                # Normalize logits and spans to retrieve the answer
                start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
                end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

                if kwargs["handle_impossible_answer"]:
                    min_null_score = min(min_null_score, (start_[0] * end_[0]).item())

                # Mask CLS
                start_[0] = end_[0] = 0.0

                starts, ends, scores = self.decode(start_, end_, kwargs["topk"], kwargs["max_answer_len"])
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                answers += [
                    {
                        "score": score.item(),
                        "start": np.where(char_to_word == feature.token_to_orig_map[s])[0][0].item(),
                        "end": np.where(char_to_word == feature.token_to_orig_map[e])[0][-1].item(),
                        "answer": " ".join(
                            example.doc_tokens[feature.token_to_orig_map[s] : feature.token_to_orig_map[e] + 1]
                        ),
                    }
                    for s, e, score in zip(starts, ends, scores)
                ]

            if kwargs["handle_impossible_answer"]:
                answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

            answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: kwargs["topk"]]
            all_answers += answers

        if len(all_answers) == 1:
            return all_answers[0]
        return all_answers

    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and will generate probalities for each span to be
        the actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than
        max_answer_len or answer end position being before the starting position.
        The method supports output the k-best answer through the topk argument.

        Args:
            start (:obj:`np.ndarray`): Individual start probabilities for each token.
            end (:obj:`np.ndarray`): Individual end probabilities for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end]

    def span_to_answer(self, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
        """
        When decoding from token probalities, this method maps token indexes to actual word in
        the initial context.

        Args:
            text (:obj:`str`): The actual context to extract the answer from.
            start (:obj:`int`): The answer starting token index.
            end (:obj:`int`): The answer end token index.

        Returns:
            Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }


# Register all the supported tasks here
SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "tf": TFAutoModel if is_tf_available() else None,
        "pt": AutoModel if is_torch_available() else None,
        "default": {"model": {"pt": "distilbert-base-cased", "tf": "distilbert-base-cased"}},
    },
    "sentiment-analysis": {
        "impl": TextClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
                "tf": "distilbert-base-uncased-finetuned-sst-2-english",
            },
        },
    },
    "ner": {
        "impl": TokenClassificationPipeline,
        "tf": TFAutoModelForTokenClassification if is_tf_available() else None,
        "pt": AutoModelForTokenClassification if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tf": "dbmdz/bert-large-cased-finetuned-conll03-english",
            },
        },
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "tf": TFAutoModelForQuestionAnswering if is_tf_available() else None,
        "pt": AutoModelForQuestionAnswering if is_torch_available() else None,
        "default": {
            "model": {"pt": "distilbert-base-cased-distilled-squad", "tf": "distilbert-base-cased-distilled-squad"},
        },
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "tf": TFAutoModelForSequenceClassification if is_tf_available() else None,
        "pt": AutoModelForSequenceClassification if is_torch_available() else None,
        "default": {
            "model": {"pt": "roberta-large-mnli", "tf": "roberta-large-mnli"},
            "config": {"pt": "roberta-large-mnli", "tf": "roberta-large-mnli"},
            "tokenizer": {"pt": "roberta-large-mnli", "tf": "roberta-large-mnli"},
        },
    },
}


def pipeline(
    task: str,
    model: Optional[str] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    framework: Optional[str] = None,
    onnx: bool = True,
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`.
            - :obj:`"sentiment-analysis"`: will return a :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"ner"`: will return a :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`.
            - :obj:`"zero-shot-classification"`: will return a :class:`~transformers.ZeroShotClassificationPipeline`.
        model (:obj:`str`, `optional`):
            The model that will be used by the pipeline to make predictions. This should be a model identifier

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default for the :obj:`task` will be loaded.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from
            :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default for the :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified
            and both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no
            model is provided.
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        # Sentiment analysis pipeline
        pipeline('sentiment-analysis')

        # Question answering pipeline, specifying the checkpoint identifier
        pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    framework = framework or get_framework(model)

    targeted_task = SUPPORTED_TASKS[task]
    task_class, model_class = targeted_task["impl"], targeted_task[framework]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"][framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        elif isinstance(config, str):
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )

    modelcard = None
    # Try to infer modelcard from model or config name (if provided as str)
    if isinstance(model, str):
        modelcard = model
    elif isinstance(config, str):
        modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config
    if config is not None and isinstance(config, str):
        config = AutoConfig.from_pretrained(config)
    elif config is None:
        config = AutoConfig.from_pretrained(model)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard)

    # Instantiate model if needed
    graph_name = f"{os.path.basename(model)}.onnx"
    graph_path = ONNX_CACHE_DIR.joinpath(model, graph_name)

    # TODO: assert when model is not `str`
    # Instantiate the model if graph is not found or if doing normal inference
    if (onnx and not os.path.exists(graph_path)) or not onnx:
        # Handle transparent TF/PT model conversion
        model_kwargs = {}
        if framework == "pt" and model.endswith(".h5"):
            model_kwargs["from_tf"] = True
            logger.warning(
                "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                "Trying to load the model with PyTorch."
            )
        elif framework == "tf" and model.endswith(".bin"):
            model_kwargs["from_pt"] = True
            logger.warning(
                "Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. "
                "Trying to load the model with Tensorflow."
            )
        model = model_class.from_pretrained(model, config=config, **model_kwargs)

    return task_class(
        model=model,
        tokenizer=tokenizer,
        modelcard=modelcard,
        framework=framework,
        task=task,
        onnx=onnx,
        graph_path=graph_path,
        config=config,
        **kwargs,
    )
