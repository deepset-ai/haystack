from typing import Any, Dict, List, Optional

from numpy import mean as np_mean

from haystack import component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, expit
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from .preprocess import _preprocess_text

with LazyImport(message="Run 'pip install scikit-learn \"sentence-transformers>=2.2.0\"'") as metrics_import:
    from sentence_transformers import CrossEncoder, SentenceTransformer, util
    from transformers import AutoConfig


@component
class SASEvaluator:
    """
    SASEvaluator computes the Semantic Answer Similarity (SAS) between a list of predictions and a list of labels.
    It's usually used in Retrieval Augmented Generation (RAG) pipelines to evaluate the quality of the generated answers.

    The SAS is computed using a pre-trained model from the Hugging Face model hub. The model can be either a
    Bi-Encoder or a Cross-Encoder. The choice of the model is based on the `model` parameter.
    The default model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
    """

    def __init__(
        self,
        labels: List[str],
        model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        regexes_to_ignore: Optional[List[str]] = None,
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        ignore_numbers: bool = False,
        batch_size: int = 32,
        device: Optional[ComponentDevice] = None,
        token: Secret = Secret.from_env_var("HF_API_TOKEN", strict=False),
    ):
        """
        Creates a new instance of SASEvaluator.

        :param labels: The list of expected answers.
        :param model: SentenceTransformers semantic textual similarity model, should be path or string pointing to
            a downloadable model.
        :param regexes_to_ignore: A list of regular expressions. If provided, it removes substrings
            matching these regular expressions from both predictions and labels before comparison. Defaults to None.
        :param ignore_case: If True, performs case-insensitive comparison. Defaults to False.
        :param ignore_punctuation: If True, removes punctuation from both predictions and labels before
            comparison. Defaults to False.
        :param ignore_numbers: If True, removes numerical digits from both predictions and labels
            before comparison. Defaults to False.
        :param batch_size: Number of prediction-label pairs to encode at once.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected.
        :param token: The Hugging Face token for HTTP bearer authorization.
            You can find your HF token at https://huggingface.co/settings/tokens.
        """
        metrics_import.check()

        self._labels = labels
        self._model = model
        self._regexes_to_ignore = regexes_to_ignore
        self._ignore_case = ignore_case
        self._ignore_punctuation = ignore_punctuation
        self._ignore_numbers = ignore_numbers
        self._batch_size = batch_size
        self._device = device
        self._token = token

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            labels=self._labels,
            regexes_to_ignore=self._regexes_to_ignore,
            ignore_case=self._ignore_case,
            ignore_punctuation=self._ignore_punctuation,
            ignore_numbers=self._ignore_numbers,
            model=self._model,
            batch_size=self._batch_size,
            device=self._device.to_dict() if self._device else None,
            token=self._token.to_dict() if self._token else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SASEvaluator":
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        if device := data.get("init_parameters", {}).get("device"):
            data["init_parameters"]["device"] = ComponentDevice.from_dict(device)
        return default_from_dict(cls, data)

    @component.output_types(sas=float, scores=List[float])
    def run(self, predictions: List[str]) -> Dict[str, Any]:
        if len(predictions) != len(self._labels):
            raise ValueError("The number of predictions and labels must be the same.")

        if len(predictions) == 0:
            return {"sas": 0.0, "scores": [0.0]}

        token = self._token.resolve_value() if self._token else None

        predictions = _preprocess_text(
            predictions, self._regexes_to_ignore, self._ignore_case, self._ignore_punctuation, self._ignore_numbers
        )
        labels = _preprocess_text(
            self._labels, self._regexes_to_ignore, self._ignore_case, self._ignore_punctuation, self._ignore_numbers
        )
        config = AutoConfig.from_pretrained(self._model, use_auth_token=token)
        cross_encoder_used = False
        if config.architectures:
            cross_encoder_used = any(arch.endswith("ForSequenceClassification") for arch in config.architectures)

        device = ComponentDevice.resolve_device(self._device)

        # Based on the Model string we can load either Bi-Encoders or Cross Encoders.
        # Similarity computation changes for both approaches

        if cross_encoder_used:
            # For Cross Encoders we create a list of pairs of predictions and labels
            similarity_model = CrossEncoder(
                self._model,
                device=device.to_torch_str(),
                tokenizer_args={"use_auth_token": token},
                automodel_args={"use_auth_token": token},
            )
            sentence_pairs = [[pred, label] for pred, label in zip(predictions, labels)]
            similarity_scores = similarity_model.predict(
                sentence_pairs, batch_size=self._batch_size, convert_to_numpy=True
            )

            # All Cross Encoders do not return a set of logits scores that are normalized
            # We normalize scores if they are larger than 1
            if (similarity_scores > 1).any():
                similarity_scores = expit(similarity_scores)

            # Convert scores to list of floats from numpy array
            similarity_scores = similarity_scores.tolist()

        else:
            # For Bi-encoders we create embeddings separately for predictions and labels
            similarity_model = SentenceTransformer(self._model, device=device.to_torch_str(), use_auth_token=token)
            predictions_embeddings = similarity_model.encode(
                predictions, batch_size=self._batch_size, convert_to_tensor=True
            )
            label_embeddings = similarity_model.encode(labels, batch_size=self._batch_size, convert_to_tensor=True)

            # Compute cosine-similarities
            scores = util.cos_sim(predictions_embeddings, label_embeddings)

            # cos_sim computes cosine similarity between all pairs of vectors in pred_embeddings and label_embeddings
            # It returns a matrix with shape (len(predictions), len(labels))
            similarity_scores = [scores[i][i].item() for i in range(len(predictions))]

        sas_score = np_mean(similarity_scores)

        return {"sas": sas_score, "scores": similarity_scores}
