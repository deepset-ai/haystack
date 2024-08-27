# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from numpy import mean as np_mean

from haystack import component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, expit
from haystack.utils.auth import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install \"sentence-transformers>=3.0.0\"'") as sas_import:
    from sentence_transformers import CrossEncoder, SentenceTransformer, util
    from transformers import AutoConfig


@component
class SASEvaluator:
    """
    SASEvaluator computes the Semantic Answer Similarity (SAS) between a list of predictions and a one of ground truths.

    It's usually used in Retrieval Augmented Generation (RAG) pipelines to evaluate the quality of the generated
    answers. The SAS is computed using a pre-trained model from the Hugging Face model hub. The model can be either a
    Bi-Encoder or a Cross-Encoder. The choice of the model is based on the `model` parameter.

    Usage example:
    ```python
    from haystack.components.evaluators.sas_evaluator import SASEvaluator

    evaluator = SASEvaluator(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    evaluator.warm_up()
    ground_truths = [
        "A construction budget of US $2.3 billion",
        "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
    ]
    predictions = [
        "A construction budget of US $2.3 billion",
        "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
        "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
    ]
    result = evaluator.run(
        ground_truths_answers=ground_truths, predicted_answers=predictions
    )

    print(result["score"])
    # 0.9999673763910929

    print(result["individual_scores"])
    # [0.9999765157699585, 0.999968409538269, 0.9999572038650513]
    ```
    """

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        batch_size: int = 32,
        device: Optional[ComponentDevice] = None,
        token: Secret = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
    ):
        """
        Creates a new instance of SASEvaluator.

        :param model:
            SentenceTransformers semantic textual similarity model, should be path or string pointing to a downloadable
            model.
        :param batch_size:
            Number of prediction-label pairs to encode at once.
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically selected.
        :param token:
            The Hugging Face token for HTTP bearer authorization.
            You can find your HF token in your [account settings](https://huggingface.co/settings/tokens)
        """
        sas_import.check()

        self._model = model
        self._batch_size = batch_size
        self._device = device
        self._token = token
        self._similarity_model = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self._model,
            batch_size=self._batch_size,
            device=self._device.to_dict() if self._device else None,
            token=self._token.to_dict() if self._token else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SASEvaluator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        if device := data.get("init_parameters", {}).get("device"):
            data["init_parameters"]["device"] = ComponentDevice.from_dict(device)
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._similarity_model:
            return

        token = self._token.resolve_value() if self._token else None
        config = AutoConfig.from_pretrained(self._model, use_auth_token=token)
        cross_encoder_used = False
        if config.architectures:
            cross_encoder_used = any(arch.endswith("ForSequenceClassification") for arch in config.architectures)
        device = ComponentDevice.resolve_device(self._device).to_torch_str()
        # Based on the Model string we can load either Bi-Encoders or Cross Encoders.
        # Similarity computation changes for both approaches
        if cross_encoder_used:
            self._similarity_model = CrossEncoder(
                self._model,
                device=device,
                tokenizer_args={"use_auth_token": token},
                automodel_args={"use_auth_token": token},
            )
        else:
            self._similarity_model = SentenceTransformer(self._model, device=device, use_auth_token=token)

    @component.output_types(score=float, individual_scores=List[float])
    def run(self, ground_truth_answers: List[str], predicted_answers: List[str]) -> Dict[str, Any]:
        """
        SASEvaluator component run method.

        Run the SASEvaluator to compute the Semantic Answer Similarity (SAS) between a list of predicted answers
        and a list of ground truth answers. Both must be list of strings of same length.

        :param ground_truth_answers:
            A list of expected answers for each question.
        :param predicted_answers:
            A list of generated answers for each question.
        :returns:
            A dictionary with the following outputs:
                - `score`: Mean SAS score over all the predictions/ground-truth pairs.
                - `individual_scores`: A list of similarity scores for each prediction/ground-truth pair.
        """
        if len(ground_truth_answers) != len(predicted_answers):
            raise ValueError("The number of predictions and labels must be the same.")

        if any(answer is None for answer in predicted_answers):
            raise ValueError("Predicted answers must not contain None values.")

        if len(predicted_answers) == 0:
            return {"score": 0.0, "individual_scores": [0.0]}

        if not self._similarity_model:
            msg = "The model has not been initialized. Call warm_up() before running the evaluator."
            raise RuntimeError(msg)

        if isinstance(self._similarity_model, CrossEncoder):
            # For Cross Encoders we create a list of pairs of predictions and labels
            sentence_pairs = list(zip(predicted_answers, ground_truth_answers))
            similarity_scores = self._similarity_model.predict(
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
            predictions_embeddings = self._similarity_model.encode(
                predicted_answers, batch_size=self._batch_size, convert_to_tensor=True
            )
            label_embeddings = self._similarity_model.encode(
                ground_truth_answers, batch_size=self._batch_size, convert_to_tensor=True
            )

            # Compute cosine-similarities
            similarity_scores = [
                float(util.cos_sim(p, l).cpu().numpy()) for p, l in zip(predictions_embeddings, label_embeddings)
            ]

        sas_score = np_mean(similarity_scores)

        return {"score": sas_score, "individual_scores": similarity_scores}
