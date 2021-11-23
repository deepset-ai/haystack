import logging
from typing import Union

from transformers import AutoModelForQuestionAnswering

from haystack.modeling.model import adaptive_model as am
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.prediction_head import QuestionAnsweringHead


logger = logging.getLogger(__name__)


class Converter:

    @staticmethod
    def convert_to_transformers(adaptive_model):
        """
        Convert an adaptive model to huggingface's transformers format. Returns a list containing one model for each
        prediction head.

        :param adaptive_model: Model to convert.
        :type adaptive_model: AdaptiveModel
        :return: List of huggingface transformers models.
        """
        converted_models = []

        # convert model for each prediction head
        for prediction_head in adaptive_model.prediction_heads:
            if len(prediction_head.layer_dims) != 2:
                logger.error(
                    f"Currently conversion only works for PredictionHeads that are a single layer Feed Forward NN with dimensions [LM_output_dim, number_classes].\n"
                    f"            Your PredictionHead has {str(prediction_head.layer_dims)} dimensions."
                )
                continue
            if prediction_head.model_type == "span_classification":
                transformers_model = Converter._convert_to_transformers_qa(adaptive_model, prediction_head)
                converted_models.append(transformers_model)
            else:
                logger.error(f"Haystack -> Transformers conversion is not supported yet for"
                             f" prediction heads of type {prediction_head.model_type}")

        return converted_models

    @staticmethod
    def convert_from_transformers(model_name_or_path, device, revision=None, task_type=None, processor=None,  use_auth_token: Union[bool, str] = None, **kwargs):
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
        :param device: "cpu" or "cuda"
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str
        :param task_type: One of :
                          - 'question_answering'
                          More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :return: AdaptiveModel
        """

        lm = LanguageModel.load(model_name_or_path, revision=revision,use_auth_token=use_auth_token, **kwargs)
        if task_type is None:
            # Infer task type from config
            architecture = lm.model.config.architectures[0]
            if "QuestionAnswering" in architecture:
                task_type = "question_answering"
            else:
                logger.error("Could not infer task type from model config. Please provide task type manually. "
                             "('question_answering' or 'embeddings')")


        if task_type == "question_answering":
            ph = QuestionAnsweringHead.load(model_name_or_path, revision=revision, **kwargs)
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                              lm_output_types="per_token", device=device)
        elif task_type == "embeddings":
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[], embeds_dropout_prob=0.1,
                                              lm_output_types=["per_token", "per_sequence"], device=device)

        if processor:
            adaptive_model.connect_heads_with_processor(processor.tasks)

        return adaptive_model

    @staticmethod
    def _convert_to_transformers_qa(adaptive_model, prediction_head):
        # TODO add more infos to config

        # remove pooling layer
        adaptive_model.language_model.model.pooler = None
        # init model
        transformers_model = AutoModelForQuestionAnswering.from_config(adaptive_model.language_model.model.config)
        # transfer weights for language model + prediction head
        setattr(transformers_model, transformers_model.base_model_prefix, adaptive_model.language_model.model)
        transformers_model.qa_outputs.load_state_dict(
            prediction_head.feed_forward.feed_forward[0].state_dict())

        return transformers_model
