import logging

from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelWithLMHead, \
    AutoModelForTokenClassification

from farm.modeling import adaptive_model as am
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import BertLMHead, QuestionAnsweringHead, RegressionHead, TextClassificationHead, \
    TokenClassificationHead

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
        if len(adaptive_model.prediction_heads) == 2 and adaptive_model.prediction_heads[
            0].model_type == "language_modelling":
            logger.warning("Currently only the Masked Language Modeling component of the prediction head is converted, "
                           "not the Next Sentence Prediction or Sentence Order Prediction components")

        converted_models = []

        # convert model for each prediction head
        for prediction_head in adaptive_model.prediction_heads:
            if len(prediction_head.layer_dims) != 2:
                logger.error(
                    f"Currently conversion only works for PredictionHeads that are a single layer Feed Forward NN with dimensions [LM_output_dim, number_classes].\n"
                    f"            Your PredictionHead has {str(prediction_head.layer_dims)} dimensions."
                )
                continue

            if prediction_head.model_type in ["text_classification", "regression"]:
                transformers_model = Converter._convert_to_transformers_classification_regression(adaptive_model,
                                                                                                  prediction_head)
                converted_models.append(transformers_model)

            elif prediction_head.model_type == "span_classification":
                transformers_model = Converter._convert_to_transformers_qa(adaptive_model, prediction_head)
                converted_models.append(transformers_model)

            elif prediction_head.model_type == "language_modelling":
                transformers_model = Converter._convert_to_transformers_lm(adaptive_model, prediction_head)
                converted_models.append(transformers_model)

            elif prediction_head.model_type == "token_classification":
                transformers_model = Converter._convert_to_transformers_ner(adaptive_model, prediction_head)
                converted_models.append(transformers_model)

            else:
                logger.error(f"FARM -> Transformers conversion is not supported yet for"
                             f" prediction heads of type {prediction_head.model_type}")

        return converted_models

    @staticmethod
    def convert_from_transformers(model_name_or_path, device, revision=None, task_type=None, processor=None, **kwargs):
        """
        Load a (downstream) model from huggingface's transformers format. Use cases:
         - continue training in FARM (e.g. take a squad QA model and fine-tune on your own data)
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
                          - 'text_classification'
                          - 'embeddings'
                          More tasks coming soon ...
        :param processor: populates prediction head with information coming from tasks
        :type processor: Processor
        :return: AdaptiveModel
        """

        lm = LanguageModel.load(model_name_or_path, revision=revision, **kwargs)
        if task_type is None:
            # Infer task type from config
            architecture = lm.model.config.architectures[0]
            if "MaskedLM" in architecture:
                task_type = "lm"
            elif "QuestionAnswering" in architecture:
                task_type = "question_answering"
            elif "SequenceClassification" in architecture:
                if lm.model.config.num_labels == 1:
                    task_type = "regression"
                else:
                    task_type = "text_classification"
            elif "TokenClassification" in architecture:
                task_type = "ner"
            else:
                logger.error("Could not infer task type from model config. Please provide task type manually. "
                             "('lm', 'question_answering', 'regression', 'text_classification', 'ner' or 'embeddings')")

        if task_type == "lm":
            ph = BertLMHead.load(model_name_or_path, revision=revision, **kwargs)
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                              lm_output_types="per_token", device=device)

        elif task_type == "question_answering":
            ph = QuestionAnsweringHead.load(model_name_or_path, revision=revision, **kwargs)
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                              lm_output_types="per_token", device=device)

        elif task_type == "regression":
            if "roberta" in model_name_or_path:
                # The RobertaClassificationHead has components: input2dense, dropout, tanh, dense2output
                # The tanh function cannot be mapped to current FARM style linear Feed Forward PredictionHeads.
                logger.error("Conversion for Regression with Roberta or XLMRoberta not possible at the moment.")
                raise NotImplementedError
            ph = RegressionHead.load(model_name_or_path, **kwargs)
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                              lm_output_types="per_sequence", device=device)

        elif task_type == "text_classification":
            if "roberta" in model_name_or_path:
                # The RobertaClassificationHead has components: input2dense, dropout, tanh, dense2output
                # The tanh function cannot be mapped to current FARM style linear Feed Forward PredictionHeads.
                logger.error(
                    "Conversion for Text Classification with Roberta or XLMRoberta not possible at the moment.")
                raise NotImplementedError
            ph = TextClassificationHead.load(model_name_or_path, revision=revision, **kwargs)
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                              lm_output_types="per_sequence", device=device)

        elif task_type == "ner":
            ph = TokenClassificationHead.load(model_name_or_path, revision=revision, **kwargs)
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[ph], embeds_dropout_prob=0.1,
                                              lm_output_types="per_token", device=device)

        elif task_type == "embeddings":
            adaptive_model = am.AdaptiveModel(language_model=lm, prediction_heads=[], embeds_dropout_prob=0.1,
                                              lm_output_types=["per_token", "per_sequence"], device=device)

        if processor:
            adaptive_model.connect_heads_with_processor(processor.tasks)

        return adaptive_model

    @staticmethod
    def _convert_to_transformers_classification_regression(adaptive_model, prediction_head):
        if adaptive_model.language_model.model.base_model_prefix == "roberta":
            # Classification Heads in transformers have different architecture across Language Model variants
            # The RobertaClassificationhead has components: input2dense, dropout, tanh, dense2output
            # The tanh function cannot be mapped to current FARM style linear Feed Forward ClassificationHeads.
            # So conversion for this type cannot work. We would need a compatible FARM RobertaClassificationHead
            logger.error("Conversion for Text Classification and Regression with Roberta or XLMRoberta "
                         "not possible at the moment.")

        # add more info to config
        adaptive_model.language_model.model.config.num_labels = prediction_head.num_labels
        adaptive_model.language_model.model.config.id2label = {id: label for id, label in
                                                               enumerate(prediction_head.label_list)}
        adaptive_model.language_model.model.config.label2id = {label: id for id, label in
                                                               enumerate(prediction_head.label_list)}
        adaptive_model.language_model.model.config.finetuning_task = prediction_head.model_type
        adaptive_model.language_model.model.config.language = adaptive_model.language_model.language

        # init model
        transformers_model = AutoModelForSequenceClassification.from_config(adaptive_model.language_model.model.config)
        # transfer weights for language model + prediction head
        setattr(transformers_model, transformers_model.base_model_prefix, adaptive_model.language_model.model)
        transformers_model.classifier.load_state_dict(
            prediction_head.feed_forward.feed_forward[0].state_dict())

        return transformers_model

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

    @staticmethod
    def _convert_to_transformers_lm(adaptive_model, prediction_head):
        # remove pooling layer
        adaptive_model.language_model.model.pooler = None
        # init model
        transformers_model = AutoModelWithLMHead.from_config(adaptive_model.language_model.model.config)
        # transfer weights for language model + prediction head
        setattr(transformers_model, transformers_model.base_model_prefix, adaptive_model.language_model.model)
        # Adding decoder bias (required for conversion to transformers)
        prediction_head.decoder.bias = prediction_head.bias

        ph_state_dict = prediction_head.state_dict()
        ph_state_dict["transform.dense.weight"] = ph_state_dict.pop("dense.weight")
        ph_state_dict["transform.dense.bias"] = ph_state_dict.pop("dense.bias")
        ph_state_dict["transform.LayerNorm.weight"] = ph_state_dict.pop("LayerNorm.weight")
        ph_state_dict["transform.LayerNorm.bias"] = ph_state_dict.pop("LayerNorm.bias")
        transformers_model.cls.predictions.load_state_dict(ph_state_dict)

        return transformers_model

    @staticmethod
    def _convert_to_transformers_ner(adaptive_model, prediction_head):
        # remove pooling layer
        adaptive_model.language_model.model.pooler = None
        # add more info to config
        adaptive_model.language_model.model.config.num_labels = prediction_head.num_labels
        adaptive_model.language_model.model.config.id2label = {id: label for id, label in
                                                               enumerate(prediction_head.label_list)}
        adaptive_model.language_model.model.config.label2id = {label: id for id, label in
                                                               enumerate(prediction_head.label_list)}
        adaptive_model.language_model.model.config.finetuning_task = "token_classification"
        adaptive_model.language_model.model.config.language = adaptive_model.language_model.language

        # init model
        transformers_model = AutoModelForTokenClassification.from_config(adaptive_model.language_model.model.config)
        # transfer weights for language model + prediction head
        setattr(transformers_model, transformers_model.base_model_prefix, adaptive_model.language_model.model)
        transformers_model.classifier.load_state_dict(
            prediction_head.feed_forward.feed_forward[0].state_dict())

        return transformers_model
