from typing import Dict, Optional, List, Union

import random
import logging
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizer

from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.data_handler.samples import Sample, SampleBasket
from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


class _EvaluationMixin:
    pass


class _TrainingMixin:
    pass


# Copied from TableTextSimilarityProcessor


class MultiModalDatasetProcessor(_EvaluationMixin, _TrainingMixin):  # was inheriting from Processor, review
    """
    Used to handle the Multimodal Retrieval datasets consisting of documents
    that come in different formats.
    """

    def __init__(
        self,
        query_feature_extractor: PreTrainedTokenizer,
        passage_feature_extractors: Dict[ContentTypes, PreTrainedTokenizer],
    ):
        """
        :param query_feature_extractor: Used to split a query into features
        :param passage_feature_extractors: Used to split a document's content into features
        """
        self.query_feature_extractor = query_feature_extractor
        self.passage_feature_extractors = passage_feature_extractors

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset.
        For conversion we have an internal representation called "baskets".
        Each basket is one query and related text passages (positive passages fitting to the query and negative
        passages that do not fit the query)
        Each stage adds or transforms specific information to our baskets.

        :param dicts: List of dicts, input dictionary with DPR-style content
                        {"query": str,
                         "passages": List[
                                        {'title': str,
                                        'text': str,
                                        'label': 'hard_negative',
                                        'external_id': str},
                                        ....
                                        ]
                         }
        :param indices: list, indices used during multiprocessing so that IDs assigned to our baskets is unique
        :param return_baskets: boolean, whether to return the baskets or not (baskets are needed during inference)
        """
        if indices is None:
            indices = range(len(dicts))
        baskets = [
            SampleBasket(id_external=None, id_internal=id_internal, raw=d) for d, id_internal in zip(dicts, indices)
        ]

        baskets = self._convert_queries(baskets=baskets)
        baskets = self._convert_contexts(baskets=baskets)
        # Convert features into pytorch dataset, this step also removes and logs potential errors during preprocessing
        dataset, tensor_names, problematic_ids, baskets = self._create_dataset(baskets)

        if problematic_ids:
            logger.error(
                f"There were {len(problematic_ids)} errors during preprocessing at positions: {problematic_ids}"
            )

        if return_baskets:
            return dataset, tensor_names, problematic_ids, baskets
        return dataset, tensor_names, problematic_ids

    def _convert_queries(self, baskets: List[SampleBasket]):
        # FIXME this assumes the query is text. Fix later.
        for basket in baskets:
            clear_text = {}
            tokenized = {}
            features: List[Dict] = [{}]
            # extract query, positive context passages and titles, hard-negative passages and titles
            if "query" in basket.raw:
                try:
                    query = self._normalize_question(basket.raw["query"])

                    # featurize the query
                    query_inputs = self.query_tokenizer.encode_plus(
                        text=query,
                        max_length=self.max_seq_len_query,
                        add_special_tokens=True,
                        truncation=True,
                        truncation_strategy="longest_first",
                        padding="max_length",
                        return_token_type_ids=True,
                    )

                    # tokenize query
                    tokenized_query = self.query_tokenizer.convert_ids_to_tokens(query_inputs["input_ids"])

                    if len(tokenized_query) == 0:
                        logger.warning(
                            f"The query could not be tokenized, likely because it contains a character that the query tokenizer does not recognize"
                        )
                        return None

                    clear_text["query_text"] = query
                    tokenized["query_tokens"] = tokenized_query
                    features[0]["query_input_ids"] = query_inputs["input_ids"]
                    features[0]["query_segment_ids"] = query_inputs["token_type_ids"]
                    features[0]["query_attention_mask"] = query_inputs["attention_mask"]
                except Exception as e:
                    features = None

            sample = Sample(id="", clear_text=clear_text, tokenized=tokenized, features=features)
            basket.samples = [sample]
        return baskets

    def _convert_contexts(self, baskets: List[SampleBasket]):
        """
        Converts context by content type.
        """
        for basket in baskets:
            if "passages" in basket.raw:
                try:
                    contexts_data = {"positive": [], "hard_negative": []}
                    content_types = []

                    positive_context = [x for x in basket.raw["passages"] if x["label"] == "positive"]
                    if self.shuffle_positives:
                        positive_context = random.sample(positive_context, self.num_positives)
                    else:
                        positive_context = positive_context[: self.num_positives]

                    hard_negative_context = [x for x in basket.raw["passages"] if x["label"] == "hard_negative"]
                    if self.shuffle_negatives:
                        hard_negative_context = random.sample(hard_negative_context, self.num_hard_negatives)
                    else:
                        hard_negative_context = hard_negative_context[: self.num_hard_negatives]

                    for name, context in [("positive", positive_context), ("hard_negative", hard_negative_context)]:
                        for ctx in context:

                            meta = " ".join(ctx.get("meta") if ctx.get("meta") else "")
                            if ctx["type"] == "text":
                                data = (meta, ctx["text"]) if self.embed_meta_fields else ctx["text"]

                            elif ctx["type"] == "table":
                                data = [cell for row in ctx["rows"] for cell in row]
                                data = " ".join(ctx["columns"] + data)
                                data = (meta, data) if self.embed_meta_fields else data

                            elif ctx["type"] == "image":
                                data = ctx["image"]
                            else:
                                raise NotImplementedError(f"FIXME: Not implemented yet: {ctx['type']}")

                            contexts_data[name].append(data)
                            content_types.append(ctx["type"])

                    # all context passages and labels: 1 for positive context and 0 for hard-negative context
                    ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
                    # concatenate title with positive context passages + negative context passages
                    all_ctx = contexts_data["positive"] + contexts_data["hard_negative"]

                    # assign empty string tuples if hard_negative passages less than num_hard_negatives
                    # all_ctx += [("", "")] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

                    # Create the input vectors using the proper tokenizer for each content_type
                    inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": []}

                    for content_type, passage_tokenizer in self.passage_tokenizers.items():
                        selected_contexts = [
                            ctx for ctx, cnt_type in zip(all_ctx, content_types) if cnt_type == content_type
                        ]
                        if selected_contexts:
                            content_type_inputs = passage_tokenizer.extract_features(
                                selected_contexts,
                                # add_special_tokens=True,
                                # truncation=True,
                                # padding="max_length",
                                # max_length=self.max_seq_len_passages[content_type],
                                # return_token_type_ids=True,
                            )
                            for tensor_name, tensor in content_type_inputs.items():
                                if not tensor_name in inputs.keys():
                                    inputs[tensor_name] = []
                                inputs[tensor_name].append(tensor)

                    for tensor_name, list_of_tensors in inputs.items():
                        if list_of_tensors:
                            inputs[tensor_name] = np.stack(list_of_tensors, axis=1)

                    input_ids = inputs.get("input_ids")
                    passage_segment_ids = inputs.get("token_type_ids")
                    attention_mask = inputs.get("attention_mask")
                    pixel_values = inputs.get("pixel_values")

                    # get tokens in string format
                    tokenized = [self.passage_tokenizer.convert_ids_to_tokens(ctx) for ctx in input_ids]

                    # for DPR we only have one sample containing query and corresponding (multiple) context features
                    sample = basket.samples[0]
                    sample.clear_text["passages"] = positive_context + hard_negative_context
                    sample.tokenized["passages_tokens"] = tokenized
                    sample.features[0]["passage_input_ids"] = input_ids
                    sample.features[0]["passage_segment_ids"] = passage_segment_ids
                    sample.features[0]["table_segment_ids"] = passage_segment_ids
                    sample.features[0]["passage_attention_mask"] = attention_mask
                    sample.features[0]["pixel_values"] = pixel_values
                    sample.features[0]["label_ids"] = ctx_label
                    sample.features[0]["content_types"] = content_types
                except Exception as e:
                    logger.exception(e)
                    basket.samples[0].features = None

        return baskets

    def _create_dataset(self, baskets: List[SampleBasket]):
        """
        Convert python features into pytorch dataset.
        Also removes potential errors during preprocessing.
        Flattens nested basket structure to create a flat list of features
        """
        features_flat: List = []
        basket_to_remove = []
        problematic_ids = set()
        for basket in baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        if len(basket_to_remove) > 0:
            for basket in basket_to_remove:
                # if basket_to_remove is not empty remove the related baskets
                problematic_ids.add(basket.id_internal)
                baskets.remove(basket)

        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names, problematic_ids, baskets

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Removes '?' from queries/questions"""
        if question[-1] == "?":
            question = question[:-1]
        return question

    @staticmethod
    def _combine_meta_context(meta_fields: List[str], texts: List[str]):
        return [("" if meta is None else meta, ctx) for meta, ctx in zip(meta_fields, texts)]
