
from typing import Dict, Optional, List, Union

import random
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizer

from haystack.modeling.data_handler.processor import Processor
from haystack.modeling.data_handler.samples import Sample, SampleBasket
from haystack.modeling.data_handler.dataset import convert_features_to_dataset
from haystack.schema import ContentTypes


logger = logging.getLogger(__name__)


# Copied from TableTextSimilarityProcessor

class MultiModalSimilarityProcessor(Processor):
    """
    Used to handle the Multimodal Retrieval datasets consisting of documents
    that come in different formats.
    """

    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizer,
        passage_tokenizers: Dict[ContentTypes, PreTrainedTokenizer],
        max_seq_len_query: int,
        max_seq_len_passages: List[int],
        data_dir: str = "",
        metric: Optional[str] = None,
        train_filename: Optional[Union[Path, str]] = "train.json",
        dev_filename: Optional[Union[Path, str]] = None,
        test_filename: Optional[Union[Path, str]] = "test.json",
        dev_split: float = 0.1,
        proxies: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        embed_meta_fields: List[str] = ["page_title", "section_title", "caption"],
        num_positives: int = 1,
        num_hard_negatives: int = 1,
        shuffle_negatives: bool = True,
        shuffle_positives: bool = False,
        label_list: Optional[List[str]] = None
    ):
        """
        :param query_tokenizer: Used to split a question (str) into tokens
        :param passage_tokenizer: Used to split a text passage (str) into tokens.
        :param table_tokenizer: Used to split a table into tokens
        :param max_seq_len_query: Query samples are truncated after this many tokens.
        :param max_seq_len_passage: Context/Passage Samples are truncated after this many tokens.
        :param max_seq_len_table: Table samples are truncated after this many tokens.
        :param data_dir: The directory in which the train and dev files can be found.
                         If not available the dataset will be loaded automatically
                         if the last directory has the same name as a predefined dataset.
                         These predefined datasets are defined as the keys in the dict DOWNSTREAM_TASK_MAP
        :param metric: Name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :param train_filename: The name of the file containing training data.
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :param test_filename: The name of the file containing the test data.
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None.
        :param proxies: Proxy configuration to allow downloads of remote datasets.
                        Format as in  "requests" library: https://2.python-requests.org//en/latest/user/advanced/#proxies
        :param max_samples: maximum number of samples to use.
        :param embed_meta_fields: List of meta fields to embed in text passages and tables during tensorization.
        :param num_hard_negatives: Maximum number of hard negative context passages in a sample.
        :param num_positives: Maximum number of positive context passages in a sample.
        :param shuffle_negatives: Whether to shuffle all the hard_negative passages before selecting the
                                  num_hard_negative number of passages.
        :param shuffle_positives: Whether to shuffle all the positive passages before selecting the
                                  num_positive number of passages.
        :param label_list: List of labels to predict. Usually ["hard_negative", "positive"].
        :param kwargs: Placeholder for passing generic parameters
        """
        # Custom processor attributes
        self.max_samples = max_samples
        self.query_tokenizer = query_tokenizer
        self.passage_tokenizers = passage_tokenizers
        self.embed_meta_fields = embed_meta_fields
        self.num_hard_negatives = num_hard_negatives
        self.num_positives = num_positives
        self.shuffle_negatives = shuffle_negatives
        self.shuffle_positives = shuffle_positives
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passages = max_seq_len_passages

        super().__init__(
            tokenizer=self.query_tokenizer,
            max_seq_len=0,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric:
            self.add_task(
                name="text_similarity",
                metric=metric,
                label_list=label_list,
                label_name="label",
                task_type="text_similarity",
            )
        else:
            logger.info(
                "Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                "using the default task or add a custom task later via processor.add_task()"
            )

    # @classmethod
    # def load_from_dir(cls, load_dir: str):
    #     """
    #      Overwriting method from parent class to **always** load the TableTextSimilarityProcessor
    #      instead of the specific class stored in the config.

    #     :param load_dir: Directory that contains a 'processor_config.json'
    #     :return: An instance of an TableTextSimilarityProcessor.
    #     """
    #     # read config
    #     processor_config_file = Path(load_dir) / "processor_config.json"
    #     config = json.load(open(processor_config_file))
    #     # init tokenizer
    #     query_tokenizer = get_tokenizer(load_dir, tokenizer_class=config["query_tokenizer"], subfolder="query")
    #     passage_tokenizer = get_tokenizer(load_dir, tokenizer_class=config["passage_tokenizer"], subfolder="passage")
    #     table_tokenizer = get_tokenizer(load_dir, tokenizer_class=config["table_tokenizer"], subfolder="table")

    #     # we have to delete the tokenizer string from config, because we pass it as Object
    #     del config["query_tokenizer"]
    #     del config["passage_tokenizer"]
    #     del config["table_tokenizer"]

    #     processor = cls.load(
    #         query_tokenizer=query_tokenizer,
    #         passage_tokenizer=passage_tokenizer,
    #         table_tokenizer=table_tokenizer,
    #         processor_name="TableTextSimilarityProcessor",
    #         **config,
    #     )
    #     for task_name, task in config["tasks"].items():
    #         processor.add_task(name=task_name, metric=task["metric"], label_list=task["label_list"])

    #     if processor is None:
    #         raise Exception

    #     return processor

    # def save(self, save_dir: Union[str, Path]):
    #     """
    #     Saves the vocabulary to file and also creates a json file containing all the
    #     information needed to load the same processor.

    #     :param save_dir: Directory where the files are to be saved.
    #     """
    #     if isinstance(save_dir, str):
    #         save_dir = Path(save_dir)
    #     os.makedirs(save_dir, exist_ok=True)
    #     config = self.generate_config()
    #     # save tokenizer incl. attributes
    #     config["query_tokenizer"] = self.query_tokenizer.__class__.__name__
    #     config["passage_tokenizer"] = self.passage_tokenizer.__class__.__name__
    #     config["table_tokenizer"] = self.table_tokenizer.__class__.__name__

    #     # Because the fast tokenizers expect a str and not Path
    #     # always convert Path to str here.
    #     self.query_tokenizer.save_pretrained(str(save_dir / "query"))
    #     self.passage_tokenizer.save_pretrained(str(save_dir / "passage"))
    #     self.table_tokenizer.save_pretrained(str(save_dir / "table"))

    #     # save processor
    #     config["processor"] = self.__class__.__name__
    #     output_config_file = Path(save_dir) / "processor_config.json"
    #     with open(output_config_file, "w") as file:
    #         json.dump(config, file)

    def file_to_dicts(self, file: str) -> List[Dict]:
        raise NotImplementedError("FIXME: Not yet")
    #     """
    #     Converts a Multimodal Retrieval data file in json format to a list of dictionaries.

    #     :param file: filename of DPR data in json format
    #             Each sample is a dictionary of format:
    #             {"question": str,
    #             "answers": list of str
    #             "positive_ctxs": list of dictionaries of format
    #                 {'title': str, 'text': str, 'passage_id': str, 'type': 'text', 'source': str}
    #                 or
    #                 {'page_title': str, 'section_title': str, 'caption': str, 'columns': list of str,
    #                  'rows': list of list of str, 'type': 'table', 'source': str}
    #             "hard_negative_ctxs": list of dictionaries of format
    #                 {'title': str, 'text': str, 'passage_id': str, 'type': 'text', 'source': str}
    #                 or
    #                 {'page_title': str, 'section_title': str, 'caption': str, 'columns': list of str,
    #                  'rows': list of list of str, 'type': 'table', 'source': str}
    #             }


    #     Returns:
    #     List of dictionaries: List[dict]
    #         each dictionary:
    #         {"query": str,
    #         "passages": [
    #             {"title": str, "text": str, "label": "positive" / "hard_negative", "type": "text", "external_id": id}
    #             or
    #             {"page_title": str, "section_title": str, "caption": str, "columns": list of str,
    #              "rows": list of list of str, "label": "positive" / "hard_negative", "type": "table", "external_id": id}
    #         ...]}
    #     """
    #     dicts = self._read_multimodal_dpr_json(file, max_samples=self.max_samples)
    #     return dicts

    # def _read_multimodal_dpr_json(self, file: str, max_samples: Optional[int] = None) -> List[Dict]:
    #     """
    #     Reads a Multimodal Retrieval data file in json format and returns a list of dictionaries.

    #     :param file: filename of MMR data in json format

    #     Returns:
    #         list of dictionaries: List[dict]
    #         each dictionary: {
    #                     "query": str -> query_text
    #                     "passages": List[dictionaries] -> [
    #                                 {"text": str, "title": str, "label": "positive" / "hard_negative, "external_id": id},
    #                                 or
    #                                 {"page_title": str, "section_title": str, "caption": str, "columns": list of str,
    #                                  "rows": list of lists of str, "label": "positive" / "hard_negative", "type": "table", "external_id": id}
    #                                 ...]
    #                     }
    #     """
    #     dicts = json.load(open(file))
    #     if max_samples:
    #         dicts = random.sample(dicts, min(max_samples, len(dicts)))

    #     # convert DPR dictionary to standard dictionary
    #     query_json_keys = ["question", "questions", "query"]
    #     positive_context_json_keys = ["positive_condata", "positive_ctxs", "positive_context", "positive_ctx"]
    #     hard_negative_json_keys = [
    #         "hard_negative_condata",
    #         "hard_negative_ctxs",
    #         "hard_negative_context",
    #         "hard_negative_ctx",
    #     ]
    #     standard_dicts = []
    #     for dict in dicts:
    #         sample = {}
    #         docs = []
    #         for key, val in dict.items():
    #             if key in query_json_keys:
    #                 sample["query"] = val
    #             elif key in positive_context_json_keys + hard_negative_json_keys:
    #                 for doc in val:
    #                     if doc["type"] == "table":
    #                         docs.append(
    #                             {
    #                                 "meta": [
    #                                     doc[meta_field] for meta_field in self.embed_meta_fields if meta_field in doc
    #                                 ],
    #                                 "columns": doc.get("columns"),
    #                                 "rows": doc.get("rows"),
    #                                 "label": "positive" if key in positive_context_json_keys else "hard_negative",
    #                                 "type": "table",
    #                             }
    #                         )
    #                     elif doc["type"] == "text":
    #                         docs.append(
    #                             {
    #                                 "meta": [
    #                                     doc[meta_field] for meta_field in self.embed_meta_fields if meta_field in doc
    #                                 ],
    #                                 "text": doc["text"],
    #                                 "label": "positive" if key in positive_context_json_keys else "hard_negative",
    #                                 "type": "text",
    #                             }
    #                         )

    #             sample["passages"] = docs
    #         standard_dicts.append(sample)
    #     return standard_dicts

    def dataset_from_dicts(
        self, dicts: List[Dict], indices: List[int] = [], return_baskets: bool = False, debug: bool = False
    ):
        """
        Convert input dictionaries into a pytorch dataset for TextSimilarity.
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
        baskets = [SampleBasket(id_external=None, id_internal=id_internal, raw=d) for d, id_internal in zip(dicts, indices)]

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
                    contexts_data = {
                        "positive": {"meta": [], "data": []},
                        "hard_negative": {"meta": [], "data": []} ,
                    }
                    content_types = []

                    positive_context = list(filter(lambda x: x["label"] == "positive", basket.raw["passages"]))
                    if self.shuffle_positives:
                        random.shuffle(positive_context)
                    positive_context = positive_context[: self.num_positives]

                    hard_negative_context = list(
                        filter(lambda x: x["label"] == "hard_negative", basket.raw["passages"])
                    )
                    if self.shuffle_negatives:
                        random.shuffle(hard_negative_context)
                    hard_negative_context = hard_negative_context[: self.num_hard_negatives]

                    for name, context in [("positive", positive_context), ("hard_negative", hard_negative_context)]:
                        for ctx in context:
                            if ctx["type"] == "text":
                                contexts_data[name]["meta"].append(" ".join(ctx.get("meta")))
                                contexts_data[name]["data"].append(ctx["text"])
                                content_types.append("text")

                            elif ctx["type"] == "table":
                                contexts_data[name]["meta"].append(" ".join(ctx.get("meta")))
                                linearized_rows = [cell for row in ctx["rows"] for cell in row]
                                linearized_table = " ".join(ctx["columns"] + linearized_rows)
                                contexts_data[name]["data"].append(linearized_table)
                                content_types.append("table")

                            elif ctx["type"] == "image":
                                contexts_data[name]["meta"].append(" ".join(ctx.get("meta")))
                                contexts_data[name]["data"].append(Image.open(ctx["path"]))
                                content_types.append("image")

                            else:
                                raise NotImplementedError(f"FIXME: Not implemented yet: {ctx['type']}")

                    # all context passages and labels: 1 for positive context and 0 for hard-negative context
                    ctx_label = [1] * self.num_positives + [0] * self.num_hard_negatives
                    # featurize context passages
                    if self.embed_meta_fields:
                        # concatenate title with positive context passages + negative context passages
                        all_ctx = self._combine_meta_context(
                            contexts_data["positive"]["meta"], contexts_data["positive"]["data"]
                        ) + self._combine_meta_context(contexts_data["hard_negative"]["meta"], contexts_data["hard_negative"]["data"])
                    else:
                        all_ctx = contexts_data["positive"]["data"] + contexts_data["hard_negative"]["data"]

                    # assign empty string tuples if hard_negative passages less than num_hard_negatives
                    all_ctx += [("", "")] * ((self.num_positives + self.num_hard_negatives) - len(all_ctx))

                    # Create the input vectors using the proper tokenizer for each content_type
                    inputs = {"input_ids": np.array(), "token_type_ids": np.array(), "attention_mask": np.array(), }
                    for content_type, passage_tokenizer in self.passage_tokenizers.items():
                        content_type_inputs = passage_tokenizer.batch_encode_plus(
                            all_ctx[content_types == content_type],
                            add_special_tokens=True,
                            truncation=True,
                            padding="max_length",
                            max_length=self.max_seq_len_passage,
                            return_token_type_ids=True,
                        )
                        inputs = {name: np.stack(all_inputs, new_inputs, axis=1) for (name, all_inputs), new_inputs in zip(inputs.items(), content_type_inputs.values())}

                    input_ids = inputs["input_ids"]
                    passage_segment_ids = inputs["token_type_ids"]
                    attention_mask = inputs["attention_mask"]

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
                    sample.features[0]["label_ids"] = ctx_label
                    sample.features[0]["content_types"] = content_types
                except Exception as e:
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
        res = []
        for meta, ctx in zip(meta_fields, texts):
            if meta is None:
                meta = ""
            res.append(tuple((meta, ctx)))
        return res
