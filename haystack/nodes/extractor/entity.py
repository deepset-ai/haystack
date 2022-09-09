import logging
from typing import List, Union, Dict, Optional, Tuple, Any

import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import evaluate
from datasets import load_dataset, ClassLabel, DatasetDict
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    PretrainedConfig,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    set_seed,
)
from transformers import pipeline
from tqdm.auto import tqdm
from haystack.errors import HaystackError
from haystack.schema import Document
from haystack.nodes.base import BaseComponent
from haystack.modeling.utils import initialize_device_settings
from haystack.utils.torch_utils import ListDataset

logger = logging.getLogger(__name__)


class EntityExtractor(BaseComponent):
    """
    This node is used to extract entities out of documents.
    The most common use case for this would be as a named entity extractor.
    The default model used is dslim/bert-base-NER.
    This node can be placed in a querying pipeline to perform entity extraction on retrieved documents only,
    or it can be placed in an indexing pipeline so that all documents in the document store have extracted entities.
    The entities extracted by this Node will populate Document.entities

    :param model_name_or_path: The name of the model to use for entity extraction.
    :param model_version: The version of the model to use for entity extraction.
    :param use_gpu: Whether to use the GPU or not.
    :param progress_bar: Whether to show a progress bar or not.
    :param batch_size: The batch size to use for entity extraction.
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
    :param aggregation_strategy: The strategy to fuse (or not) tokens based on the model prediction.
        “none”: Will not do any aggregation and simply return raw results from the model.
        “simple”: Will attempt to group entities following the default schema.
                  (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being
                  [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}]
                  Notice that two consecutive B tags will end up as different entities.
                  On word based languages, we might end up splitting words undesirably: Imagine Microsoft being tagged
                  as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}].
                  Look at the options FIRST, MAX, and AVERAGE for ways to mitigate this example and disambiguate words
                  (on languages that support that meaning, which is basically tokens separated by a space).
                  These mitigations will only work on real words, “New york” might still be tagged with two different entities.
        “first”: (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with
                 different tags. Words will simply use the tag of the first token of the word when there is ambiguity.
        “average”: (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with
                   different tags. The scores will be averaged across tokens, and then the label with the maximum score is chosen.
        “max”: (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with
               different tags. Word entity will simply be the token with the maximum score.
    :param add_prefix_space: Do this if you do not want the first word to be treated differently. This is relevant for
        model types such as "bloom", "gpt2", and "roberta".
        Explained in more detail here:
        https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "elastic/distilbert-base-cased-finetuned-conll03-english",
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        aggregation_strategy: str = "first",
        add_prefix_space: bool = None,
    ):
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
            )
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.model_name_or_path = model_name_or_path
        self.use_auth_token = use_auth_token

        if add_prefix_space is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, use_auth_token=use_auth_token, add_prefix_space=add_prefix_space
            )
        self.tokenizer = tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, use_auth_token=use_auth_token, revision=model_version
        )
        self.model.to(str(self.devices[0]))
        # TODO Check that after training the pipeline uses the trained model
        self.extractor_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=self.devices[0],
            use_auth_token=use_auth_token,
        )

    def run(self, documents: Optional[Union[List[Document], List[dict]]] = None) -> Tuple[Dict, str]:  # type: ignore
        """
        This is the method called when this node is used in a pipeline
        """
        if documents:
            for doc in tqdm(documents, disable=not self.progress_bar, desc="Extracting entities"):
                # In a querying pipeline, doc is a haystack.schema.Document object
                try:
                    doc.meta["entities"] = self.extract(doc.content)  # type: ignore
                # In an indexing pipeline, doc is a dictionary
                except AttributeError:
                    doc["meta"]["entities"] = self.extract(doc["content"])  # type: ignore
        output = {"documents": documents}
        return output, "output_1"

    def run_batch(self, documents: Union[List[Document], List[List[Document]], List[dict], List[List[dict]]], batch_size: Optional[int] = None):  # type: ignore
        # TODO check that this works with List of dicts and List of List of dicts
        if isinstance(documents[0], Document) or isinstance(documents[0], dict):
            flattened_documents = documents
        else:
            flattened_documents = list(itertools.chain.from_iterable(documents))  # type: ignore

        if batch_size is None:
            batch_size = self.batch_size

        try:
            docs = [doc.content for doc in flattened_documents if isinstance(doc, Document)]
        except AttributeError:
            docs = [doc["content"] for doc in flattened_documents if isinstance(doc, dict)]
        all_entities = self.extract_batch(docs, batch_size=batch_size)

        for entities_per_doc, doc in zip(all_entities, flattened_documents):
            if not isinstance(doc, Document):
                raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
            try:
                doc.meta["entities"] = entities_per_doc
            except AttributeError:
                doc["meta"]["entities"] = entities_per_doc
        output = {"documents": documents}

        return output, "output_1"

    def _ensure_tensor_on_device(self, inputs: Union[dict, list, tuple, torch.Tensor], device: torch.device):
        if isinstance(inputs, dict):
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def preprocess(self, sentence: Union[str, List[str]], offset_mapping: Optional[torch.Tensor] = None):
        """Preprocessing step to tokenize the provided text.

        :param sentence: Text to tokenize. This works with a list of texts or a single text.
        :param offset_mapping: Only needed if a slow tokenizer is used. Will be used in the postprocessing step to
            determine the original character positions of the detected entities.
        """
        # NOTE: This already can work with List of texts and returns batch size of length list.
        model_inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_overflowing_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        model_inputs["sentence"] = sentence

        return model_inputs

    def forward(self, model_inputs: Dict[str, Any]):
        """Forward step

        :param model_inputs: Dictionary of inputs to be given to the model.
        """
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        overflow_to_sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
        sentence = model_inputs.pop("sentence")

        logits = self.model(**model_inputs)[0]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "sentence": sentence,
            **model_inputs,
        }

    def postprocess(self, model_outputs: Dict[str, Any]):
        """Aggregate each of the items in `model_outputs` based on which text document they originally came from.
        Then we pass the grouped `model_outputs` to `self.extractor_pipeline.postprocess` to take advantage of the
        advanced postprocessing features available in the HuggingFace TokenClassificationPipeline object.

        :param model_outputs: Dictionary of model outputs
        """
        # overflow_to_sample_mapping tells me which documents need be aggregated
        # e.g. model_outputs['overflow_to_sample_mapping'] = [0, 0, 1, 1, 1, 1] means first two elements of
        # predictions belong to document 0 and the other four elements belong to document 1.
        sample_mapping = model_outputs["overflow_to_sample_mapping"]

        # TODO Group according to sample_mapping.
        #  Right now this assumes all of model_outputs corresponds to one document.
        logits = model_outputs["logits"]  # num_splits x model_max_length x num_classes
        input_ids = model_outputs["input_ids"]  # num_splits x model_max_length
        offset_mapping = model_outputs["offset_mapping"]  # num_splits x model_max_length x 2
        special_tokens_mask = model_outputs["special_tokens_mask"]  # num_splits x model_max_length
        sentence = model_outputs["sentence"]  # batch_size x length of text

        logits = torch.reshape(logits, (1, -1, logits.shape[2]))  # 1 x (num_splits * model_max_length) x num_classes
        input_ids = torch.reshape(input_ids, (1, -1))  # 1 x (num_splits * model_max_length)
        offset_mapping = torch.reshape(
            offset_mapping, (1, -1, offset_mapping.shape[2])
        )  # 1 x (num_splits * model_max_length) x num_classes
        special_tokens_mask = torch.reshape(special_tokens_mask, (1, -1))  # 1 x (num_splits * model_max_length)
        sentence = sentence[0]  # Make sure this is a str of the whole doc

        model_outputs_grouped_by_doc = {
            "logits": logits,
            "sentence": sentence,
            "input_ids": input_ids,
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
        }

        results_per_doc = []
        num_docs = sample_mapping[-1].item() + 1
        for i in range(num_docs):
            results_per_doc.append(
                self.extractor_pipeline.postprocess(
                    model_outputs_grouped_by_doc, **self.extractor_pipeline._postprocess_params
                )
            )
        return results_per_doc

    @staticmethod
    def flatten_predictions(predictions: List[Dict[str, Any]]):
        """Flatten the predictions

        :param predictions: List of model output dictionaries
        """
        flattened_predictions = {
            "logits": [],
            "input_ids": [],
            "special_tokens_mask": [],
            "offset_mapping": [],
            "overflow_to_sample_mapping": [],
            "sentence": [],
        }
        for pred in predictions:
            flattened_predictions["logits"].append(pred["logits"])
            flattened_predictions["input_ids"].append(pred["input_ids"])
            flattened_predictions["special_tokens_mask"].append(pred["special_tokens_mask"])
            flattened_predictions["offset_mapping"].append(pred["offset_mapping"])
            flattened_predictions["overflow_to_sample_mapping"].append(pred["overflow_to_sample_mapping"])
            flattened_predictions["sentence"].extend(pred["sentence"])

        flattened_predictions["logits"] = torch.vstack(flattened_predictions["logits"])
        flattened_predictions["input_ids"] = torch.vstack(flattened_predictions["input_ids"])
        flattened_predictions["special_tokens_mask"] = torch.vstack(flattened_predictions["special_tokens_mask"])
        flattened_predictions["offset_mapping"] = torch.vstack(flattened_predictions["offset_mapping"])
        flattened_predictions["overflow_to_sample_mapping"] = torch.vstack(
            flattened_predictions["overflow_to_sample_mapping"]
        )
        return flattened_predictions

    def extract(self, text: Union[str, List[str]], batch_size: int = 1):
        """
        This function can be called to perform entity extraction when using the node in isolation.

        :param text: Text to extract entities from. Can be a str or a List of str.
        :param batch_size:
        """
        if isinstance(text, str):
            text = [text]

        model_inputs = self.preprocess(text)
        dataset = TokenClassificationDataset(model_inputs)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)

        predictions = []
        for batch in tqdm(dataloader, disable=not self.progress_bar, total=len(dataloader), desc="Extracting entities"):
            batch = self._ensure_tensor_on_device(batch, device=self.devices[0])
            with torch.inference_mode():
                model_outputs = self.forward(batch)
            model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            predictions.append(model_outputs)

        flattened_predictions = self.flatten_predictions(predictions)
        return self.postprocess(flattened_predictions)

    def new_extract_batch(self, texts: Union[List[str], List[List[str]]], batch_size: Optional[int] = None):
        """
        This function allows the extraction of entities out of a list of strings or a list of lists of strings.

        :param texts: List of str or list of lists of str to extract entities from.
        :param batch_size: Number of texts to make predictions on at a time.
        """
        # Will have flattened list of texts after this step
        if isinstance(texts[0], str):
            single_list_of_texts = True
            number_of_texts = [len(texts)]
        else:
            single_list_of_texts = False
            number_of_texts = [len(text_list) for text_list in texts]
            texts = list(itertools.chain.from_iterable(texts))

        entities = self.extract(texts, batch_size=batch_size)

        if single_list_of_texts:
            return entities
        else:
            # Group entities together
            grouped_entities = []
            left_idx = 0
            for number in number_of_texts:
                right_idx = left_idx + number
                grouped_entities.append(entities[left_idx:right_idx])
                left_idx = right_idx
            return grouped_entities

    def extract_batch(self, texts: Union[List[str], List[List[str]]], batch_size: Optional[int] = None):
        """
        This function allows the extraction of entities out of a list of strings or a list of lists of strings.

        :param texts: List of str or list of lists of str to extract entities from.
        :param batch_size: Number of texts to make predictions on at a time.
        """
        if isinstance(texts[0], str):
            single_list_of_texts = True
            number_of_texts = [len(texts)]
        else:
            single_list_of_texts = False
            number_of_texts = [len(text_list) for text_list in texts]
            texts = list(itertools.chain.from_iterable(texts))

        # progress bar hack since HF pipeline does not support them
        entities = []
        texts_dataset = ListDataset(texts) if self.progress_bar else texts
        for out in tqdm(
            self.extractor_pipeline(texts_dataset, batch_size=batch_size),
            disable=not self.progress_bar,
            total=len(texts_dataset),
            desc="Extracting entities",
        ):
            entities.append(out)

        if single_list_of_texts:
            return entities
        else:
            # Group entities together
            grouped_entities = []
            left_idx = 0
            for number in number_of_texts:
                right_idx = left_idx + number
                grouped_entities.append(entities[left_idx:right_idx])
                left_idx = right_idx
            return grouped_entities

    def train(
        self,
        do_eval: bool,
        do_test: bool,
        fp16: bool,
        resume_from_checkpoint: str,
        output_dir: str,
        lr: float,
        batch_size: int,
        epochs: int,
        pad_to_max_length: bool = False,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
        preprocessing_num_workers: Optional[int] = None,
        overwrite_cache: bool = False,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        text_column_name: Optional[str] = None,
        label_column_name: Optional[str] = None,
        cache_dir: str = None,
        label_all_tokens: bool = False,
        return_entity_level_metrics: bool = False,
        max_seq_length: int = None,
        task_name: str = "ner",
        push_to_hub: bool = False,
    ):
        """
        Run NER training which was adapted from
        https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

        :param do_eval:
        :param do_test:
        :param fp16:
        :param resume_from_checkpoint:
        :param output_dir:
        :param lr:
        :param batch_size:
        :param epochs:
        :param pad_to_max_length: Whether to pad all samples to model maximum sentence length.
            If False, this function will pad the samples dynamically when batching to the maximum length in the batch.
            This dynamic behavior is more efficient on GPUs but performs very poorly on TPUs.
        :param train_file:
        :param validation_file:
        :param test_file:
        :param preprocessing_num_workers: The number of processes to use for the preprocessing.
        :param overwrite_cache: If True overwrite the cached training, evaluation and test datasets.
        :param dataset_name:
        :param dataset_config_name:
        :param text_column_name:
        :param label_column_name:
        :param cache_dir: Location to store datasets loaded from huggingface
        :param label_all_tokens: Whether to put the label for one word on all tokens of generated by that word or just
            on the one (in which case the other tokens will have a padding index).
        :param return_entity_level_metrics:
        :param max_seq_length: The maximum total input sequence length after tokenization. If set, sequences longer
            than this will be truncated, sequences shorter will be padded.
        :param task_name:
        :param push_to_hub: If True push the model to the HuggingFace model hub.
        """

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps" if do_eval else "no",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="no",
            fp16=fp16,
            resume_from_checkpoint=resume_from_checkpoint,
            push_to_hub=push_to_hub,
        )

        # Set seed before initializing model.
        set_seed(training_args.seed)

        raw_datasets = self.get_raw_datasets(
            dataset_name, dataset_config_name, cache_dir, train_file, validation_file, test_file
        )

        column_names = raw_datasets["train"].column_names

        # Auto determine the column containing the text
        if text_column_name is not None:
            text_column_name = text_column_name
        elif "tokens" in column_names:
            text_column_name = "tokens"
        else:
            text_column_name = column_names[0]

        # Auto determine the column containing the labels
        if label_column_name is not None:
            label_column_name = label_column_name
        elif f"{task_name}_tags" in column_names:
            label_column_name = f"{task_name}_tags"
        else:
            label_column_name = column_names[1]

        # TODO Consider updating model labels outside of ner_processor since it is a bit hidden right now.
        ner_processor = NERDataProcessor(model=self.model, tokenizer=self.tokenizer)
        train_dataset, eval_dataset, test_dataset = ner_processor.preprocess_datasets(
            raw_datasets=raw_datasets,
            training_args=training_args,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            pad_to_max_length=pad_to_max_length,
            max_seq_length=max_seq_length,
            label_all_tokens=label_all_tokens,
            preprocessing_num_workers=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8 if fp16 else None)

        metric = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            if return_entity_level_metrics:
                # Unpack nested dictionaries
                final_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            final_results[f"{key}_{n}"] = v
                    else:
                        final_results[key] = value
                return final_results
            else:
                return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    "accuracy": results["overall_accuracy"],
                }

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Training
        checkpoint = None
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluation
        if do_eval:
            logger.info("*** Final Evaluation ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Test
        if do_test:
            logger.info("*** Final Test ***")
            predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="test")
            metrics["test_samples"] = len(test_dataset)
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

        model_card_kwargs = {"finetuned_from": self.model_name_or_path, "tasks": "token-classification"}
        if dataset_name is not None:
            model_card_kwargs["dataset_tags"] = dataset_name
            if dataset_config_name is not None:
                model_card_kwargs["dataset_args"] = dataset_config_name
                model_card_kwargs["dataset"] = f"{dataset_name} {dataset_config_name}"
            else:
                model_card_kwargs["dataset"] = dataset_name

        if push_to_hub:
            trainer.push_to_hub(**model_card_kwargs)
        else:
            trainer.create_model_card(**model_card_kwargs)

    def get_raw_datasets(
        self,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
    ) -> DatasetDict:
        """Retrieve the datasets. You can either provide your own CSV/JSON/TXT training and evaluation files
        or provide the name of one of the public datasets available on HuggingFace at https://huggingface.co/datasets/.

        For CSV/JSON files, this function will use the column called 'text' or the first column if no column called
        'text' is found.

        :param dataset_name: The name of a dataset available on HuggingFace
        :param dataset_config_name: The name of the dataset configuration file on HuggingFace
        :param cache_dir: The directory to read and write data. This defaults to "~/.cache/huggingface/datasets".
        :param train_file: The path to the file with the training data.
        :param validation_file: The path to the file with the validation data.
        :param test_file: The path to the file with the test data.
        """
        if dataset_name is None and train_file is None:
            raise ValueError("Either `dataset_name` or `train_file` must be provided.")

        if dataset_name and train_file:
            logger.warning(
                "Both `dataset_name` and `train_file` were provided. We will ignore `train_file` and use"
                "`dataset_name`."
            )

        # TODO Check does load_dataset automatically split the text into words
        if dataset_name is not None:
            raw_datasets = load_dataset(
                dataset_name, dataset_config_name, cache_dir=cache_dir, use_auth_token=self.use_auth_token
            )
        else:
            data_files = {}
            if train_file is not None:
                data_files["train"] = train_file
            if validation_file is not None:
                data_files["validation"] = validation_file
            if test_file is not None:
                data_files["test"] = test_file
            extension = train_file.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)

        return raw_datasets


class NERDataProcessor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_labels(self, raw_datasets, features, label_column_name):
        """If the labels are of type ClassLabel, they are already integers, and we have the map stored somewhere.
        Otherwise, we have to get the list of labels manually.

        :param raw_datasets:
        :param features:
        :param label_column_name:
        """

        labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
        if labels_are_int:
            label_list = features[label_column_name].feature.names
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = self.get_unique_label_list(raw_datasets["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}

        # If the model has labels use them.
        num_labels = len(label_list)
        if self.model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
            if list(sorted(self.model.config.label2id.keys())) == list(sorted(label_list)):
                # Reorganize `label_list` to match the ordering of the model.
                if labels_are_int:
                    label_to_id = {i: int(self.model.config.label2id[l]) for i, l in enumerate(label_list)}
                    label_list = [self.model.config.id2label[i] for i in range(num_labels)]
                else:
                    label_list = [self.model.config.id2label[i] for i in range(num_labels)]
                    label_to_id = {l: i for i, l in enumerate(label_list)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(self.model.config.label2id.keys()))}, dataset labels:"
                    f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
                )

        # Set the correspondences label/ID inside the model config
        self.model.config.label2id = {l: i for i, l in enumerate(label_list)}
        self.model.config.id2label = {i: l for i, l in enumerate(label_list)}

        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

        return label_list, label_to_id, b_to_i_label

    def tokenize_and_align_labels(
        self,
        examples,  # needs to come first
        padding,
        max_seq_length,
        text_column_name,
        label_column_name,
        label_to_id,
        label_all_tokens,
        b_to_i_label,
    ):
        """Tokenize all texts and align the labels with them.

        :param examples:
        :param padding:
        :param max_seq_length: The maximum total input sequence length after tokenization. If set, sequences longer
            than this will be truncated, sequences shorter will be padded.
        :param text_column_name: The column name of text to input in the file (a csv or JSON file).
        :param label_column_name: The column name of label to input in the file (a csv or JSON file).
        :param label_to_id:
        :param label_all_tokens:
        :param b_to_i_label:
        """

        tokenized_inputs = self.tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
            # TODO Check this is true when providing your own dataset
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def preprocess_datasets(
        self,
        raw_datasets: DatasetDict,
        training_args: TrainingArguments,
        text_column_name: str,
        label_column_name: str,
        pad_to_max_length: bool = False,
        max_seq_length: Optional[int] = None,
        label_all_tokens: bool = False,
        preprocessing_num_workers: Optional[int] = None,
        overwrite_cache: bool = False,
    ):
        """Preprocess the raw datasets

        :param raw_datasets:
        :param training_args:
        :param text_column_name: The column name of text to input in the file (a csv or JSON file).
        :param label_column_name: The column name of label to input in the file (a csv or JSON file).
        :param pad_to_max_length: Whether to pad all samples to model maximum sentence length.
            If False, this function will pad the samples dynamically when batching to the maximum length in the batch.
            This dynamic behavior is more efficient on GPUs but performs very poorly on TPUs.
        :param max_seq_length: The maximum total input sequence length after tokenization. If set, sequences longer
            than this will be truncated, sequences shorter will be padded.
        :param label_all_tokens: Whether to put the label for one word on all tokens of generated by that word or just
            on the one (in which case the other tokens will have a padding index).
        :param preprocessing_num_workers: The number of processes to use for the preprocessing.
        :param overwrite_cache: If True overwrite the cached training, evaluation and test datasets.
        """

        if "train" not in raw_datasets:
            raise ValueError("Please provide a training dataset perform training.")
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets.get("validation", None)
        test_dataset = raw_datasets.get("test", None)

        label_list, label_to_id, b_to_i_label = self.get_labels(
            raw_datasets=raw_datasets, features=raw_datasets["train"].features, label_column_name=label_column_name
        )

        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if pad_to_max_length else False

        kwargs_tokenize_and_align_labels = {
            "padding": padding,
            "max_seq_length": max_seq_length,
            "text_column_name": text_column_name,
            "label_column_name": label_column_name,
            "label_to_id": label_to_id,
            "label_all_tokens": label_all_tokens,
            "b_to_i_label": b_to_i_label,
        }

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
                desc="Running tokenizer on train dataset",
                fn_kwargs=kwargs_tokenize_and_align_labels,
            )

        if eval_dataset:
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                    fn_kwargs=kwargs_tokenize_and_align_labels,
                )

        if test_dataset:
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                test_dataset = test_dataset.map(
                    self.tokenize_and_align_labels,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    load_from_cache_file=not overwrite_cache,
                    desc="Running tokenizer on test dataset",
                    fn_kwargs=kwargs_tokenize_and_align_labels,
                )

        return train_dataset, eval_dataset, test_dataset

    @staticmethod
    def get_unique_label_list(labels):
        """In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        unique labels.

        :param labels: List of labels in a dataset
        """
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list


def simplify_ner_for_qa(output):
    """
    Returns a simplified version of the output dictionary
    with the following structure:
    [
        {
            answer: { ... }
            entities: [ { ... }, {} ]
        }
    ]
    The entities included are only the ones that overlap with
    the answer itself.
    """
    compact_output = []
    for answer in output["answers"]:

        entities = []
        for entity in answer.meta["entities"]:
            if (
                entity["start"] >= answer.offsets_in_document[0].start
                and entity["end"] <= answer.offsets_in_document[0].end
            ):
                entities.append(entity["word"])

        compact_output.append({"answer": answer.answer, "entities": entities})
    return compact_output


class TokenClassificationDataset(Dataset):
    def __init__(self, model_inputs: transformers.tokenization_utils_base.BatchEncoding):
        self.model_inputs = model_inputs

    def __getitem__(self, item):
        input_ids = self.model_inputs["input_ids"][item]
        attention_mask = self.model_inputs["attention_mask"][item]
        special_tokens_mask = self.model_inputs["special_tokens_mask"][item]
        try:
            offset_mapping = self.model_inputs["offset_mapping"][item]
        except KeyError:
            offset_mapping = None
        overflow_to_sample_mapping = self.model_inputs["overflow_to_sample_mapping"][item]
        sentence = self.model_inputs["sentence"][overflow_to_sample_mapping]
        single_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "overflow_to_sample_mapping": overflow_to_sample_mapping,
            "sentence": sentence,
        }
        return single_input

    def __len__(self):
        return len(self.model_inputs.encodings)
