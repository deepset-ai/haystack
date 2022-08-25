from typing import List, Union, Dict, Optional, Tuple

from collections import defaultdict
import logging
import itertools
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers.pre_tokenizers import WhitespaceSplit

from transformers import AutoTokenizer, AutoModelForTokenClassification, PretrainedConfig
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
    :param use_gpu: Whether to use the GPU or not.
    :param progress_bar: Whether to show a progress bar or not.
    :param batch_size: The batch size to use for entity extraction.
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    :param aggregation_strategy: The strategy to fuse (or not) tokens based on the model prediction.
        “none” : Will simply not do any aggregation and simply return raw results from the model
        “simple” : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{“word”: ABC, “entity”: “TAG”}, {“word”: “D”, “entity”: “TAG2”}, {“word”: “E”, “entity”: “TAG2”}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{“word”: “Micro”, “entity”: “ENTERPRISE”}, {“word”: “soft”, “entity”: “NAME”}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, “New york” might still be tagged with two different entities.
        “first” : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity.
        “average” : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. scores will be averaged first across tokens, and then the maximum label is applied.
        “max” : (works only on word based models) Will use the SIMPLE strategy except that words, cannot end up with different tags. Word entity will simply be the token with the maximum score.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "dslim/bert-base-NER",
        use_gpu: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        aggregation_strategy: str = "simple",
    ):
        super().__init__()

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)
        self.batch_size = batch_size
        self.progress_bar = progress_bar

        # TODO Consider adding AutoConfig. When is it needed?
        # config = AutoConfig.from_pretrained(
        #     model_name_or_path,
        #     num_labels=num_labels,
        #     finetuning_task=task_name,
        #     cache_dir=cache_dir,
        #     revision=model_revision,
        #     use_auth_token=use_auth_token,
        # )
        # TODO Consider using add_prefix_space=True. Do this if you do not want the first word to be treated
        #      differently.
        #      Explained in more detail here:
        #      https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaTokenizer
        # if config.model_type in {"bloom", "gpt2", "roberta"}:
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         model_name_or_path,
        #         use_auth_token=use_auth_token,
        #         add_prefix_space=True,
        #     )
        # else:
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         model_name_or_path,
        #         use_auth_token=use_auth_token,
        #     )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        self.model = token_classifier = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            use_auth_token=use_auth_token,
            # config=config,
        )
        self.model.to(str(self.devices[0]))
        self.extractor_pipeline = pipeline(
            "ner",
            model=token_classifier,
            tokenizer=tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=0 if self.devices[0].type == "cuda" else -1,
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

    def run_batch(self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None):  # type: ignore
        if isinstance(documents[0], Document):
            flattened_documents = documents
        else:
            flattened_documents = list(itertools.chain.from_iterable(documents))  # type: ignore

        if batch_size is None:
            batch_size = self.batch_size

        docs = [doc.content for doc in flattened_documents if isinstance(doc, Document)]
        all_entities = self.extract_batch(docs, batch_size=batch_size)

        for entities_per_doc, doc in zip(all_entities, flattened_documents):
            if not isinstance(doc, Document):
                raise HaystackError(f"doc was of type {type(doc)}, but expected a Document.")
            doc.meta["entities"] = entities_per_doc
        output = {"documents": documents}

        return output, "output_1"

    def extract(self, text):
        """
        This function can be called to perform entity extraction when using the node in isolation.
        """
        entities = self.extractor_pipeline(text)
        return entities

    def extract_batch(self, texts: Union[List[str], List[List[str]]], batch_size: Optional[int] = None):
        """
        This function allows to extract entities out of a list of strings or a list of lists of strings.

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
            right_idx = 0
            for number in number_of_texts:
                right_idx = left_idx + number
                grouped_entities.append(entities[left_idx:right_idx])
                left_idx = right_idx
            return grouped_entities

    def train(self):
        # Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

        # Model has labels -> use them.
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

        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if data_args.pad_to_max_length else False

        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

        if do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    tokenize_and_align_labels,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

        if do_predict:
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    tokenize_and_align_labels,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

        # Metrics
        metric = evaluate.load("seqeval")

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            trainer.save_model()  # Saves the tokenizer too for easy upload

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate()

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # Predict
        if training_args.do_predict:
            logger.info("*** Predict ***")

            predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            # Save predictions
            output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
            if trainer.is_world_process_zero():
                with open(output_predictions_file, "w") as writer:
                    for prediction in true_predictions:
                        writer.write(" ".join(prediction) + "\n")

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(self, examples, b_to_i_label, padding):
        # Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

        tokenized_inputs = self.tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
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
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
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
    def __init__(self, samples: List, labels: Optional[List] = None, create_tensors: bool = False):
        self.samples = samples
        self.create_tensors = create_tensors
        self.labels = labels

    def __getitem__(self, item):
        if not self.create_tensors:
            return self.samples[item]
        else:
            sample = self.samples[item]
            inputs = {
                "input_ids": torch.tensor(sample.ids, dtype=torch.long),
                "attention_mask": torch.tensor(sample.attention_mask, dtype=torch.long),
            }

            if self.labels:
                inputs["labels"] = torch.tensor(self.labels[item], dtype=torch.long)

            return inputs

    def __len__(self):
        try:
            return len(self.samples.encodings)
        except AttributeError:
            return len(self.samples)


class TokenClassificationNode:
    def __init__(
        self, model_name_or_path: str, label_to_label_id: dict, label_filter_mapping: dict, use_gpu: bool = True
    ):
        """
        :param model_name_or_path: Directory of a saved model or the name of a public model
        :param label_to_label_id: dictionary converting label name to label id
        :param label_filter_mapping: dictionary that can be used filter label names
        :param use_gpu:
        """
        self.label_to_label_id = label_to_label_id
        self.label_id_to_label = {label_id: label for label, label_id in label_to_label_id.items()}
        self.label_filter_mapping = label_filter_mapping
        self.seen_labels = defaultdict(int)

        self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=False)

        self.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, add_prefix_space=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, num_labels=len(label_to_label_id)
        )
        self.model.to(str(self.devices[0]))

    # @staticmethod
    # def _convert_char_positions_to_word_positions(texts: List[str], labels: List[List[dict]]):
    #     """Convert character positions to word positions
    #
    #     :param texts: list of text samples
    #     :param labels: list of labels that contain the name of the entity and the character start and end positions
    #                    where the original name of the entity appears in the text. Note that the original name of the
    #                    entity can have a different length than the length of the name.
    #
    #     Example Input:
    #         labels = [{
    #             'name': "GVUL",
    #             'start': 774
    #             'end': 802
    #         }, ...]
    #
    #     Example Output:
    #         labels = [{
    #             'name': "GVUL",
    #             'word_indices': [33, 34, 35, 36]
    #         }, ...]
    #     """
    #     converted_labels = []
    #     for text, text_labels in zip(texts, labels):
    #         converted_text_labels = []
    #         for label in text_labels:
    #             before = text[:label['start']]
    #             span = text[label['start']:label['end']]
    #             after = text[label['end']:]
    #
    #             # We only want to split at the beginning of a word
    #             # So in case the label does not start at the beginning, we walk left until we hit a space
    #             if len(before) > 0 and not span[0].isspace():
    #                 b_space_idx = -1
    #                 c = before[b_space_idx]
    #                 while not c.isspace() and label['start'] + b_space_idx > 0:
    #                     c = before[b_space_idx]
    #                     b_space_idx -= 1
    #
    #                 before = before[:b_space_idx]
    #                 span = text[label['start'] + b_space_idx:label['end']]
    #
    #             # # We also walk right until we hit a space because
    #             if len(after) > 0 and not span[-1].isspace():
    #                 a_space_idx = 0
    #                 c = after[a_space_idx]
    #                 while not c.isspace() and label['start'] + a_space_idx < len(after):
    #                     a_space_idx += 1
    #                     c = after[a_space_idx]
    #
    #                 span = span + after[:a_space_idx]
    #
    #             n_before_words = len(before.split())
    #             span_words = span.split()
    #             word_indices = [idx + n_before_words for idx in range(len(span_words))]
    #             word_indices = [idx for idx in word_indices if idx < len(text.split())]
    #             converted_text_labels.append({
    #                 'name': label['name'],
    #                 'word_indices': word_indices
    #             })
    #         converted_labels.append(converted_text_labels)
    #
    #     return converted_labels

    # @staticmethod
    # def fill_gaps(texts: List[str], labels: List[List[dict]]):
    #     """Create a list of labels for each item in `texts`. All words not present in the list of dictionaries,
    #     `labels`, will be assigned the label "O". All words present in the list of dictionaries, `labels`,
    #     will be assigned the name stored in the dictionary.
    #
    #     :param texts: list of original texts
    #     :param labels: list of labels that contain the name and the word indices at which that name appears in the text
    #
    #     Example:
    #         labels = [
    #             [{
    #                 'name': "GVUL",
    #                 'word_indices': [33, 34, 35, 36]
    #             }, ...],
    #             ...
    #         ]
    #     """
    #     filled_labels = []
    #     for text, labels in zip(texts, labels):
    #         local_labels = ['O' for word in text.split()]
    #         for label in labels:
    #             for idx, word_idx in enumerate(label['word_indices']):
    #                 if idx == 0:
    #                     label_id = 'B-' + label['name']
    #                 else:
    #                     label_id = 'I-' + label['name']
    #
    #                 try:
    #                     local_labels[word_idx] = label_id
    #                 except:
    #                     print(f'Label out of bounds: {label}')
    #
    #         # local_labels is a list that converts word_idx to token label
    #         filled_labels.append(local_labels)
    #
    #     return filled_labels

    # def report(self, texts: List[str], eval_labels: List[List[str]], pred_labels: List[List[str]]):
    #     report = classification_report(eval_labels, pred_labels)
    #     report_dict = classification_report(eval_labels, pred_labels, output_dict=True)
    #
    #     texts_split = [t.split() for t in texts]
    #     filtered_test = self.remove_seen_labels(texts_split, eval_labels)
    #     filtered_train = self.remove_seen_labels(texts_split, pred_labels)
    #
    #     report_filtered = classification_report(filtered_test, filtered_train)
    #
    #     print('### Evaluation Report ###')
    #     print(report)
    #
    #     print(' ')
    #     print('### Unseen Entity Evaluation Report ###')
    #     print(report_filtered)
    #
    #     return report_dict

    # def load_and_preprocess_doccano_file(self, doccano_file: str):
    #     eval_data = load_jsonl(doccano_file)
    #     texts, labels = self.preprocess_doccano_data(eval_data)
    #     converted_labels = self._convert_char_positions_to_word_positions(texts=texts, labels=labels)
    #     converted_labels = self.fill_gaps(texts, converted_labels)
    #     return texts, converted_labels

    # def evaluate(self, eval_file: str):
    #     """Evaluate the TokenClassification model on *eval_file*
    #
    #     :param eval_file: file path to evaluation file in jsonl format
    #     """
    #     texts, converted_labels = self.load_and_preprocess_doccano_file(eval_file)
    #     preds = self.predict(texts)
    #     preds = [[self.label_id_to_label[label_id] for label_id in labels] for labels in preds]
    #
    #     report_dict = self.report(texts, eval_labels=converted_labels, pred_labels=preds)
    #     return report_dict
    #
    # def eval_from_files(self, eval_file: str, prediction_file: str):
    #     """Compare the predictions of two files. For example, this can be used to calculate inter-annotator agreement
    #     between two annotators using two doccano files.
    #
    #     :param eval_file: file path to the doccano file that will be used as the ground truth
    #     :param prediction_file: file path to doccano file that will be evaluated
    #     """
    #     texts, eval_labels = self.load_and_preprocess_doccano_file(eval_file)
    #     texts, pred_labels = self.load_and_preprocess_doccano_file(prediction_file)
    #     report_dict = self.report(texts, eval_labels=eval_labels, pred_labels=pred_labels)
    #     return report_dict

    def tokenize(self, samples: List[List[str]], max_len: int = 512, stride: int = 20):
        """Tokenize the input *samples*

        :param samples: List of text samples
        :param max_len:
        :param stride:
        """
        return self.tokenizer(
            samples,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
            padding="max_length",
            add_special_tokens=True,
            stride=stride,
            truncation=True,
            max_length=max_len,
        )

    # @staticmethod
    # def align_labels_to_tokenized_texts(tokenized_texts, token_labels: List[List[str]]):
    #     """Align the labels to how tokenized_texts was split. Also, only label the first token of a given word and
    #     assign -100 to other subtokens from the same word.
    #
    #     :param tokenized_texts: transformers.tokenization_utils_base.BatchEncoding
    #     :param token_labels: list of lists of token labels ordered by word index
    #     """
    #     labels = []
    #     for mapped_idx, sample_idx in enumerate(tokenized_texts['overflow_to_sample_mapping']):
    #         word_ids = tokenized_texts.word_ids(mapped_idx)
    #         current_labels = {word_idx: label for word_idx, label in enumerate(token_labels[sample_idx])}
    #         label_ids = []
    #         previous_word_id = None
    #         for word_id in word_ids:  # Set the special tokens to -100 so PyTorch loss function will ignore them.
    #             if word_id is None:
    #                 label_ids.append(-100)
    #             elif word_id != previous_word_id:  # Only label the first token of a given word.
    #                 label_ids.append(current_labels[word_id])
    #             else:
    #                 label_ids.append(-100)
    #             previous_word_id = word_id
    #         labels.append(label_ids)
    #
    #     return labels

    # def preprocess_doccano_data(self, doccano_data: List[dict]):
    #     """Preprocess the data from doccano
    #
    #     :param doccano_data: list of samples in doccano format
    #
    #     Example
    #     {
    #         "text": ""
    #         "label": [[4, 15, "GVUL"], [266, 277, "GVUL"]],
    #         "uid": ""
    #     }
    #     """
    #     texts = []
    #     labels = []
    #     for sample in doccano_data:
    #         texts.append(sample['text'])
    #         text_labels = [
    #             {
    #                 'start': l[0],
    #                 'end': l[1],
    #                 'name': self.label_filter_mapping.get(l[2])
    #             }
    #             for l in sample['label'] if self.label_filter_mapping.get(l[2])
    #         ]
    #         labels.append(text_labels)
    #     return texts, labels

    # def tokenize_from_doccano(
    #     self,
    #     doccano_file: str,
    #     max_len: int,
    #     stride: int
    # ):
    #     """Tokenize the text and extract the labels from doccano data
    #
    #     :param doccano_file: file path to the doccano file
    #     :param max_len: max length of the tokenizer
    #     :param stride: stride of the tokenizer
    #     """
    #     texts, converted_labels = self.load_and_preprocess_doccano_file(doccano_file)
    #
    #     self.set_seen_labels(texts, converted_labels)
    #     texts_with_char_positions = [
    #         self.pre_tokenizer.pre_tokenize_str(t) for t in texts
    #     ]
    #     texts = [[label[0] for label in text] for text in texts_with_char_positions]
    #
    #     tokenized_texts = self.tokenize(texts, max_len=max_len, stride=stride)
    #     converted_labels = self.align_labels_to_tokenized_texts(tokenized_texts, converted_labels)
    #     converted_labels = [
    #         [self.label_to_label_id[l] if l != -100 else -100 for l in labels]
    #         for labels in converted_labels
    #     ]
    #
    #     return tokenized_texts, converted_labels

    # def train(
    #     self,
    #     train_file: str,
    #     epochs: int = 1,
    #     lr: float = 1e-05,
    #     max_len: int = 384,
    #     stride: int = 20,
    #     batch_size: int = 16,
    #     model_save_dir: Optional[str] = None
    # ):
    #     train_samples, train_labels = self.tokenize_from_doccano(
    #         doccano_file=train_file,
    #         max_len=max_len,
    #         stride=stride
    #     )
    #
    #     filtered_train_samples = []
    #     filtered_train_labels = []
    #     for text, labels in zip(train_samples.encodings, train_labels):
    #         if max(labels) > 0:
    #             filtered_train_samples.append(text)
    #             filtered_train_labels.append(labels)
    #
    #     train_dataset = TokenClassificationDataset(
    #         samples=filtered_train_samples, labels=filtered_train_labels, create_tensors=True
    #     )
    #
    #     data_collator = DefaultDataCollator()
    #     self.model.train()
    #     training_args = TrainingArguments(
    #         output_dir=model_save_dir,
    #         evaluation_strategy="no",
    #         learning_rate=lr,
    #         per_device_train_batch_size=batch_size,
    #         per_device_eval_batch_size=batch_size * 2,
    #         num_train_epochs=epochs,
    #         weight_decay=0.01,
    #         save_strategy='no'
    #     )
    #
    #     trainer = Trainer(
    #         model=self.model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         tokenizer=self.tokenizer,
    #         data_collator=data_collator,
    #     )
    #
    #     trainer.train()
    #
    #     self.model.save_pretrained(model_save_dir)
    #     self.tokenizer.save_pretrained(model_save_dir)

    def predict(
        self,
        texts: Union[str, List[str]],
        max_len: int = 384,
        stride: int = 20,
        batch_size: int = 64,
        output_raw_predictions: bool = False,
        return_text_spans: bool = False,
    ):
        """Token Classification Prediction. Predict the most likely token for each word in *texts*.

        :param texts: input text to perform token classification on. Can be a string or a List of strings.
        :param max_len: max length for the tokenizer
        :param stride: stride for the tokenizer
        :param batch_size: batch size of the DataLoader
        :param output_raw_predictions: If *True* return the individual classification scores (logits) of each class.
            Otherwise, only the most probable token is returned for each word.
        :param return_text_spans:
        """

        self.model.eval()

        input_texts = copy.deepcopy(texts)
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        texts_with_char_positions = [
            self.pre_tokenizer.pre_tokenize_str(t) if isinstance(t, str) else t for t in input_texts
        ]
        input_texts = [[label[0] for label in text] for text in texts_with_char_positions]

        tokenized_samples = self.tokenize(input_texts, max_len=max_len, stride=stride)

        dataset = TokenClassificationDataset(tokenized_samples, create_tensors=True)

        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True)

        predictions = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.devices[0])
            attention_mask = batch["attention_mask"].to(self.devices[0])

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.append(outputs)

        reshaped_predictions = self._reshape_predictions(predictions)

        raw_predictions = self._postprocess_predictions(reshaped_predictions, tokenized_samples)

        if output_raw_predictions:
            return raw_predictions

        # Return most likely label for each word in each doc
        argmax_predictions = [[np.argmax(word_preds) for word_preds in text] for text in raw_predictions]

        if not return_text_spans:
            return argmax_predictions

        # Convert label ids (ints) to text labels
        preds = [[self.label_id_to_label[label_id] for label_id in label_ids] for label_ids in argmax_predictions]
        return self.convert_predictions_to_text_spans(texts_with_char_positions, preds)

    @staticmethod
    def _postprocess_predictions(predictions: List[np.ndarray], tokenized_texts):
        """Aggregate each of the items of `predictions` based on which text document they originally came from.

        :param predictions: List of predictions. Each item in the list should have shape max_length x num_labels
        :param tokenized_texts: Output from self.tokenize. The key 'overflow_to_sample_mapping' is being used to
                                determine which item of `predictions` belongs to each original document.
                                The method 'word_ids' is being used to determine how the tokens match to individual
                                words.
        """

        # overflow_to_sample_mapping tells me which documents need be aggregated
        # e.g. tokenized_texts['overflow_to_sample_mapping'] = [0, 0, 1, 1, 1, 1] means first two elements of
        # predictions belong to document 0 and the other four elements belong to document 1.
        sample_mapping = tokenized_texts["overflow_to_sample_mapping"]

        recovered_labels = []
        current_sample = {}
        prev_sample = 0

        for mapped_idx, sample_idx in enumerate(sample_mapping):
            # word_ids is the token idx to word id (e.g. [None, 0, 0, 1, 1, 1, None]). Nones are special tokens, tokens
            # 1 and 2 correspond to word_id=1, and tokens 3, 4, and 5 correspond to word_id=2.
            word_ids = tokenized_texts.word_ids(mapped_idx)
            current_preds = predictions[mapped_idx]

            for idx, word_id in enumerate(word_ids):
                # word_id is None for start and end tokens
                if word_id is None:
                    continue

                if prev_sample == sample_idx:
                    existing_pred = current_sample.get(word_id, None)

                    # We only use the prediction from the first token for a given word since only the first token is
                    # assigned the true label. See more details in the align_labels_to_tokenized_texts method.
                    if existing_pred is not None:
                        continue

                    current_sample[word_id] = current_preds[idx]
                else:
                    recovered_labels.append(current_sample)
                    current_sample = {word_id: current_preds[idx]}
                    prev_sample = sample_idx
        # Make sure to grab the last current_sample
        recovered_labels.append(current_sample)

        out_labels = []
        for labels in recovered_labels:
            flattened = sorted(labels.items(), key=lambda label: label[0])
            flattened = [label for _, label in flattened]
            out_labels.append(flattened)

        return out_labels

    @staticmethod
    def _reshape_predictions(predictions: List):
        """Send the predictions to the cpu and reshape to remove the batch dimension.

        :param predictions: list of TokenClassifierOutput predictions
        """
        final_predictions = []
        # predictions has length num_batches
        for pred in predictions:
            # pred.logits has shape num_splits x max_length x num_labels
            pred = pred.logits.detach().cpu().numpy()

            # This converts pred from a tensor into a list of tensors.
            # List has length num_splits
            # Tensors have shape 1 x max_length x num_labels
            pred_list = np.vsplit(pred, pred.shape[0])
            assert len(pred_list) == pred.shape[0]

            # This is removing the first dimension so each tensor now has shape max_length x num_labels
            pred_list = [np.hstack(p) for p in pred_list]

            # Combines all predictions into a single list
            final_predictions.extend(pred_list)
        return final_predictions

    @staticmethod
    def convert_predictions_to_text_spans(texts_with_char_positions: List[List[tuple]], predictions: List[List[str]]):
        """Convert predictions output from `self.predict` into labels with original text spans. The output has shape
        number of docs by number of words in each doc. Additionally, this function implements the logic for how to
        combine 'B-' and 'I-' entities.

        Example:
            - For the prediction `["B-ATTACKER", "I-ATTACKER"]` the two words will be combined into one entity.
            - For the prediction `["B-DEFENDER", "O", "I-DEFENDER"]` only the word labeled "B-DEFENDER" will be
            considered as the entity and the word labeled as "I-DEFENDER" will be ignored since they are separated by
            the "O" label.
            - For the prediction `["O", "O", "I-VICTIM", "O", "O"]` no entities will be returned since the word
            labeled as "I-VICTIM" does not have a "B-VICTIM" immediately before it.

        :param texts_with_char_positions: List of outputs from `self.pre_tokenizer.pre_tokenize_str`
        :param predictions: predictions providing the most likely label name for each word in a list of docs
                            (has shape # of docs x # of words in doc)
        """
        assert len(texts_with_char_positions) == len(predictions)

        all_labels_w_positions = []
        for text_idx, text in enumerate(texts_with_char_positions):
            doc_labels_w_positions = []
            for word_idx, word_tuple in enumerate(text):
                word_label = predictions[text_idx][word_idx]

                if "B-" in word_label:
                    final_word_label = word_label[2:]  # Remove 'B-' prefix.
                    start = word_tuple[1][0]
                    end = word_tuple[1][1]
                    # Check for all intermediate entities (I-) and combine together.
                    for i in range(word_idx + 1, len(predictions[text_idx])):
                        next_word_label = predictions[text_idx][i]
                        next_word_tuple = text[i]
                        if "I-" in next_word_label and word_label[2:] == next_word_label[2:]:
                            end = next_word_tuple[1][1]
                        else:
                            break
                # Skip all non-relevant words
                elif "O" in word_label:
                    continue
                # Skip all intermediate (I-) entities. They are checked for in the 'B-' branch.
                elif "I-" in word_label:
                    continue
                else:
                    raise ValueError(f"Unexpected label {word_label} in predictions")

                doc_labels_w_positions.append((start, end, final_word_label))
            all_labels_w_positions.append(doc_labels_w_positions)

        return all_labels_w_positions
