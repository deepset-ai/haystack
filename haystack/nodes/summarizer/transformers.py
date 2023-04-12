import itertools
from typing import List, Optional, Set, Union

import logging

import torch
from tqdm.auto import tqdm
from transformers import pipeline

from haystack.schema import Document
from haystack.nodes.summarizer.base import BaseSummarizer
from haystack.modeling.utils import initialize_device_settings
from haystack.utils.torch_utils import ListDataset

logger = logging.getLogger(__name__)


class TransformersSummarizer(BaseSummarizer):
    """
     Summarizes documents using the Hugging Face's transformers framework.

    You can use any model fine-tuned on a summarization task. For example:
    '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'.
    See the up-to-date list of available models in [Hugging Face Documentattion](https://huggingface.co/models?filter=summarization).

    **Example**

     ```python
     docs = [Document(content="PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions."
            "The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by"
            "the shutoffs which were expected to last through at least midday tomorrow.")]

     # Summarize
     summary = summarizer.predict(
        documents=docs)

     # Show results (List of Documents, containing summary and original content)
     print(summary)

    [
      {
        "content": "PGE stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. ...",
        ...
        "meta": {
                   "summary": "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
                   ...
              },
        ...
      },
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "google/pegasus-xsum",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        max_length: int = 200,
        min_length: int = 5,
        use_gpu: bool = True,
        clean_up_tokenization_spaces: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        """
        Load a summarization model from transformers.
        See the up-to-date list of available models at [Hugging Face](https://huggingface.co/models?filter=summarization).

        :param model_name_or_path: The path to the locally saved model or the name of a public model, for example
                                   'facebook/rag-token-nq', 'facebook/rag-sequence-nq'.
                                   See [Hugging Face](https://huggingface.co/models?filter=summarization) for a full list of available models.
        :param model_version: The version of the model to use from the Hugging Face model hub. Can be a tag name, a branch name, or a commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model).
        :param max_length: Maximum length of the summarized text.
        :param min_length: Minimum length of the summarized text.
        :param use_gpu: Whether to use GPU (if available).
        :param clean_up_tokenization_spaces: Whether or not to clean up the potential extra spaces in the text output.
        :param batch_size: Number of documents to process at a time.
        :param progress_bar: Whether to show a progress bar.
        :param use_auth_token: The API token used to download private models from Hugging Face.
                               If set to `True`, the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) is used.
                               More information at [Hugging Face](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained).
        :param devices: List of torch devices (for example, cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects or strings is supported (for example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). If you specify `use_gpu=False`, the devices
                        parameter is not used and a single CPU device is used for inference.
        """
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        if len(self.devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s} inference, using the first device %s.",
                self.__class__.__name__,
                self.devices[0],
            )

        if tokenizer is None:
            tokenizer = model_name_or_path

        self.summarizer = pipeline(
            task="summarization",
            model=model_name_or_path,
            tokenizer=tokenizer,
            revision=model_version,
            device=self.devices[0],
            use_auth_token=use_auth_token,
        )
        self.max_length = max_length
        self.min_length = min_length
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.print_log: Set[str] = set()
        self.batch_size = batch_size
        self.progress_bar = progress_bar

    def predict(self, documents: List[Document]) -> List[Document]:
        """
        Produce the summarization from the supplied documents.
        The documents can come from the Retriever.

        :param documents: A list of Documents (for example, coming from a Retriever) to summarize individually.
        :return: List of Documents, where Document.meta["summary"] contains the summarization.
        """
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot be greater than max_length")

        if len(documents) == 0:
            raise ValueError("Summarizer needs at least one document to produce a summary.")

        contexts: List[str] = [doc.content for doc in documents]

        encoded_input = self.summarizer.tokenizer(contexts, verbose=False)
        for input_id in encoded_input["input_ids"]:
            tokens_count: int = len(input_id)
            if tokens_count > self.summarizer.tokenizer.model_max_length:
                truncation_warning = (
                    "One or more of your input document texts is longer than the specified "
                    f"maximum sequence length for this summarizer model. "
                    f"Generating summary from first {self.summarizer.tokenizer.model_max_length}"
                    f" tokens."
                )
                if truncation_warning not in self.print_log:
                    logger.warning(truncation_warning)
                    self.print_log.add(truncation_warning)

        summaries = self.summarizer(
            contexts,
            min_length=self.min_length,
            max_length=self.max_length,
            return_text=True,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            truncation=True,
        )

        result: List[Document] = []

        for summary, document in zip(summaries, documents):
            document.meta.update({"summary": summary["summary_text"]})
            result.append(document)

        return result

    def predict_batch(
        self, documents: Union[List[Document], List[List[Document]]], batch_size: Optional[int] = None
    ) -> Union[List[Document], List[List[Document]]]:
        """
        Summarize supplied documents in batches.
        These documents can come from the Retriever.

        :param documents: A single list of documents or a list of lists of documents
                          (for example, coming from a Retriever) to summarize.
        :param batch_size: Number of Documents to process at a time.
        """
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot be greater than max_length")

        if len(documents) == 0 or (
            isinstance(documents[0], list) and all(len(docs) == 0 for docs in documents if isinstance(docs, list))
        ):
            raise ValueError("Summarizer needs at least one document to produce a summary.")

        if batch_size is None:
            batch_size = self.batch_size

        is_doclist_flat = isinstance(documents[0], Document)
        if is_doclist_flat:
            contexts = [doc.content for doc in documents if isinstance(doc, Document)]
        else:
            contexts = [
                [doc.content for doc in docs if isinstance(doc, Document)]
                for docs in documents
                if isinstance(docs, list)
            ]
            number_of_docs = [len(context_group) for context_group in contexts]
            contexts = list(itertools.chain.from_iterable(contexts))

        encoded_input = self.summarizer.tokenizer(contexts, verbose=False)
        for input_id in encoded_input["input_ids"]:
            tokens_count: int = len(input_id)
            if tokens_count > self.summarizer.tokenizer.model_max_length:
                truncation_warning = (
                    "One or more of your input document texts is longer than the specified "
                    f"maximum sequence length for this summarizer model. "
                    f"Generating summary from first {self.summarizer.tokenizer.model_max_length}"
                    f" tokens."
                )
                logger.warning(truncation_warning)
                break

        summaries = []
        # HF pipeline progress bar hack, see https://discuss.huggingface.co/t/progress-bar-for-hf-pipelines/20498/2
        summaries_dataset = ListDataset(contexts)
        for summary_batch in tqdm(
            self.summarizer(
                summaries_dataset,
                min_length=self.min_length,
                max_length=self.max_length,
                return_text=True,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                truncation=True,
                batch_size=batch_size,
            ),
            disable=not self.progress_bar,
            total=len(summaries_dataset),
            desc="Summarizing",
        ):
            summaries.extend(summary_batch)

        if is_doclist_flat:
            flat_result: List[Document] = []
            flat_doc_list: List[Document] = [doc for doc in documents if isinstance(doc, Document)]
            for summary, document in zip(summaries, flat_doc_list):
                document.meta.update({"summary": summary["summary_text"]})
                flat_result.append(document)
            return flat_result
        else:
            nested_result: List[List[Document]] = []
            nested_doc_list: List[List[Document]] = [lst for lst in documents if isinstance(lst, list)]

            # Group summaries together
            grouped_summaries = []
            left_idx = 0
            right_idx = 0
            for number in number_of_docs:
                right_idx = left_idx + number
                grouped_summaries.append(summaries[left_idx:right_idx])
                left_idx = right_idx

            for summary_group, docs_group in zip(grouped_summaries, nested_doc_list):
                cur_summaries = []
                for summary, document in zip(summary_group, docs_group):
                    document.meta.update({"summary": summary["summary_text"]})
                    cur_summaries.append(document)
                nested_result.append(cur_summaries)
            return nested_result
