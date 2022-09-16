import logging
from typing import List, Union, Dict, Optional, Tuple
import itertools
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification
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
    :param batch_size: The batch size to use for entity extraction.
    :param progress_bar: Whether to show a progress bar or not.
    :param use_auth_token: The API token used to download private models from Huggingface.
                           If this parameter is set to `True`, then the token generated when running
                           `transformers-cli login` (stored in ~/.huggingface) will be used.
                           Additional information can be found here
                           https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
    :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
    """

    outgoing_edges = 1

    def __init__(
        self,
        model_name_or_path: str = "dslim/bert-base-NER",
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 16,
        progress_bar: bool = True,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        super().__init__()

        self.devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        self.batch_size = batch_size
        self.progress_bar = progress_bar

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token=use_auth_token)
        token_classifier = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, use_auth_token=use_auth_token, revision=model_version
        )
        token_classifier.to(str(self.devices[0]))
        self.model = pipeline(
            "ner",
            model=token_classifier,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=self.devices[0],
            use_auth_token=use_auth_token,
        )
        if len(self.devices) > 1:
            logger.warning(
                f"Multiple devices are not supported in {self.__class__.__name__} inference, "
                f"using the first device {self.devices[0]}."
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
        entities = self.model(text)
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
            self.model(texts_dataset, batch_size=batch_size),
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
