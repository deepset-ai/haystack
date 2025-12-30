# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any

import numpy as np

from haystack import Document, component, logging
from haystack.components.embedders.types import DocumentEmbedder
from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


@component
class EmbeddingBasedDocumentSplitter:
    """
    Splits documents based on embedding similarity using cosine distances between sequential sentence groups.

    This component first splits text into sentences, optionally groups them, calculates embeddings for each group,
    and then uses cosine distance between sequential embeddings to determine split points. Any distance above
    the specified percentile is treated as a break point. The component also tracks page numbers based on form feed
    characters (`\f`) in the original document.

    This component is inspired by [5 Levels of Text Splitting](
        https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    ) by Greg Kamradt.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder
    from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter

    # Create a document with content that has a clear topic shift
    doc = Document(
        content="This is a first sentence. This is a second sentence. This is a third sentence. "
        "Completely different topic. The same completely different topic."
    )

    # Initialize the embedder to calculate semantic similarities
    embedder = SentenceTransformersDocumentEmbedder()

    # Configure the splitter with parameters that control splitting behavior
    splitter = EmbeddingBasedDocumentSplitter(
        document_embedder=embedder,
        sentences_per_group=2,      # Group 2 sentences before calculating embeddings
        percentile=0.95,            # Split when cosine distance exceeds 95th percentile
        min_length=50,              # Merge splits shorter than 50 characters
        max_length=1000             # Further split chunks longer than 1000 characters
    )
    splitter.warm_up()
    result = splitter.run(documents=[doc])

    # The result contains a list of Document objects, each representing a semantic chunk
    # Each split document includes metadata: source_id, split_id, and page_number
    print(f"Original document split into {len(result['documents'])} chunks")
    for i, split_doc in enumerate(result['documents']):
        print(f"Chunk {i}: {split_doc.content[:50]}...")
    ```
    """

    def __init__(
        self,
        *,
        document_embedder: DocumentEmbedder,
        sentences_per_group: int = 3,
        percentile: float = 0.95,
        min_length: int = 50,
        max_length: int = 1000,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
    ):
        """
        Initialize EmbeddingBasedDocumentSplitter.

        :param document_embedder: The DocumentEmbedder to use for calculating embeddings.
        :param sentences_per_group: Number of sentences to group together before embedding.
        :param percentile: Percentile threshold for cosine distance. Distances above this percentile
            are treated as break points.
        :param min_length: Minimum length of splits in characters. Splits below this length will be merged.
        :param max_length: Maximum length of splits in characters. Splits above this length will be recursively split.
        :param language: Language for sentence tokenization.
        :param use_split_rules: Whether to use additional split rules for sentence tokenization. Applies additional
            split rules from SentenceSplitter to the sentence spans.
        :param extend_abbreviations: If True, the abbreviations used by NLTK's PunktTokenizer are extended by a list
            of curated abbreviations. Currently supported languages are: en, de.
            If False, the default abbreviations are used.
        """
        self.document_embedder = document_embedder

        if sentences_per_group <= 0:
            raise ValueError("sentences_per_group must be greater than 0.")
        self.sentences_per_group = sentences_per_group

        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be between 0.0 and 1.0.")
        self.percentile = percentile

        if min_length < 0:
            raise ValueError("min_length must be greater than or equal to 0.")
        self.min_length = min_length

        if max_length <= min_length:
            raise ValueError("max_length must be greater than min_length.")
        self.max_length = max_length

        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations
        self.sentence_splitter: SentenceSplitter | None = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the component by initializing the sentence splitter.
        """
        self.sentence_splitter = SentenceSplitter(
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
            keep_white_spaces=True,
        )
        if hasattr(self.document_embedder, "warm_up"):
            self.document_embedder.warm_up()
        self._is_warmed_up = True

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Split documents based on embedding similarity.

        :param documents: The documents to split.
        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field `source_id` to track the original document.
                - A metadata field `split_id` to track the split number.
                - A metadata field `page_number` to track the original page number.
                - All other metadata copied from the original document.

        :raises:
            - `RuntimeError`: If the component wasn't warmed up.
            - `TypeError`: If the input is not a list of Documents.
            - `ValueError`: If the document content is None or empty.
        """
        if not self._is_warmed_up:
            raise RuntimeError(
                "The component EmbeddingBasedDocumentSplitter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("EmbeddingBasedDocumentSplitter expects a List of Documents as input.")

        split_docs: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"EmbeddingBasedDocumentSplitter only works with text documents but content for "
                    f"document ID {doc.id} is None."
                )
            if doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue

            doc_splits = self._split_document(doc=doc)
            split_docs.extend(doc_splits)

        return {"documents": split_docs}

    def _split_document(self, doc: Document) -> list[Document]:
        """
        Split a single document based on embedding similarity.
        """
        # Create an initial split of the document content into smaller chunks
        # doc.content is validated in `run`
        splits = self._split_text(text=doc.content)  # type: ignore

        # Merge splits smaller than min_length
        merged_splits = self._merge_small_splits(splits=splits)

        # Recursively split splits larger than max_length
        final_splits = self._split_large_splits(splits=merged_splits)

        # Create Document objects from the final splits
        return EmbeddingBasedDocumentSplitter._create_documents_from_splits(splits=final_splits, original_doc=doc)

    def _split_text(self, text: str) -> list[str]:
        """
        Split a text into smaller chunks based on embedding similarity.
        """

        # NOTE: `self.sentence_splitter.split_sentences` strips all white space types (e.g. new lines, page breaks,
        # etc.) at the end of the provided text. So to not lose them, we need keep track of them and add them back to
        # the last sentence.
        rstripped_text = text.rstrip()
        trailing_whitespaces = text[len(rstripped_text) :]

        # Split the text into sentences
        sentences_result = self.sentence_splitter.split_sentences(rstripped_text)  # type: ignore[union-attr]

        # Add back the stripped white spaces to the last sentence
        if sentences_result and trailing_whitespaces:
            sentences_result[-1]["sentence"] += trailing_whitespaces
            sentences_result[-1]["end"] += len(trailing_whitespaces)

        sentences = [sentence["sentence"] for sentence in sentences_result]
        sentence_groups = self._group_sentences(sentences=sentences)
        embeddings = self._calculate_embeddings(sentence_groups=sentence_groups)
        split_points = self._find_split_points(embeddings=embeddings)
        sub_splits = self._create_splits_from_points(sentence_groups=sentence_groups, split_points=split_points)

        return sub_splits

    def _group_sentences(self, sentences: list[str]) -> list[str]:
        """
        Group sentences into groups of sentences_per_group.
        """
        if self.sentences_per_group == 1:
            return sentences

        groups = []
        for i in range(0, len(sentences), self.sentences_per_group):
            group = sentences[i : i + self.sentences_per_group]
            groups.append("".join(group))

        return groups

    def _calculate_embeddings(self, sentence_groups: list[str]) -> list[list[float]]:
        """
        Calculate embeddings for each sentence group using the DocumentEmbedder.
        """
        # Create Document objects for each group
        group_docs = [Document(content=group) for group in sentence_groups]
        result = self.document_embedder.run(group_docs)
        embedded_docs = result["documents"]
        embeddings = [doc.embedding for doc in embedded_docs]
        return embeddings

    def _find_split_points(self, embeddings: list[list[float]]) -> list[int]:
        """
        Find split points based on cosine distances between sequential embeddings.
        """
        if len(embeddings) <= 1:
            return []

        # Calculate cosine distances between sequential pairs
        distances = []
        for i in range(len(embeddings) - 1):
            distance = EmbeddingBasedDocumentSplitter._cosine_distance(
                embedding1=embeddings[i], embedding2=embeddings[i + 1]
            )
            distances.append(distance)

        # Calculate threshold based on percentile
        threshold = np.percentile(distances, self.percentile * 100)

        # Find indices where distance exceeds threshold
        split_points = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                split_points.append(i + 1)  # +1 because we want to split after this point

        return split_points

    @staticmethod
    def _cosine_distance(embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine distance between two embeddings.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))

        if norm1 == 0 or norm2 == 0:
            return 1.0

        cosine_sim = float(np.dot(vec1, vec2) / (norm1 * norm2))

        return 1.0 - cosine_sim

    @staticmethod
    def _create_splits_from_points(sentence_groups: list[str], split_points: list[int]) -> list[str]:
        """
        Create splits based on split points.
        """
        if not split_points:
            return ["".join(sentence_groups)]

        splits = []
        start = 0

        for point in split_points:
            split_text = "".join(sentence_groups[start:point])
            if split_text:
                splits.append(split_text)
            start = point

        # Add the last split
        if start < len(sentence_groups):
            split_text = "".join(sentence_groups[start:])
            if split_text:
                splits.append(split_text)

        return splits

    def _merge_small_splits(self, splits: list[str]) -> list[str]:
        """
        Merge splits that are below min_length.
        """
        if not splits:
            return splits

        merged = []
        current_split = splits[0]

        for split in splits[1:]:
            # We merge splits that are smaller than min_length but only if the newly merged split is still below
            # max_length.
            if len(current_split) < self.min_length and len(current_split) + len(split) < self.max_length:
                # Merge with next split
                current_split += split
            else:
                # Current split is long enough, save it and start a new one
                merged.append(current_split)
                current_split = split

        # Don't forget the last split
        merged.append(current_split)

        return merged

    def _split_large_splits(self, splits: list[str]) -> list[str]:
        """
        Recursively split splits that are above max_length.

        This method checks each split and if it exceeds max_length, it attempts to split it further using the same
        embedding-based approach. This is done recursively until all splits are within the max_length limit or no
        further splitting is possible.

        This works because the threshold for splits is calculated dynamically based on the provided of embeddings.
        """
        final_splits = []

        for split in splits:
            if len(split) <= self.max_length:
                final_splits.append(split)
            else:
                # Recursively split large splits
                # We can reuse the same _split_text method to split the text into smaller chunks because the threshold
                # for splits is calculated dynamically based on embeddings from `split`.
                sub_splits = self._split_text(text=split)

                # Stop splitting if no further split is possible or continue with recursion
                if len(sub_splits) == 1:
                    logger.warning(
                        f"Could not split a chunk further below max_length={self.max_length}. "
                        f"Returning chunk of length {len(split)}."
                    )
                    final_splits.append(split)
                else:
                    final_splits.extend(self._split_large_splits(splits=sub_splits))

        return final_splits

    @staticmethod
    def _create_documents_from_splits(splits: list[str], original_doc: Document) -> list[Document]:
        """
        Create Document objects from splits.
        """
        documents = []
        metadata = deepcopy(original_doc.meta)
        metadata["source_id"] = original_doc.id

        # Calculate page numbers for each split
        current_page = 1

        for i, split_text in enumerate(splits):
            split_meta = deepcopy(metadata)
            split_meta["split_id"] = i

            # Calculate page number for this split
            # Count page breaks in the split itself
            page_breaks_in_split = split_text.count("\f")

            # Calculate the page number for this split
            split_meta["page_number"] = current_page

            doc = Document(content=split_text, meta=split_meta)
            documents.append(doc)

            # Update page counter for next split
            current_page += page_breaks_in_split

        return documents

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Serialized dictionary representation of the component.
        """
        return default_to_dict(
            self,
            document_embedder=component_to_dict(obj=self.document_embedder, name="document_embedder"),
            sentences_per_group=self.sentences_per_group,
            percentile=self.percentile,
            min_length=self.min_length,
            max_length=self.max_length,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingBasedDocumentSplitter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize and create the component.

        :returns:
            The deserialized component.
        """
        deserialize_component_inplace(data["init_parameters"], key="document_embedder")
        return default_from_dict(cls, data)
