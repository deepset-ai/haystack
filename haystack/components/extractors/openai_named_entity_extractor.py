import os
import json

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret

from openai import OpenAI
from tqdm import tqdm


@component
class OpenAINamedEntityExtractor:

    api_key: str
    model: str
    client: OpenAI

    _METADATA_KEY = "named_entities"

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-4o-mini",
        batch_size: int = 32,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Extracts named entities from text using OpenAI models.

        Args:
            api_key (Secret, optional): OpenAI API Key. Defaults to Secret.from_env_var("OPENAI_API_KEY").
            model (str, optional): OpenAI model to be used. Defaults to "gpt-4o-mini".
            batch_size (int, optional): Batch size for processing. Defaults to 32.
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", 5))

        self.client = OpenAI(
            api_key=api_key.resolve_value(),
            organization=organization,
            base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def warm_up(self):
        pass

    def _prepare_texts_to_process(self, documents: List[Document]) -> List[str]:
        return [doc.content for doc in documents]

    def _extract_entities(self, texts: List[str]) -> List[str]:
        tags = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting named entities"):
            batch = texts[i : i + self.batch_size]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract all named entities such as Names, Places from the text array provided. Return a 2-dimensional JSON array without any markdown formatting.",
                    },
                    {"role": "user", "content": json.dumps(batch)},
                ],
            )
            try:
                entities = json.loads(completion.choices[0].message.content)
                tags.extend(entities)
            except json.JSONDecodeError:
                pass
        return tags

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        if (
            not isinstance(documents, list)
            or documents
            and not isinstance(documents[0], Document)
        ):
            raise TypeError(
                "OpenAINamedEntityExtractor expects a list of Documents as input."
            )
        
        texts_to_process = self._prepare_texts_to_process(documents)

        tags = self._extract_entities(texts_to_process)
        
        for doc, entities in zip(documents, tags):
            doc.meta[OpenAINamedEntityExtractor._METADATA_KEY] = entities
        return {"documents": documents}

    @classmethod
    def get_stored_annotations(cls, document: Document) -> Optional[List[str]]:
        """
        Returns the document's named entity annotations stored in its metadata, if any.

        :param document:
            Document whose annotations are to be fetched.
        :returns:
            The stored annotations.
        """

        return document.meta.get(cls._METADATA_KEY)
