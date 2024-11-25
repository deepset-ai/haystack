# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream, Document
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install jq'") as jq_import:
    import jq


@component
class JSONConverter:
    """
    Converts one or more JSON files into a text document.

    ### Usage examples

    ```python
    import json

    from haystack.components.converters import JSONConverter
    from haystack.dataclasses import ByteStream

    source = ByteStream.from_string(json.dumps({"text": "This is the content of my document"}))

    converter = JSONConverter(content_key="text")
    results = converter.run(sources=[source])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is the content of my document'
    ```

    Optionally, you can also provide a `jq_schema` string to filter the JSON source files and `extra_meta_fields`
    to extract from the filtered data:

    ```python
    import json

    from haystack.components.converters import JSONConverter
    from haystack.dataclasses import ByteStream

    data = {
        "laureates": [
            {
                "firstname": "Enrico",
                "surname": "Fermi",
                "motivation": "for his demonstrations of the existence of new radioactive elements produced "
                "by neutron irradiation, and for his related discovery of nuclear reactions brought about by"
                " slow neutrons",
            },
            {
                "firstname": "Rita",
                "surname": "Levi-Montalcini",
                "motivation": "for their discoveries of growth factors",
            },
        ],
    }
    source = ByteStream.from_string(json.dumps(data))
    converter = JSONConverter(
        jq_schema=".laureates[]", content_key="motivation", extra_meta_fields={"firstname", "surname"}
    )

    results = converter.run(sources=[source])
    documents = results["documents"]
    print(documents[0].content)
    # 'for his demonstrations of the existence of new radioactive elements produced by
    # neutron irradiation, and for his related discovery of nuclear reactions brought
    # about by slow neutrons'

    print(documents[0].meta)
    # {'firstname': 'Enrico', 'surname': 'Fermi'}

    print(documents[1].content)
    # 'for their discoveries of growth factors'

    print(documents[1].meta)
    # {'firstname': 'Rita', 'surname': 'Levi-Montalcini'}
    ```

    """

    def __init__(
        self,
        jq_schema: Optional[str] = None,
        content_key: Optional[str] = None,
        extra_meta_fields: Optional[Union[Set[str], Literal["*"]]] = None,
        store_full_path: bool = True,
    ):
        """
        Creates a JSONConverter component.

        An optional `jq_schema` can be provided to extract nested data in the JSON source files.
        See the [official jq documentation](https://jqlang.github.io/jq/) for more info on the filters syntax.
        If `jq_schema` is not set, whole JSON source files will be used to extract content.

        Optionally, you can provide a `content_key` to specify which key in the extracted object must
        be set as the document's content.

        If both `jq_schema` and `content_key` are set, the component will search for the `content_key` in
        the JSON object extracted by `jq_schema`. If the extracted data is not a JSON object, it will be skipped.

        If only `jq_schema` is set, the extracted data must be a scalar value. If it's a JSON object or array,
        it will be skipped.

        If only `content_key` is set, the source JSON file must be a JSON object, else it will be skipped.

        `extra_meta_fields` can either be set to a set of strings or a literal `"*"` string.
        If it's a set of strings, it must specify fields in the extracted objects that must be set in
        the extracted documents. If a field is not found, the meta value will be `None`.
        If set to `"*"`, all fields that are not `content_key` found in the filtered JSON object will
        be saved as metadata.

        Initialization will fail if neither `jq_schema` nor `content_key` are set.

        :param jq_schema:
            Optional jq filter string to extract content.
            If not specified, whole JSON object will be used to extract information.
        :param content_key:
            Optional key to extract document content.
            If `jq_schema` is specified, the `content_key` will be extracted from that object.
        :param extra_meta_fields:
            An optional set of meta keys to extract from the content.
            If `jq_schema` is specified, all keys will be extracted from that object.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        self._compiled_filter = None
        if jq_schema:
            jq_import.check()
            self._compiled_filter = jq.compile(jq_schema)

        self._jq_schema = jq_schema
        self._content_key = content_key
        self._meta_fields = extra_meta_fields
        self._store_full_path = store_full_path

        if self._compiled_filter is None and self._content_key is None:
            msg = "No `jq_schema` nor `content_key` specified. Set either or both to extract data."
            raise ValueError(msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            jq_schema=self._jq_schema,
            content_key=self._content_key,
            extra_meta_fields=self._meta_fields,
            store_full_path=self._store_full_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def _get_content_and_meta(self, source: ByteStream) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Utility function to extract text and metadata from a JSON file.

        :param source:
            UTF-8 byte stream.
        :returns:
            Collection of text and metadata dict tuples, each corresponding
            to a different document.
        """
        try:
            file_content = source.data.decode("utf-8")
        except UnicodeError as exc:
            logger.warning(
                "Failed to extract text from {source}. Skipping it. Error: {error}",
                source=source.meta["file_path"],
                error=exc,
            )

        meta_fields = self._meta_fields or set()

        if self._compiled_filter is not None:
            try:
                objects = list(self._compiled_filter.input_text(file_content))
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source.meta["file_path"],
                    error=exc,
                )
                return []
        else:
            # We just load the whole file as JSON if the user didn't provide a jq filter.
            # We put it in a list even if it's not to ease handling it later on.
            objects = [json.loads(file_content)]

        result = []
        if self._content_key is not None:
            for obj in objects:
                if not isinstance(obj, dict):
                    logger.warning("Expected a dictionary but got {obj}. Skipping it.", obj=obj)
                    continue
                if self._content_key not in obj:
                    logger.warning(
                        "'{content_key}' not found in {obj}. Skipping it.", content_key=self._content_key, obj=obj
                    )
                    continue

                text = obj[self._content_key]
                if isinstance(text, (dict, list)):
                    logger.warning("Expected a scalar value but got {obj}. Skipping it.", obj=obj)
                    continue

                meta = {}
                if meta_fields == "*":
                    meta = {k: v for k, v in obj.items() if k != self._content_key}
                else:
                    for field in meta_fields:
                        meta[field] = obj.get(field, None)
                result.append((text, meta))
        else:
            for obj in objects:
                if isinstance(obj, (dict, list)):
                    logger.warning("Expected a scalar value but got {obj}. Skipping it.", obj=obj)
                    continue
                result.append((str(obj), {}))

        return result

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts a list of JSON files to documents.

        :param sources:
            A list of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, the length of the list must match the number of sources.
            If `sources` contain ByteStream objects, their `meta` will be added to the output documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of created documents.
        """
        documents = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as exc:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=exc)
                continue

            data = self._get_content_and_meta(bytestream)

            warnings.warn(
                "The `store_full_path` parameter defaults to True, storing full file paths in metadata. "
                "In the 2.9.0 release, the default value for `store_full_path` will change to False, "
                "storing only file names to improve privacy.",
                DeprecationWarning,
            )
            for text, extra_meta in data:
                merged_metadata = {**bytestream.meta, **metadata, **extra_meta}

                if not self._store_full_path and (file_path := bytestream.meta.get("file_path")):
                    merged_metadata["file_path"] = os.path.basename(file_path)
                document = Document(content=text, meta=merged_metadata)
                documents.append(document)

        return {"documents": documents}
