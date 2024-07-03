# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils.type_serialization import deserialize_type

with LazyImport("Run 'pip install flatten_json' Github - https://github.com/amirziai/flatten ") as flatten_import:
    from flatten_json import flatten

import json

logger = logging.getLogger(__name__)


@component
class JSONToDocument:
    """ 
    
    Convert JSON files to Documents.
    
    Converts JSON files to Documents by flattening the JSON and then
    mapping it to content field of the Document class.  The number of
    Document objects will be the same as the number of sources provided.  The
    assumption is that the JSON is encoded in utf-8


    :param sources:
    
        List of HTML, file paths or ByteStream objects.
        
    :param meta:
    
        Optional metadata to attach to the Documents.
        This value can be either a list of dictionaries or a single dictionary.
        If it's a single dictionary, its content is added to the metadata of all produced Documents.
        If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
        If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.
        
    :returns:
    
    A dictionary with the following keys:
    - `documents`: Created Documents

    Usage example:
    ```python
    from haystack.components.converters.txt import JSONToDocument

    converter = JSONToDocument()
    results = converter.run(sources=["sample.json"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is the JSON content from the JSON file.'


    The flatten_json library is used to flatten JSON for the purposes of embedding
    and semantic searching.  Example:
    nested_json = {
    "store": {
        "book": [
            {"category": "fiction", "price": 8.95, "title": "Book A"},
            {"category": "non-fiction", "price": 12.99, "title": "Book B"}
        ]
    }
}

flattened JSON

    {'store_book_0_category': 'fiction', 'store_book_0_price': 8.95, 'store_book_0_title': 'Book A', 'store_book_1_category': 'non-fiction', 'store_book_1_price': 12.99, 'store_book_1_title': 'Book B'}


    """

    
    def __init__(self: "JSONToDocument") -> None:
        """
        Create a JSONToDocument component.
        """

        flatten_import.check()




    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Converts JSON files to Documents.

        :param sources:
            List of HTML file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """
        documents = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                # Convert the ByteStream data to a string
                json_str = bytestream.data.decode("utf-8")

                # Load the string into a dictionary
                json_dict = json.loads(json_str)

                # Flatten the dictionary
                flattened_json_dict = flatten(json_dict)

                # Convert to a string
                flattened_string = json.dumps(flattened_json_dict)


            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue



            merged_metadata = {**bytestream.meta, **metadata}
            document = Document(content=flattened_string, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
