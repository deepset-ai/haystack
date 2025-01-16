# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import networkx as nx
import pandas as pd

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"azure-ai-formrecognizer>=3.2.0b2\"'") as azure_import:
    from azure.ai.formrecognizer import AnalyzeResult, DocumentAnalysisClient, DocumentLine, DocumentParagraph
    from azure.core.credentials import AzureKeyCredential


@component
class AzureOCRDocumentConverter:
    """
    Converts files to documents using Azure's Document Intelligence service.

    Supported file formats are: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, and HTML.

    To use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. For help with setting up your resource, see
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

    ### Usage example

    ```python
    from haystack.components.converters import AzureOCRDocumentConverter
    from haystack.utils import Secret

    converter = AzureOCRDocumentConverter(endpoint="<url>", api_key=Secret.from_token("<your-api-key>"))
    results = converter.run(sources=["path/to/doc_with_images.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the PDF file.'
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        endpoint: str,
        api_key: Secret = Secret.from_env_var("AZURE_AI_API_KEY"),
        model_id: str = "prebuilt-read",
        preceding_context_len: int = 3,
        following_context_len: int = 3,
        merge_multiple_column_headers: bool = True,
        page_layout: Literal["natural", "single_column"] = "natural",
        threshold_y: Optional[float] = 0.05,
        store_full_path: bool = False,
    ):
        """
        Creates an AzureOCRDocumentConverter component.

        :param endpoint:
            The endpoint of your Azure resource.
        :param api_key:
            The API key of your Azure resource.
        :param model_id:
            The ID of the model you want to use. For a list of available models, see [Azure documentation]
            (https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/choose-model-feature).
        :param preceding_context_len: Number of lines before a table to include as preceding context
            (this will be added to the metadata).
        :param following_context_len: Number of lines after a table to include as subsequent context (
            this will be added to the metadata).
        :param merge_multiple_column_headers: If `True`, merges multiple column header rows into a single row.
        :param page_layout: The type reading order to follow. Possible options:
            - `natural`: Uses the natural reading order determined by Azure.
            - `single_column`: Groups all lines with the same height on the page based on a threshold
            determined by `threshold_y`.
        :param threshold_y: Only relevant if `single_column` is set to `page_layout`.
            The threshold, in inches, to determine if two recognized PDF elements are grouped into a
            single line. This is crucial for section headers or numbers which may be spatially separated
            from the remaining text on the horizontal axis.
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        azure_import.check()

        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key.resolve_value() or "")
        )  # type: ignore
        self.endpoint = endpoint
        self.model_id = model_id
        self.api_key = api_key
        self.preceding_context_len = preceding_context_len
        self.following_context_len = following_context_len
        self.merge_multiple_column_headers = merge_multiple_column_headers
        self.page_layout = page_layout
        self.threshold_y = threshold_y
        self.store_full_path = store_full_path
        if self.page_layout == "single_column" and self.threshold_y is None:
            self.threshold_y = 0.05

    @component.output_types(documents=List[Document], raw_azure_response=List[Dict])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[List[Dict[str, Any]]] = None):
        """
        Convert a list of files to Documents using Azure's Document Intelligence service.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be
            zipped. If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of created Documents
            - `raw_azure_response`: List of raw Azure responses used to create the Documents
        """
        documents = []
        azure_output = []
        meta_list: List[Dict[str, Any]] = normalize_metadata(meta=meta, sources_count=len(sources))
        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            poller = self.document_analysis_client.begin_analyze_document(
                model_id=self.model_id, document=bytestream.data
            )
            result = poller.result()
            azure_output.append(result.to_dict())

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)
            docs = self._convert_tables_and_text(result=result, meta=merged_metadata)
            documents.extend(docs)

        return {"documents": documents, "raw_azure_response": azure_output}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            endpoint=self.endpoint,
            model_id=self.model_id,
            preceding_context_len=self.preceding_context_len,
            following_context_len=self.following_context_len,
            merge_multiple_column_headers=self.merge_multiple_column_headers,
            page_layout=self.page_layout,
            threshold_y=self.threshold_y,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureOCRDocumentConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    # pylint: disable=line-too-long
    def _convert_tables_and_text(self, result: "AnalyzeResult", meta: Optional[Dict[str, Any]]) -> List[Document]:
        """
        Converts the tables and text extracted by Azure's Document Intelligence service into Haystack Documents.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
            Can be any custom keys and values.
        :returns: List of Documents containing the tables and text extracted from the AnalyzeResult object.
        """
        tables = self._convert_tables(result=result, meta=meta)
        if self.page_layout == "natural":
            text = self._convert_to_natural_text(result=result, meta=meta)
        else:
            assert isinstance(self.threshold_y, float)
            text = self._convert_to_single_column_text(result=result, meta=meta, threshold_y=self.threshold_y)
        docs = [*tables, text]
        return docs

    def _convert_tables(self, result: "AnalyzeResult", meta: Optional[Dict[str, Any]]) -> List[Document]:
        """
        Converts the tables extracted by Azure's Document Intelligence service into Haystack Documents.

        :param result: The AnalyzeResult Azure object
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.

        :returns: List of Documents containing the tables extracted from the AnalyzeResult object.
        """
        converted_tables: List[Document] = []

        if not result.tables:
            return converted_tables

        for table in result.tables:
            # Initialize table with empty cells
            table_list = [[""] * table.column_count for _ in range(table.row_count)]
            additional_column_header_rows = set()
            caption = ""
            row_idx_start = 0

            for idx, cell in enumerate(table.cells):
                # Remove ':selected:'/':unselected:' tags from cell's content
                cell.content = cell.content.replace(":selected:", "")
                cell.content = cell.content.replace(":unselected:", "")

                # Check if first row is a merged cell spanning whole table
                # -> exclude this row and use as a caption
                if idx == 0 and cell.column_span == table.column_count:
                    caption = cell.content
                    row_idx_start = 1
                    table_list.pop(0)
                    continue

                column_span = cell.column_span if cell.column_span else 0
                for c in range(column_span):  # pylint: disable=invalid-name
                    row_span = cell.row_span if cell.row_span else 0
                    for r in range(row_span):  # pylint: disable=invalid-name
                        if (
                            self.merge_multiple_column_headers
                            and cell.kind == "columnHeader"
                            and cell.row_index > row_idx_start
                        ):
                            # More than one row serves as column header
                            table_list[0][cell.column_index + c] += f"\n{cell.content}"
                            additional_column_header_rows.add(cell.row_index - row_idx_start)
                        else:
                            table_list[cell.row_index + r - row_idx_start][cell.column_index + c] = cell.content

            # Remove additional column header rows, as these got attached to the first row
            for row_idx in sorted(additional_column_header_rows, reverse=True):
                del table_list[row_idx]

            # Get preceding context of table
            if table.bounding_regions:
                table_beginning_page = next(
                    page for page in result.pages if page.page_number == table.bounding_regions[0].page_number
                )
            else:
                table_beginning_page = None
            table_start_offset = table.spans[0].offset
            if table_beginning_page and table_beginning_page.lines:
                preceding_lines = [
                    line.content for line in table_beginning_page.lines if line.spans[0].offset < table_start_offset
                ]
            else:
                preceding_lines = []
            preceding_context = "\n".join(preceding_lines[-self.preceding_context_len :]) + f"\n{caption}"
            preceding_context = preceding_context.strip()

            # Get following context
            if table.bounding_regions and len(table.bounding_regions) == 1:
                table_end_page = table_beginning_page
            elif table.bounding_regions:
                table_end_page = next(
                    page for page in result.pages if page.page_number == table.bounding_regions[-1].page_number
                )
            else:
                table_end_page = None

            table_end_offset = table_start_offset + table.spans[0].length
            if table_end_page and table_end_page.lines:
                following_lines = [
                    line.content for line in table_end_page.lines if line.spans[0].offset > table_end_offset
                ]
            else:
                following_lines = []
            following_context = "\n".join(following_lines[: self.following_context_len])

            table_meta = copy.deepcopy(meta)

            if isinstance(table_meta, dict):
                table_meta["preceding_context"] = preceding_context
                table_meta["following_context"] = following_context
            else:
                table_meta = {"preceding_context": preceding_context, "following_context": following_context}

            if table.bounding_regions:
                table_meta["page"] = table.bounding_regions[0].page_number

            table_df = pd.DataFrame(columns=table_list[0], data=table_list[1:])

            # Use custom ID for tables, as columns might not be unique and thus failing in the default ID generation
            pd_hashes = self._hash_dataframe(table_df)
            data = f"{pd_hashes}{table_meta}"
            doc_id = hashlib.sha256(data.encode()).hexdigest()
            converted_tables.append(Document(id=doc_id, dataframe=table_df, meta=table_meta))

        return converted_tables

    def _convert_to_natural_text(self, result: "AnalyzeResult", meta: Optional[Dict[str, Any]]) -> Document:
        """
        This converts the `AnalyzeResult` object into a single document.

        We add "\f" separators between to differentiate between the text on separate pages. This is the expected format
        for the PreProcessor.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
            Can be any custom keys and values.
        :returns: A single Document containing all the text extracted from the AnalyzeResult object.
        """
        table_spans_by_page = self._collect_table_spans(result=result)

        texts = []
        if result.paragraphs:
            paragraphs_to_pages: Dict[int, str] = defaultdict(str)
            for paragraph in result.paragraphs:
                if paragraph.bounding_regions:
                    # If paragraph spans multiple pages we group it with the first page number
                    page_numbers = [b.page_number for b in paragraph.bounding_regions]
                else:
                    # If page_number is not available we put the paragraph onto an existing page
                    current_last_page_number = sorted(paragraphs_to_pages.keys())[-1] if paragraphs_to_pages else 1
                    page_numbers = [current_last_page_number]
                tables_on_page = table_spans_by_page[page_numbers[0]]
                # Check if paragraph is part of a table and if so skip
                if self._check_if_in_table(tables_on_page, line_or_paragraph=paragraph):
                    continue
                paragraphs_to_pages[page_numbers[0]] += paragraph.content + "\n"

            max_page_number: int = max(paragraphs_to_pages)
            for page_idx in range(1, max_page_number + 1):
                # We add empty strings for missing pages so the preprocessor can still extract the correct page number
                # from the original PDF.
                page_text = paragraphs_to_pages.get(page_idx, "")
                texts.append(page_text)
        else:
            logger.warning("No text paragraphs were detected by the OCR conversion.")

        all_text = "\f".join(texts)
        return Document(content=all_text, meta=meta if meta else {})

    def _convert_to_single_column_text(
        self, result: "AnalyzeResult", meta: Optional[Dict[str, str]], threshold_y: float = 0.05
    ) -> Document:
        """
        This converts the `AnalyzeResult` object into a single Haystack Document.

        We add "\f" separators between to differentiate between the text on separate pages. This is the expected format
        for the PreProcessor.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method. Docs on Analyze result
            can be found [here](https://azuresdkdocs.blob.core.windows.net/$web/python/azure-ai-formrecognizer/3.3.0/azure.ai.formrecognizer.html?highlight=read#azure.ai.formrecognizer.AnalyzeResult).
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
            Can be any custom keys and values.
        :param threshold_y: height threshold in inches for PDF and pixels for images
        :returns: A single Document containing all the text extracted from the AnalyzeResult object.
        """
        table_spans_by_page = self._collect_table_spans(result=result)

        # Find all pairs of lines that should be grouped together based on the y-value of the upper left coordinate
        # of their bounding box
        pairs_by_page = defaultdict(list)
        for page_idx, page in enumerate(result.pages):
            lines = page.lines if page.lines else []
            # Only works if polygons is available
            if all(line.polygon is not None for line in lines):
                for i in range(len(lines)):  # pylint: disable=consider-using-enumerate
                    # left_upi, right_upi, right_lowi, left_lowi = lines[i].polygon
                    left_upi, _, _, _ = lines[i].polygon  # type: ignore
                    pairs_by_page[page_idx].append([i, i])
                    for j in range(i + 1, len(lines)):  # pylint: disable=invalid-name
                        left_upj, _, _, _ = lines[j].polygon  # type: ignore
                        close_on_y_axis = abs(left_upi[1] - left_upj[1]) < threshold_y
                        if close_on_y_axis:
                            pairs_by_page[page_idx].append([i, j])
            # Default if polygon is not available
            else:
                logger.info(
                    "Polygon information for lines on page {page_idx} is not available so it is not possible "
                    "to enforce a single column page layout.".format(page_idx=page_idx)
                )
                for i in range(len(lines)):
                    pairs_by_page[page_idx].append([i, i])

        # merged the line pairs that are connected by page
        merged_pairs_by_page = {}
        for page_idx in pairs_by_page:
            graph = nx.Graph()
            graph.add_edges_from(pairs_by_page[page_idx])
            merged_pairs_by_page[page_idx] = [list(a) for a in list(nx.connected_components(graph))]

        # Convert line indices to the DocumentLine objects
        merged_lines_by_page = {}
        for page_idx, page in enumerate(result.pages):
            rows = []
            lines = page.lines if page.lines else []
            # We use .get(page_idx, []) since the page could be empty
            for row_of_lines in merged_pairs_by_page.get(page_idx, []):
                lines_in_row = [lines[line_idx] for line_idx in row_of_lines]
                rows.append(lines_in_row)
            merged_lines_by_page[page_idx] = rows

        # Sort the merged pairs in each row by the x-value of the upper left bounding box coordinate
        x_sorted_lines_by_page = {}
        for page_idx, _ in enumerate(result.pages):
            sorted_rows = []
            for row_of_lines in merged_lines_by_page[page_idx]:
                sorted_rows.append(sorted(row_of_lines, key=lambda x: x.polygon[0][0]))  # type: ignore
            x_sorted_lines_by_page[page_idx] = sorted_rows

        # Sort each row within the page by the y-value of the upper left bounding box coordinate
        y_sorted_lines_by_page = {}
        for page_idx, _ in enumerate(result.pages):
            sorted_rows = sorted(x_sorted_lines_by_page[page_idx], key=lambda x: x[0].polygon[0][1])  # type: ignore
            y_sorted_lines_by_page[page_idx] = sorted_rows

        # Construct the text to write
        texts = []
        for page_idx, page in enumerate(result.pages):
            tables_on_page = table_spans_by_page[page.page_number]
            page_text = ""
            for row_of_lines in y_sorted_lines_by_page[page_idx]:
                # Check if line is part of a table and if so skip
                if any(self._check_if_in_table(tables_on_page, line_or_paragraph=line) for line in row_of_lines):
                    continue
                page_text += " ".join(line.content for line in row_of_lines)
                page_text += "\n"
            texts.append(page_text)
        all_text = "\f".join(texts)
        return Document(content=all_text, meta=meta if meta else {})

    def _collect_table_spans(self, result: "AnalyzeResult") -> Dict:
        """
        Collect the spans of all tables by page number.

        :param result: The AnalyzeResult object returned by the `begin_analyze_document` method.
        :returns: A dictionary with the page number as key and a list of table spans as value.
        """
        table_spans_by_page = defaultdict(list)
        tables = result.tables if result.tables else []
        for table in tables:
            if not table.bounding_regions:
                continue
            table_spans_by_page[table.bounding_regions[0].page_number].append(table.spans[0])
        return table_spans_by_page

    def _check_if_in_table(
        self, tables_on_page: dict, line_or_paragraph: Union["DocumentLine", "DocumentParagraph"]
    ) -> bool:
        """
        Check if a line or paragraph is part of a table.

        :param tables_on_page: A dictionary with the page number as key and a list of table spans as value.
        :param line_or_paragraph: The line or paragraph to check.
        :returns: True if the line or paragraph is part of a table, False otherwise.
        """
        in_table = False
        # Check if line is part of a table
        for table in tables_on_page:
            if table.offset <= line_or_paragraph.spans[0].offset <= table.offset + table.length:
                in_table = True
                break
        return in_table

    def _hash_dataframe(self, df: pd.DataFrame, desired_samples=5, hash_length=4) -> str:
        """
        Returns a hash of the DataFrame content.

        The hash is based on the content of the DataFrame.
        :param df: The DataFrame to hash.
        :param desired_samples: The desired number of samples to hash.
        :param hash_length: The length of the hash for each sample.

        :returns: A hash of the DataFrame content.
        """
        # take adaptive sample of rows to hash because we can have very large dataframes
        hasher = hashlib.md5()
        total_rows = len(df)
        # sample rate based on DataFrame size and desired number of samples
        sample_rate = max(1, total_rows // desired_samples)

        hashes = pd.util.hash_pandas_object(df, index=True)
        sampled_hashes = hashes[::sample_rate]

        for hash_value in sampled_hashes:
            partial_hash = str(hash_value)[:hash_length].encode("utf-8")
            hasher.update(partial_hash)

        return hasher.hexdigest()
