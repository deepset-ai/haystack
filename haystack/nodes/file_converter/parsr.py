from typing import Optional, Dict, List, Any
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import requests
import json
from pathlib import Path
import copy
import logging

from haystack.nodes.file_converter import BaseConverter

logger = logging.getLogger(__name__)


class ParsrConverter(BaseConverter):
    """
    File converter that makes use of the open-source Parsr tool by axa-group.
    (https://github.com/axa-group/Parsr).
    This Converter extracts both text and tables.
    Supported file formats are: PDF, DOCX
    """

    def __init__(self,
                 parsr_url: str = "http://localhost:3001",
                 extractor: Literal["pdfminer", "pdfjs"] = "pdfminer",
                 table_detection_mode: Literal["lattice", "stream"] = "lattice",
                 preceding_context_len: int = 1,
                 following_context_len: int = 1,
                 remove_page_headers: bool = False,
                 remove_page_footers: bool = False,
                 remove_table_of_contents: bool = False,
                 valid_languages: Optional[List[str]] = None,
                 ):
        """
        :param parsr_url: URL endpoint to Parsr"s REST API.
        :param extractor: Backend used to extract textual structured from PDFs. ("pdfminer" or "pdfjs")
        :param table_detection_mode: Parsing method used to detect tables and their cells.
                                     "lattice" detects tables and their cells by demarcated lines between cells.
                                     "stream" detects tables and their cells by looking at whitespace between cells.
        :param preceding_context_len: Number of paragraphs before a table to extract as preceding context
                                      (will be returned as part of meta data).
        :param following_context_len: Number of paragraphs after a table to extract as preceding context
                                      (will be returned as part of meta data).
        :param remove_page_headers: Whether to remove text that Parsr detected as a page header.
        :param remove_page_footers: Whether to remove text that Parsr detected as a page footer.
        :param remove_table_of_contents: Whether to remove text that Parsr detected as a table of contents.
        :param valid_languages: Validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        # save init parameters to enable export of component config as YAML
        self.set_config(parsr_url=parsr_url, extractor=extractor, table_detection_mode=table_detection_mode,
                        preceding_context_len=preceding_context_len,  following_context_len=following_context_len,
                        remove_page_headers=remove_page_headers, remove_page_footers=remove_page_footers,
                        remove_table_of_contents=remove_table_of_contents, valid_languages=valid_languages)

        try:
            ping = requests.get(parsr_url)
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Parsr server is not reachable at the URL '{parsr_url}'. To run it locally "
                f"with Docker, execute: 'docker run -p 3001:3001 axarev/parsr:v1.2.2'"
            )
        if ping.status_code != 200:
            raise Exception(
                f"Parsr server is not reachable at the URL '{parsr_url}'. (Status code: {ping.status_code} {ping.reason})\n"
                f"To run it locally with Docker, execute: 'docker run -p 3001:3001 axarev/parsr:v1.2.2'"
            )

        self.parsr_url = parsr_url
        self.valid_languages = valid_languages
        self.config = json.loads(requests.get(f"{self.parsr_url}/api/v1/default-config").content)
        self.config["extractor"]["pdf"] = extractor
        self.config['cleaner'][5][1]['runConfig'][0]['flavor'] = table_detection_mode
        self.preceding_context_len = preceding_context_len
        self.following_context_len = following_context_len
        self.remove_page_headers = remove_page_headers
        self.remove_page_footers = remove_page_footers
        self.remove_table_of_contents = remove_table_of_contents
        super().__init__(valid_languages=valid_languages)

    def convert(self,
                file_path: Path,
                meta: Optional[Dict[str, str]] = None,
                remove_numeric_tables: Optional[bool] = None,
                valid_languages: Optional[List[str]] = None,
                encoding: Optional[str] = "utf-8",
                ) -> List[Dict[str, Any]]:
        """
        Extract text and tables from a PDF or DOCX using the open-source Parsr tool.

        :param file_path: Path to the file you want to convert.
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: Not applicable.
        :param valid_languages: Validate languages from a list of languages specified in the ISO 639-1
                                (https://en.wikipedia.org/wiki/ISO_639-1) format.
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        :param encoding: Not applicable..
        """
        if valid_languages is None:
            valid_languages = self.valid_languages

        with open(file_path, "rb") as pdf_file:
            # Send file to Parsr
            send_response = requests.post(
                url=f"{self.parsr_url}/api/v1/document",
                files={
                    "file": (file_path, pdf_file, "application/pdf"),
                    "config": ("config", json.dumps(self.config), "application/json")
                }
            )
            queue_id = send_response.text

            # Wait until Parsr processing is done
            status_response = requests.get(url=f"{self.parsr_url}/api/v1/queue/{queue_id}")
            while status_response.status_code == 200 and status_response.status_code != 201:
                status_response = requests.get(url=f"{self.parsr_url}/api/v1/queue/{queue_id}")

            # Get Parsr output
            result_response = requests.get(url=f"{self.parsr_url}/api/v1/json/{queue_id}")
            parsr_output = json.loads(result_response.content)

            # Convert Parsr output to Haystack Documents
            text = ""
            tables = []
            for page_idx, page in enumerate(parsr_output["pages"]):
                for elem_idx, element in enumerate(page["elements"]):
                    if element["type"] in ["paragraph", "heading", "table-of-contents"]:
                        current_paragraph = self._convert_text_element(element)
                        if current_paragraph:
                            text += f"{current_paragraph}\n\n"

                    elif element["type"] == "table":
                        table = self._convert_table_element(element, parsr_output["pages"], page_idx, elem_idx, meta)
                        tables.append(table)

        if valid_languages:
            file_text = text + " ".join(
                [cell for table in tables for row in table["content"] for cell in row])
            if not self.validate_language(file_text, valid_languages):
                logger.warning(
                    f"The language for {file_path} is not one of {valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        docs = tables + [{"content": text.strip(), "content_type": "text", "meta": meta}]
        return docs

    @staticmethod
    def _get_paragraph_string(paragraph: Dict[str, Any]) -> str:
        current_lines = []
        for line in paragraph["content"]:
            current_line = " ".join([word["content"] for word in line["content"]])
            current_lines.append(current_line)
        current_paragraph = "\n".join(current_lines)

        return current_paragraph

    def _convert_text_element(self, element: Dict[str, Any]) -> str:
        if self.remove_page_headers and "isHeader" in element["properties"]:
            return ""
        if self.remove_page_footers and "isFooter" in element["properties"]:
            return ""
        if element["type"] == "table-of-contents":
            if self.remove_table_of_contents:
                return ""
            else:
                current_paragraph = "\n".join([self._get_paragraph_string(elem) for elem in element["content"]])
                return current_paragraph

        current_paragraph = self._get_paragraph_string(element)
        return current_paragraph

    def _convert_table_element(self, element: Dict[str, Any], all_pages: List[Dict], page_idx: int, elem_idx: int,
                               meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        row_idx_start = 0
        caption = ""
        table_list = [[""] * len(element["content"][0]["content"])
                      for _ in range(len(element["content"]))]

        for row_idx, row in enumerate(element["content"]):
            for col_idx, cell in enumerate(row["content"]):
                # Check if first row is a merged cell spanning whole table
                # -> exclude this row and use as caption
                if (row_idx == col_idx == 0) and (cell["colspan"] == len(table_list[0])):
                    cell_paragraphs = [self._get_paragraph_string(par) for par in cell["content"]]
                    cell_content = "\n\n".join(cell_paragraphs)
                    caption = cell_content
                    row_idx_start = 1
                    table_list.pop(0)
                    break

                if cell["type"] == "table-cell":
                    cell_paragraphs = [self._get_paragraph_string(par) for par in cell["content"]]
                    cell_content = "\n\n".join(cell_paragraphs)
                    for c in range(cell["colspan"]):
                        for r in range(cell["rowspan"]):
                            table_list[row_idx + r - row_idx_start][col_idx + c] = cell_content

        # Get preceding and following elements of table
        preceding_elements = []
        following_elements = []
        for cur_page_idx, cur_page in enumerate(all_pages):
            for cur_elem_index, elem in enumerate(cur_page["elements"]):
                if (elem["type"] in ["paragraph", "heading"]) \
                        and (self.remove_page_headers and "isHeader" not in elem["properties"]) \
                        and (self.remove_page_footers and "isFooter" not in elem["properties"]):
                    if cur_page_idx < page_idx:
                        preceding_elements.append(elem)
                    elif cur_page_idx == page_idx:
                        if cur_elem_index < elem_idx:
                            preceding_elements.append(elem)
                        elif cur_elem_index > elem_idx:
                            following_elements.append(elem)
                    elif cur_page_idx > page_idx:
                        following_elements.append(elem)

        preceding_context = "\n\n".join([self._get_paragraph_string(elem)
                                         for elem in preceding_elements]) + f"\n\n{caption}"
        preceding_context = preceding_context.strip()
        following_context = "\n\n".join([self._get_paragraph_string(elem)
                                         for elem in following_elements])
        following_context = following_context.strip()

        if meta is not None:
            table_meta = copy.deepcopy(meta)
            table_meta["preceding_context"] = preceding_context
            table_meta["following_context"] = following_context
        else:
            table_meta = {"preceding_context": preceding_context,
                          "following_context": following_context}

        return {"content": table_list, "content_type": "table", "meta": table_meta}
