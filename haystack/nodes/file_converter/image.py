from typing import List, Optional, Dict, Any, Union

import logging
import subprocess
from pathlib import Path
import pytesseract
from PIL.PpmImagePlugin import PpmImageFile
from PIL import Image

from haystack.nodes.file_converter import BaseConverter


logger = logging.getLogger(__name__)


class ImageToTextConverter(BaseConverter):
    def __init__(
        self,
        remove_numeric_tables: bool = False,
        valid_languages: Optional[List[str]] = ["eng"],
    ):
        """
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages specified here
                                (https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text. Run the following line of code to check available language packs:
                                # List of available languages
                                print(pytesseract.get_languages(config=''))
        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

        verify_installation = subprocess.run(["tesseract -v"], shell=True)
        if verify_installation.returncode == 127:
            raise Exception(
                """tesseract is not installed.
                
                   Installation on Linux:
                   apt-get install tesseract-ocr libtesseract-dev poppler-utils
                   
                   Installation on MacOS:
                   brew install tesseract
                   
                   For installing specific language packs check here: https://tesseract-ocr.github.io/tessdoc/Installation.html
                """
            )
        tesseract_langs = []
        if valid_languages:
            for language in valid_languages:
                if (
                    language in pytesseract.get_languages(config="")
                    and language not in tesseract_langs
                ):
                    tesseract_langs.append(language)
                else:
                    raise Exception(
                        f"""{language} is not either a valid tesseract language code or its language pack isn't installed.

                    Check the list of valid tesseract language codes here: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html

                    For installing specific language packs check here: https://tesseract-ocr.github.io/tessdoc/Installation.html
                    """
                    )

        ## if you have more than one language in images, then pass it to tesseract like this e.g., `fra+eng`
        self.tesseract_langs = "+".join(tesseract_langs)
        super().__init__(
            remove_numeric_tables=remove_numeric_tables, valid_languages=valid_languages
        )

    def convert(
        self,
        file_path: Union[Path,str],
        meta: Optional[Dict[str, str]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> List[Dict[str, Any]]:
        """
        Extract text from image file using the pytesseract library (https://github.com/madmaze/pytesseract)

        :param file_path: path to image file
        :param meta: Optional dictionary with metadata that shall be attached to all resulting documents.
                     Can be any custom keys and values.
        :param remove_numeric_tables: This option uses heuristics to remove numeric rows from the tables.
                                      The tabular structures in documents might be noise for the reader model if it
                                      does not have table parsing capability for finding answers. However, tables
                                      may also have long strings that could possible candidate for searching answers.
                                      The rows containing strings are thus retained in this option.
        :param valid_languages: validate languages from a list of languages supported by tessarect
                                (https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).
                                This option can be used to add test for encoding errors. If the extracted text is
                                not one of the valid languages, then it might likely be encoding error resulting
                                in garbled text.
        """
        file_path = Path(file_path)
        image = Image.open(file_path)
        pages = self._image_to_text(image)
        if remove_numeric_tables is None:
            remove_numeric_tables = self.remove_numeric_tables
        if valid_languages is None:
            valid_languages = self.valid_languages

        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            cleaned_lines = []
            for line in lines:
                words = line.split()
                digits = [word for word in words if any(i.isdigit() for i in word)]

                # remove lines having > 40% of words as digits AND not ending with a period(.)
                if remove_numeric_tables:
                    if (
                        words
                        and len(digits) / len(words) > 0.4
                        and not line.strip().endswith(".")
                    ):
                        logger.debug(f"Removing line '{line}' from file")
                        continue
                cleaned_lines.append(line)

            page = "\n".join(cleaned_lines)
            cleaned_pages.append(page)

        if valid_languages:
            document_text = "".join(cleaned_pages)
            if not self.validate_language(document_text):
                logger.warning(
                    f"The language for image is not one of {self.valid_languages}. The file may not have "
                    f"been decoded in the correct text format."
                )

        text = "\f".join(cleaned_pages)
        document = {"content": text, "meta": meta}
        return [document]

    def _image_to_text(self, image: PpmImageFile) -> List[str]:
        """
        Extract text from image file.

        :param image: input image file
        """
        text = [pytesseract.image_to_string(image, lang=self.tesseract_langs)]
        return text
