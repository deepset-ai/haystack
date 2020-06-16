import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile, File

from haystack.api.config import DB_HOST, DB_PORT, DB_USER, DB_PW, DB_INDEX, ES_CONN_SCHEME, TEXT_FIELD_NAME, \
    SEARCH_FIELD_NAME, FILE_UPLOAD_PATH, EMBEDDING_DIM, EMBEDDING_FIELD_NAME, EXCLUDE_META_DATA_FIELDS, VALID_LANGUAGES, \
    FAQ_QUESTION_FIELD_NAME, REMOVE_NUMERIC_TABLES, REMOVE_WHITESPACE, REMOVE_EMPTY_LINES, REMOVE_HEADER_FOOTER
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.file_converters.pdftotext import PDFToTextConverter
from haystack.indexing.file_converters.text import TextConverter


logger = logging.getLogger(__name__)
router = APIRouter()


document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    scheme=ES_CONN_SCHEME,
    ca_certs=False,
    verify_certs=False,
    text_field=TEXT_FIELD_NAME,
    search_fields=SEARCH_FIELD_NAME,
    embedding_dim=EMBEDDING_DIM,
    embedding_field=EMBEDDING_FIELD_NAME,
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,  # type: ignore
    faq_question_field=FAQ_QUESTION_FIELD_NAME,
)

pdf_converter = PDFToTextConverter(
    remove_numeric_tables=REMOVE_NUMERIC_TABLES,
    remove_whitespace=REMOVE_WHITESPACE,
    remove_empty_lines=REMOVE_EMPTY_LINES,
    remove_header_footer=REMOVE_HEADER_FOOTER,
    valid_languages=VALID_LANGUAGES,  # type: ignore
)
txt_converter = TextConverter(
    remove_numeric_tables=REMOVE_NUMERIC_TABLES,
    remove_whitespace=REMOVE_WHITESPACE,
    remove_empty_lines=REMOVE_EMPTY_LINES,
    remove_header_footer=REMOVE_HEADER_FOOTER,
    valid_languages=VALID_LANGUAGES,  # type: ignore
)


@router.post("/file-upload")
def upload_file_to_document_store(file: UploadFile = File(...)) -> None:
    try:
        file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.split(".")[-1].lower() == "pdf":
            pages = pdf_converter.extract_pages(file_path)
        elif file.filename.split(".")[-1].lower() == "txt":
            pages = txt_converter.extract_pages(file_path)
        else:
            raise HTTPException(status_code=415, detail=f"Only .pdf and .txt file formats are supported.")

        document = {TEXT_FIELD_NAME: "\n".join(pages), "name": file.filename}
        document_store.write_documents([document])

    finally:
        file.file.close()
