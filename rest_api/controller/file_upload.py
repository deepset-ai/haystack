import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile, File, Form

from rest_api.config import DB_HOST, DB_PORT, DB_USER, DB_PW, DB_INDEX, DB_INDEX_FEEDBACK, ES_CONN_SCHEME, TEXT_FIELD_NAME, \
    SEARCH_FIELD_NAME, FILE_UPLOAD_PATH, EMBEDDING_DIM, EMBEDDING_FIELD_NAME, EXCLUDE_META_DATA_FIELDS, VALID_LANGUAGES, \
    FAQ_QUESTION_FIELD_NAME, REMOVE_NUMERIC_TABLES, REMOVE_WHITESPACE, REMOVE_EMPTY_LINES, REMOVE_HEADER_FOOTER, \
    CREATE_INDEX, UPDATE_EXISTING_DOCUMENTS, VECTOR_SIMILARITY_METRIC
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.file_converter.pdf import PDFToTextConverter
from haystack.file_converter.txt import TextConverter


logger = logging.getLogger(__name__)
router = APIRouter()


document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    label_index=DB_INDEX_FEEDBACK,
    scheme=ES_CONN_SCHEME,
    ca_certs=False,
    verify_certs=False,
    text_field=TEXT_FIELD_NAME,
    search_fields=SEARCH_FIELD_NAME,
    embedding_dim=EMBEDDING_DIM,
    embedding_field=EMBEDDING_FIELD_NAME,
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,  # type: ignore
    faq_question_field=FAQ_QUESTION_FIELD_NAME,
    create_index=CREATE_INDEX,
    update_existing_documents=UPDATE_EXISTING_DOCUMENTS,
    similarity=VECTOR_SIMILARITY_METRIC
)

os.makedirs(FILE_UPLOAD_PATH, exist_ok=True)  # create directory for uploading files


@router.post("/file-upload")
def upload_file_to_document_store(
    file: UploadFile = File(...),
    remove_numeric_tables: Optional[bool] = Form(REMOVE_NUMERIC_TABLES),
    remove_whitespace: Optional[bool] = Form(REMOVE_WHITESPACE),
    remove_empty_lines: Optional[bool] = Form(REMOVE_EMPTY_LINES),
    remove_header_footer: Optional[bool] = Form(REMOVE_HEADER_FOOTER),
    valid_languages: Optional[List[str]] = Form(VALID_LANGUAGES),
):
    try:
        file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.split(".")[-1].lower() == "pdf":
            pdf_converter = PDFToTextConverter(
                remove_numeric_tables=remove_numeric_tables,
                remove_whitespace=remove_whitespace,
                remove_empty_lines=remove_empty_lines,
                remove_header_footer=remove_header_footer,
                valid_languages=valid_languages,
            )
            document = pdf_converter.convert(file_path)
        elif file.filename.split(".")[-1].lower() == "txt":
            txt_converter = TextConverter(
                remove_numeric_tables=remove_numeric_tables,
                remove_whitespace=remove_whitespace,
                remove_empty_lines=remove_empty_lines,
                remove_header_footer=remove_header_footer,
                valid_languages=valid_languages,
            )
            document = txt_converter.convert(file_path)
        else:
            raise HTTPException(status_code=415, detail=f"Only .pdf and .txt file formats are supported.")

        document_to_write = {TEXT_FIELD_NAME: document["text"], "name": file.filename}
        document_store.write_documents([document_to_write])
        return "File upload was successful."
    finally:
        file.file.close()
