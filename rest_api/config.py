import ast
import os

# FastAPI
PROJECT_NAME = os.getenv("PROJECT_NAME", "FastAPI")

# Resources / Computation
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
GPU_NUMBER = int(os.getenv("GPU_NUMBER", 1))
MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 0))
BATCHSIZE = int(os.getenv("BATCHSIZE", 50))
CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", 4))

# DB
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 9200))
DB_USER = os.getenv("DB_USER", "")
DB_PW = os.getenv("DB_PW", "")
DB_INDEX = os.getenv("DB_INDEX", "document")
DB_INDEX_FEEDBACK = os.getenv("DB_INDEX_FEEDBACK", "label")
ES_CONN_SCHEME = os.getenv("ES_CONN_SCHEME", "http")
TEXT_FIELD_NAME = os.getenv("TEXT_FIELD_NAME", "text")
NAME_FIELD_NAME = os.getenv("NAME_FIELD_NAME", "name")
SEARCH_FIELD_NAME = os.getenv("SEARCH_FIELD_NAME", "text")
FAQ_QUESTION_FIELD_NAME = os.getenv("FAQ_QUESTION_FIELD_NAME", "question")
EMBEDDING_FIELD_NAME = os.getenv("EMBEDDING_FIELD_NAME", "embedding")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
VECTOR_SIMILARITY_METRIC = os.getenv("VECTOR_SIMILARITY_METRIC", "dot_product")
CREATE_INDEX = os.getenv("CREATE_INDEX", "True").lower() == "true"
UPDATE_EXISTING_DOCUMENTS = os.getenv("UPDATE_EXISTING_DOCUMENTS", "False").lower() == "true"

# Reader
READER_MODEL_PATH = os.getenv("READER_MODEL_PATH", "deepset/roberta-base-squad2")
READER_TYPE = os.getenv("READER_TYPE", "FARMReader") # alternative: 'TransformersReader'
READER_TOKENIZER = os.getenv("READER_TOKENIZER", None)
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", 500))
DEFAULT_TOP_K_READER = int(os.getenv("DEFAULT_TOP_K_READER", 5))
TOP_K_PER_CANDIDATE = int(os.getenv("TOP_K_PER_CANDIDATE", 3))
NO_ANS_BOOST = int(os.getenv("NO_ANS_BOOST", -10))
READER_CAN_HAVE_NO_ANSWER = os.getenv("READER_CAN_HAVE_NO_ANSWER", "True").lower() == "true"
DOC_STRIDE = int(os.getenv("DOC_STRIDE", 128))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 256))

# Retriever
RETRIEVER_TYPE = os.getenv("RETRIEVER_TYPE", "ElasticsearchRetriever") # alternatives: 'EmbeddingRetriever', 'ElasticsearchRetriever', 'ElasticsearchFilterOnlyRetriever', None
DEFAULT_TOP_K_RETRIEVER = int(os.getenv("DEFAULT_TOP_K_RETRIEVER", 5))
EXCLUDE_META_DATA_FIELDS = os.getenv("EXCLUDE_META_DATA_FIELDS", f"['question_emb','embedding']")
if EXCLUDE_META_DATA_FIELDS:
    EXCLUDE_META_DATA_FIELDS = ast.literal_eval(EXCLUDE_META_DATA_FIELDS)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "deepset/sentence_bert")
EMBEDDING_MODEL_FORMAT = os.getenv("EMBEDDING_MODEL_FORMAT", "farm")

# File uploads
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", "file-uploads")
REMOVE_NUMERIC_TABLES = os.getenv("REMOVE_NUMERIC_TABLES", "True").lower() == "true"
REMOVE_WHITESPACE = os.getenv("REMOVE_WHITESPACE", "True").lower() == "true"
REMOVE_EMPTY_LINES = os.getenv("REMOVE_EMPTY_LINES", "True").lower() == "true"
REMOVE_HEADER_FOOTER = os.getenv("REMOVE_HEADER_FOOTER", "True").lower() == "true"
VALID_LANGUAGES = os.getenv("VALID_LANGUAGES", None)
if VALID_LANGUAGES:
    VALID_LANGUAGES = ast.literal_eval(VALID_LANGUAGES)

# Monitoring
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
APM_SERVER = os.getenv("APM_SERVER", None)
APM_SERVICE_NAME = os.getenv("APM_SERVICE_NAME", "haystack-backend")
