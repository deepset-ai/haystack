import ast
import os

# Resources / Computation
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 4))
BATCHSIZE = int(os.getenv("BATCHSIZE", 50))
CONCURRENT_REQUEST_PER_WORKER = int(os.getenv("CONCURRENT_REQUEST_PER_WORKER", 4))

# DB
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "")
DB_PW = os.getenv("DB_PW", "")
DB_INDEX = os.getenv("DB_INDEX", "document")
DB_INDEX_FEEDBACK = os.getenv("DB_INDEX_FEEDBACK", "feedback")
ES_CONN_SCHEME = os.getenv("ES_CONN_SCHEME", "http")
TEXT_FIELD_NAME = os.getenv("TEXT_FIELD_NAME", "text")
SEARCH_FIELD_NAME = os.getenv("SEARCH_FIELD_NAME", "text")
EMBEDDING_FIELD_NAME = os.getenv("EMBEDDING_FIELD_NAME", None)
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM", None)

# Reader
READER_MODEL_PATH = os.getenv("READER_MODEL_PATH", None)
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", 500))
DEFAULT_TOP_K_READER = int(os.getenv("DEFAULT_TOP_K_READER", 5))
TOP_K_PER_CANDIDATE = int(os.getenv("TOP_K_PER_CANDIDATE", 3))
NO_ANS_BOOST = int(os.getenv("NO_ANS_BOOST", -10))
DOC_STRIDE = int(os.getenv("DOC_STRIDE", 128))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 256))

# Retriever
DEFAULT_TOP_K_RETRIEVER = int(os.getenv("DEFAULT_TOP_K_RETRIEVER", 10))
EXCLUDE_META_DATA_FIELDS = os.getenv("EXCLUDE_META_DATA_FIELDS", None)
if EXCLUDE_META_DATA_FIELDS:
    EXCLUDE_META_DATA_FIELDS = ast.literal_eval(EXCLUDE_META_DATA_FIELDS)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", None)

# Monitoring
APM_SERVER = os.getenv("APM_SERVER", None)
APM_SERVICE_NAME = os.getenv("APM_SERVICE_NAME", "haystack-backend")
